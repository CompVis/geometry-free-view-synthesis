import os, math
import torch
import time
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange

from geofree.main import instantiate_from_config

from geofree.modules.warp.midas import Midas
from geofree.modules.util import to_rgb


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class GeoTransformer(pl.LightningModule):
    """This one provides camera, depth and src as conditioning to transformer
    and makes sure that camera translation and depth are compatible.
    By setting merge_channels!=None, it merges depth and src into a single
    embedding to reduce/keep overall codelength.
    Four stages:
    first stage to encode dst
    cond stage to encode src.
    Then, scaled depth is inferred from src,
    this depth and camera translation t are normalized to make them
    consistent,
    normalized depth is encoded and normalized camera
    parameters are embedded.
    """
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 depth_stage_config,
                 merge_channels=None,
                 use_depth=True,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 use_scheduler=False,
                 scheduler_config=None,
                 monitor="val/loss",
                 pkeep=1.0,
                 plot_cond_stage=False,
                 log_det_sample=False,
                 manipulate_latents=False,
                 emb_stage_config=None,
                 emb_stage_key="camera",
                 emb_stage_trainable=True,
                 top_p=None,
                 top_k=None
                 ):

        super().__init__()
        if monitor is not None:
            self.monitor = monitor
        self.log_det_sample = log_det_sample
        self.manipulate_latents = manipulate_latents
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        self.init_depth_stage_from_ckpt(depth_stage_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        self.merge_channels = merge_channels
        if self.merge_channels is not None:
            self.merge_conv = torch.nn.Conv2d(self.merge_channels,
                                              self.transformer.config.n_embd,
                                              kernel_size=1,
                                              padding=0,
                                              bias=False)

        self.use_depth = use_depth
        if not self.use_depth:
            assert self.merge_channels is None

        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

        self.use_scheduler = use_scheduler
        if use_scheduler:
            assert scheduler_config is not None
            self.scheduler_config = scheduler_config
        self.plot_cond_stage = plot_cond_stage
        self.emb_stage_key = emb_stage_key
        self.emb_stage_trainable = emb_stage_trainable and emb_stage_config is not None
        self.init_emb_stage_from_ckpt(emb_stage_config)
        self.top_p = top_p if top_p is not None else 0.95
        try:
            tk = self.first_stage_model.quantize.n_e
        except:
            tk = 100
        self.top_k = top_k if top_k is not None else tk

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self._midas = Midas()
        self._midas.eval()
        self._midas.train = disabled_train

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing keys and {len(unexpected)} unexpected keys.")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train

    def init_depth_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        self.depth_stage_model = model.eval()
        self.depth_stage_model.train = disabled_train

    def init_emb_stage_from_ckpt(self, config):
        if config is None:
            self.emb_stage_model = None
        else:
            model = instantiate_from_config(config)
            self.emb_stage_model = model
            if not self.emb_stage_trainable:
                self.emb_stage_model.eval()
                self.emb_stage_model.train = disabled_train

    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, _, info = self.first_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        return quant_z, indices

    @torch.no_grad()
    def encode_to_c(self, c):
        quant_c, _, info = self.cond_stage_model.encode(c)
        indices = info[2].view(quant_c.shape[0], -1)
        return quant_c, indices

    @torch.no_grad()
    def encode_to_d(self, x):
        quant_z, _, info = self.depth_stage_model.encode(x)
        indices = info[2].view(quant_z.shape[0], -1)
        return quant_z, indices

    def encode_to_e(self, **kwargs):
        return self.emb_stage_model(**kwargs)

    @torch.no_grad()
    def get_layers(self, batch, xce=None):
        # interface for layer probing
        if xce is None:
            x, c, e = self.get_xce(batch)
        else:
            x, c, e = xce
        quant_z, z_indices = self.encode_to_z(**x)
        _, _, dc_indices, embeddings = self.get_normalized_c(c, e)
        cz_indices = torch.cat((dc_indices, z_indices), dim=1)

        trafo_layers = self.transformer(cz_indices[:, :-1],
                                  embeddings=embeddings,
                                  return_layers=True)
        layers = [quant_z] + trafo_layers
        return layers

    def get_normalized_d_indices(self, batch, to_device=False):
        # interface for layer probing
        _, cdict, edict = self.get_xce(batch)
        if to_device:
            for k in cdict:
                cdict[k] = cdict[k].to(device=self.device)
            for k in edict:
                edict[k] = edict[k].to(device=self.device)
        return self.get_normalized_c(cdict, edict, return_depth_only=True)

    def get_normalized_c(self, cdict, edict, return_depth_only=False,
                         fixed_scale=False):
        with torch.no_grad():
            quant_c, c_indices = self.encode_to_c(**cdict)
            if not fixed_scale:
                assert False # TODO debugging
                scaled_idepth = self._midas.scaled_depth(cdict["c"],
                                                         edict.pop("points"),
                                                         return_inverse_depth=True)
            else:
                scaled_idepth = self._midas.fixed_scale_depth(cdict["c"],
                                                              return_inverse_depth=True)
            alpha = scaled_idepth.amax(dim=(1,2))
            scaled_idepth = scaled_idepth/alpha[:,None,None]
            edict["t"] = edict["t"]*alpha[:,None]
            quant_d, d_indices = self.encode_to_d(scaled_idepth[:,None,:,:]*2.0-1.0)

        if return_depth_only:
            return d_indices, quant_d, scaled_idepth[:,None,:,:]*2.0-1.0
        embeddings = self.encode_to_e(**edict)

        if self.merge_channels is None:
            # concat depth and src indices into 2*h*w conditioning indices
            if self.use_depth:
                dc_indices = torch.cat((d_indices, c_indices), dim=1)
            else:
                dc_indices = c_indices
        else:
            # use empty conditioning indices and compute h*w conditioning
            # embeddings
            dc_indices = torch.zeros_like(d_indices)[:,[]]
            merge = torch.cat((quant_d, quant_c), dim=1)
            merge = self.merge_conv(merge)
            merge = merge.permute(0,2,3,1) # to b,h,w,c
            merge = merge.reshape(merge.shape[0],
                                  merge.shape[1]*merge.shape[2],
                                  merge.shape[3]) # to b,hw,c
            embeddings = torch.cat((embeddings,merge), dim=1)

        # check that unmasking is correct
        total_cond_length = embeddings.shape[1] + dc_indices.shape[1]
        assert total_cond_length == self.transformer.config.n_unmasked, (
            embeddings.shape[1], dc_indices.shape[1], self.transformer.config.n_unmasked)

        return quant_d, quant_c, dc_indices, embeddings

    def forward(self, xdict, cdict, edict):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(**xdict)
        _, _, dc_indices, embeddings = self.get_normalized_c(cdict, edict)
        cz_indices = torch.cat((dc_indices, z_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1], embeddings=embeddings)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, embeddings.shape[1]+dc_indices.shape[1]-1:]

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None, embeddings=None, **kwargs):
        # in the current variant we always use embeddings for camera
        assert embeddings is not None
        # check n_unmasked and conditioning length
        total_cond_length = embeddings.shape[1] + c.shape[1]
        assert total_cond_length == self.transformer.config.n_unmasked, (
            embeddings.shape[1], c.shape[1], self.transformer.config.n_unmasked)

        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        for k in range(steps):
            callback(k)
            assert x.size(1) <= block_size  # make sure model can see conditioning
            # do not crop as this messes with n_unmasked
            #x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
            x_cond = x
            logits, _ = self.transformer(x_cond, embeddings=embeddings)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
        # cut off conditioning
        x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        x = self.first_stage_model.decode(quant_z)
        return x

    @torch.no_grad()
    def log_images(self,
                   batch,
                   temperature=None,
                   top_k=None,
                   callback=None,
                   N=4,
                   half_sample=True,
                   sample=True,
                   det_sample=None,
                   top_p=None,
                   entropy=False,
                   **kwargs):
        det_sample = det_sample if det_sample is not None else self.log_det_sample
        log = dict()
        xdict, cdict, edict = self.get_xce(batch, N)
        for k in xdict:
            xdict[k] = xdict[k].to(device=self.device)
        for k in cdict:
            cdict[k] = cdict[k].to(device=self.device)
        for k in edict:
            edict[k] = edict[k].to(device=self.device)

        log["inputs"] = xdict["x"]
        log["conditioning"] = cdict["c"]

        quant_z, z_indices = self.encode_to_z(**xdict)
        quant_d, quant_c, dc_indices, embeddings = self.get_normalized_c(cdict,edict)

        if half_sample:
            # create a "half"" sample
            z_start_indices = z_indices[:,:z_indices.shape[1]//2]
            index_sample = self.sample(z_start_indices, dc_indices,
                                       steps=z_indices.shape[1]-z_start_indices.shape[1],
                                       temperature=temperature if temperature is not None else 1.0,
                                       sample=True,
                                       top_k=top_k if top_k is not None else self.top_k,
                                       top_p=top_p if top_p is not None else self.top_p,
                                       callback=callback if callback is not None else lambda k: None,
                                       embeddings=embeddings)
            x_sample = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_half"] = x_sample

        if sample:
            # sample
            z_start_indices = z_indices[:, :0]
            t1 = time.time()
            index_sample = self.sample(z_start_indices, dc_indices,
                                       steps=z_indices.shape[1],
                                       temperature=temperature if temperature is not None else 1.0,
                                       sample=True,
                                       top_k=top_k if top_k is not None else 100,
                                       callback=callback if callback is not None else lambda k: None,
                                       embeddings=embeddings)
            if not hasattr(self, "sampling_time"):
                self.sampling_time = time.time() - t1
                print(f"Full sampling takes about {self.sampling_time:.2f} seconds.")

            x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_nopix"] = x_sample_nopix

        if det_sample:
            # det sample
            z_start_indices = z_indices[:, :0]
            index_sample = self.sample(z_start_indices, dc_indices,
                                       steps=z_indices.shape[1],
                                       sample=False,
                                       callback=callback if callback is not None else lambda k: None,
                                       embeddings=embeddings)
            x_sample_det = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_det"] = x_sample_det

        if entropy:
            assert sample
            H, W = x_sample_nopix.shape[2], x_sample_nopix.shape[3]
            # plot entropy and spatial loss, (i) on datapoints and (ii) on sample
            # on data first
            targets = z_indices
            cz_indices = torch.cat((dc_indices, z_indices), dim=1)
            logits, _ = self.transformer(cz_indices[:, :-1], embeddings=embeddings)
            # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
            logits = logits[:, embeddings.shape[1] + dc_indices.shape[1] - 1:]
            h, w = quant_z.shape[2], quant_z.shape[3]
            # to spatial
            logits = rearrange(logits, 'b (h w) d -> b d h w', h=h)
            targets = rearrange(targets, 'b (h w) -> b h w', h=h)
            spatial_loss = F.cross_entropy(logits, targets, reduction="none")#[:, None, ...]
            log["spatial_loss_data"] = spatial_loss

            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
            entropy_up = F.interpolate(entropy[:,None,...], size=(H, W), mode="bicubic")
            log["spatial_entropy_data"] = entropy
            log["spatial_entropy_data_upsampled"] = entropy_up[:,0,...]

            # now on sample
            targets = index_sample
            cz_indices = torch.cat((dc_indices, index_sample), dim=1)
            logits, _ = self.transformer(cz_indices[:, :-1], embeddings=embeddings)
            # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
            logits = logits[:, embeddings.shape[1] + dc_indices.shape[1] - 1:]
            h, w = quant_z.shape[2], quant_z.shape[3]

            # to spatial
            logits = rearrange(logits, 'b (h w) d -> b d h w', h=h)
            targets = rearrange(targets, 'b (h w) -> b h w', h=h)
            spatial_loss = F.cross_entropy(logits, targets, reduction="none")#[:, None, ...]
            log["spatial_loss_sample"] = spatial_loss
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(1)
            entropy_up = F.interpolate(entropy[:,None,...], size=(H, W), mode="bicubic")
            log["spatial_entropy_sample"] = entropy
            log["spatial_entropy_sample_upsampled"] = entropy_up[:, 0, ...]

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)
        log["reconstructions"] = x_rec

        if self.plot_cond_stage:
            cond_rec = self.cond_stage_model.decode(quant_c)
            log["conditioning_rec"] = cond_rec
            depth_rec = self.depth_stage_model.decode(quant_d)
            log["depth_rec"] = depth_rec

        return log

    def get_input(self, key, batch, heuristics=True):
        x = batch[key]
        if heuristics:
            if len(x.shape) == 3:
                x = x[..., None]
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
            if x.dtype == torch.double:
                x = x.float()
        return x

    def get_xce(self, batch, N=None):
        xdict = dict()
        for k, v in self.first_stage_key.items():
            xdict[k] = self.get_input(v, batch, heuristics=k=="x")[:N]

        cdict = dict()
        for k, v in self.cond_stage_key.items():
            cdict[k] = self.get_input(v, batch, heuristics=k=="c")[:N]

        edict = dict()
        for k, v in self.emb_stage_key.items():
            edict[k] = self.get_input(v, batch, heuristics=False)[:N]

        return xdict, cdict, edict

    def compute_loss(self, logits, targets, split="train"):
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, {f"{split}/loss": loss.detach()}

    def shared_step(self, batch, batch_idx):
        x, c, e = self.get_xce(batch)
        logits, target = self(x, c, e)
        return logits, target

    def training_step(self, batch, batch_idx):
        logits, target = self.shared_step(batch, batch_idx)
        loss, log_dict = self.compute_loss(logits, target, split="train")
        self.log("train/loss", loss,
                 prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, target = self.shared_step(batch, batch_idx)
        loss, log_dict = self.compute_loss(logits, target, split="val")
        self.log("val/loss", loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return log_dict

    def configure_optimizers(self):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        extra_parameters = list()
        if self.emb_stage_trainable:
            extra_parameters += list(self.emb_stage_model.parameters())
        if hasattr(self, "merge_conv"):
            extra_parameters += list(self.merge_conv.parameters())
        else:
            assert self.merge_channels is None
        optim_groups.append({"params": extra_parameters, "weight_decay": 0.0})
        print(f"Optimizing {len(extra_parameters)} extra parameters.")
        optimizer = torch.optim.AdamW(optim_groups, lr=self.learning_rate, betas=(0.9, 0.95))
        if self.use_scheduler:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [optimizer], scheduler
        return optimizer
