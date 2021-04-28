import torch
import time
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR

from geofree.main import instantiate_from_config


class WarpTransformer(pl.LightningModule):
    """This one relies on WarpGPT, where the warper handles the warping to
    support warping of positional embeddings, too.
    first stage to encode dst
    cond stage to encode src
    """
    def __init__(self,
                 transformer_config,
                 first_stage_config,
                 cond_stage_config,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="dst_img",
                 cond_stage_key="src_img",
                 use_scheduler=False,
                 scheduler_config=None,
                 monitor="val/loss",
                 pkeep=1.0,
                 plot_cond_stage=False,
                 log_det_sample=False,
                 top_k=None
                 ):

        super().__init__()
        if monitor is not None:
            self.monitor = monitor
        self.log_det_sample = log_det_sample
        self.init_first_stage_from_ckpt(first_stage_config)
        self.init_cond_stage_from_ckpt(cond_stage_config)
        self.transformer = instantiate_from_config(config=transformer_config)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.pkeep = pkeep

        self.use_scheduler = use_scheduler
        if use_scheduler:
            assert scheduler_config is not None
            self.scheduler_config = scheduler_config
        self.plot_cond_stage = plot_cond_stage
        self.top_k = top_k if top_k is not None else 100
        self.warpkwargs_keys = {
            "x": "src_img",
            "points": "src_points",
            "R": "R_rel",
            "t": "t_rel",
            "K_dst": "K",
            "K_src_inv": "K_inv",
        }

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

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()

    def forward(self, x, c, warpkwargs):
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x)
        _, c_indices = self.encode_to_c(c)

        if self.training and self.pkeep < 1.0:
            mask = torch.bernoulli(self.pkeep*torch.ones(z_indices.shape,
                                                         device=z_indices.device))
            mask = mask.round().to(dtype=torch.int64)
            r_indices = torch.randint_like(z_indices, self.transformer.config.vocab_size)
            a_indices = mask*z_indices+(1-mask)*r_indices
        else:
            a_indices = z_indices

        cz_indices = torch.cat((c_indices, a_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        logits, _ = self.transformer(cz_indices[:, :-1], warpkwargs)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, top_k=None,
               callback=lambda k: None, warpkwargs=None, **kwargs):
        x = torch.cat((c,x),dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        for k in range(steps):
            callback(k)
            assert x.size(1) <= block_size  # make sure model can see conditioning
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
            logits, _ = self.transformer(x_cond, warpkwargs)
            # for the next steps, reuse precomputed embeddings for conditioning
            self.transformer.warper.set_cache(True)
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
        # disable caching again
        self.transformer.warper.set_cache(False)
        # cut off conditioning
        x = x[:, c.shape[1]:]
        return x

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
                   **kwargs):
        det_sample = det_sample if det_sample is not None else self.log_det_sample
        log = dict()
        x, c = self.get_xc(batch, N)
        warpkwargs = self.get_warpkwargs(batch, N)
        for k in warpkwargs:
            warpkwargs[k] = warpkwargs[k].to(device=self.device)
        x = x.to(device=self.device)
        if type(c) != list:
            c = c.to(device=self.device)

        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)

        if half_sample:
            # create a "half"" sample
            z_start_indices = z_indices[:,:z_indices.shape[1]//2]
            index_sample = self.sample(z_start_indices, c_indices,
                                       steps=z_indices.shape[1]-z_start_indices.shape[1],
                                       temperature=temperature if temperature is not None else 1.0,
                                       sample=True,
                                       top_k=top_k if top_k is not None else self.top_k,
                                       callback=callback if callback is not None else lambda k: None,
                                       warpkwargs=warpkwargs)
            x_sample = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_half"] = x_sample

        if sample:
            # sample
            z_start_indices = z_indices[:, :0]
            t1 = time.time()
            index_sample = self.sample(z_start_indices, c_indices,
                                       steps=z_indices.shape[1],
                                       temperature=temperature if temperature is not None else 1.0,
                                       sample=True,
                                       top_k=top_k if top_k is not None else 100,
                                       callback=callback if callback is not None else lambda k: None,
                                       warpkwargs=warpkwargs)
            if not hasattr(self, "sampling_time"):
                self.sampling_time = time.time() - t1
                print(f"Full sampling takes about {self.sampling_time:.2f} seconds.")

            x_sample_nopix = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_nopix"] = x_sample_nopix

        if det_sample:
            # det sample
            z_start_indices = z_indices[:, :0]
            index_sample = self.sample(z_start_indices, c_indices,
                                       steps=z_indices.shape[1],
                                       sample=False,
                                       callback=callback if callback is not None else lambda k: None,
                                       warpkwargs=warpkwargs)
            x_sample_det = self.decode_to_img(index_sample, quant_z.shape)
            log["samples_det"] = x_sample_det

        # reconstruction
        x_rec = self.decode_to_img(z_indices, quant_z.shape)

        log["inputs"] = x
        log["reconstructions"] = x_rec

        if self.plot_cond_stage:
            cond_rec = self.cond_stage_model.decode(quant_c)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c


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

    def get_xc(self, batch, N=None):
        x = self.get_input(self.first_stage_key, batch)
        c = self.get_input(self.cond_stage_key, batch)
        if N is not None:
            x = x[:N]
            c = c[:N]
        return x, c

    def get_warpkwargs(self, batch, N=None):
        kwargs = dict()
        for k, v in self.warpkwargs_keys.items():
            kwargs[k] = self.get_input(v, batch, heuristics=k=="x")[:N]
        return kwargs

    def compute_loss(self, logits, targets, split="train"):
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss, {f"{split}/loss": loss.detach()}

    def shared_step(self, batch, batch_idx):
        x, c = self.get_xc(batch)
        warpkwargs = self.get_warpkwargs(batch)
        logits, target = self(x, c, warpkwargs)
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
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if fpn.startswith("warper._midas"):
                    continue

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
        if hasattr(self.transformer.warper, "pos_emb"):
            no_decay.add('warper.pos_emb')

        # handle meta positional embeddings
        for pn, p in self.transformer.warper.named_parameters():
            if pn.endswith("pos_emb"):
                no_decay.add(f"warper.{pn}")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters() if not pn.startswith("warper._midas")}

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
