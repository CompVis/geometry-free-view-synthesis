import torch
import torch.nn as nn
from einops import rearrange

from geofree.modules.warp.midas import Midas


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class AbstractWarper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._midas = Midas()
        self._midas.eval()
        self._midas.train = disabled_train
        for param in self._midas.parameters():
            param.requires_grad = False

        self.n_unmasked = kwargs["n_unmasked"] # length of conditioning
        self.n_embd = kwargs["n_embd"]
        self.block_size = kwargs["block_size"]
        self.size = kwargs["size"]  # h, w tuple
        self.start_idx = kwargs.get("start_idx", 0) # hint to not modify parts

        self._use_cache = False
        self.new_emb = None # cache
        self.new_pos = None # cache

    def set_cache(self, value):
        self._use_cache = value

    def get_embeddings(self, token_embeddings, position_embeddings, warpkwargs):
        if self._use_cache:
            assert not self.training, "Do you really want to use caching during training?"
            assert self.new_emb is not None
            assert self.new_pos is not None
            return self.new_emb, self.new_pos
        self.new_emb, self.new_pos = self._get_embeddings(token_embeddings,
                                                          position_embeddings,
                                                          warpkwargs)
        return self.new_emb, self.new_pos

    def _get_embeddings(self, token_embeddings, position_embeddings, warpkwargs):
        raise NotImplementedError()

    def forward(self, token_embeddings, position_embeddings, warpkwargs):
        new_emb, new_pos = self.get_embeddings(token_embeddings,
                                               position_embeddings,
                                               warpkwargs)

        new_emb = torch.cat([new_emb, token_embeddings[:,self.n_unmasked:,:]],
                            dim=1)
        b = new_pos.shape[0]
        new_pos = torch.cat([new_pos, position_embeddings[:,self.n_unmasked:,:][b*[0],...]],
                            dim=1)

        return new_emb, new_pos

    def _to_sequence(self, x):
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x

    def _to_imglike(self, x):
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.size[0])
        return x


class AbstractWarperWithCustomEmbedding(AbstractWarper):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, self.block_size, self.n_embd))


class NoSourceWarper(AbstractWarper):
    def _get_embeddings(self, token_embeddings, position_embeddings, warpkwargs):
        cond_emb = token_embeddings[:,:self.n_unmasked,:]
        cond_pos = position_embeddings[:,:self.n_unmasked,:]

        b, seq_length, chn = cond_emb.shape
        cond_emb = self._to_imglike(cond_emb)

        cond_pos = self._to_imglike(cond_pos)
        cond_pos = cond_pos[b*[0],...]

        new_emb, _ = self._midas.warp_features(f=cond_emb, no_depth_grad=True,
                                               boltzmann_factor=0.0,
                                               **warpkwargs)
        new_pos, _ = self._midas.warp_features(f=cond_pos, no_depth_grad=True,
                                               boltzmann_factor=0.0,
                                               **warpkwargs)
        new_emb = self._filter_nans(new_emb)
        new_pos = self._filter_nans(new_pos)

        new_emb = self._to_sequence(new_emb)
        new_pos = self._to_sequence(new_pos)
        return new_emb, new_pos

    def _filter_nans(self, x):
        x[torch.isnan(x)] = 0.
        return x


class ConvWarper(AbstractWarper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emb_conv = nn.Conv2d(2*self.n_embd, self.n_embd,
                                  kernel_size=1,
                                  padding=0,
                                  bias=False)
        self.pos_conv = nn.Conv2d(2*self.n_embd, self.n_embd,
                                  kernel_size=1,
                                  padding=0,
                                  bias=False)

    def _get_embeddings(self, token_embeddings, position_embeddings, warpkwargs):
        cond_emb = token_embeddings[:,self.start_idx:self.n_unmasked,:]
        cond_pos = position_embeddings[:,self.start_idx:self.n_unmasked,:]

        b, seq_length, chn = cond_emb.shape
        cond_emb = cond_emb.reshape(b, self.size[0], self.size[1], chn)
        cond_emb = cond_emb.permute(0,3,1,2)

        cond_pos = cond_pos.reshape(1, self.size[0], self.size[1], chn)
        cond_pos = cond_pos.permute(0,3,1,2)
        cond_pos = cond_pos[b*[0],...]

        with torch.no_grad():
            warp_emb, _ = self._midas.warp_features(f=cond_emb, no_depth_grad=True, **warpkwargs)
            warp_pos, _ = self._midas.warp_features(f=cond_pos, no_depth_grad=True, **warpkwargs)

        new_emb = self.emb_conv(torch.cat([cond_emb, warp_emb], dim=1))
        new_pos = self.pos_conv(torch.cat([cond_pos, warp_pos], dim=1))

        new_emb = new_emb.permute(0,2,3,1)
        new_emb = new_emb.reshape(b,seq_length,chn)

        new_pos = new_pos.permute(0,2,3,1)
        new_pos = new_pos.reshape(b,seq_length,chn)

        # prepend unmodified ones again
        new_emb = torch.cat((token_embeddings[:,:self.start_idx,:], new_emb),
                            dim=1)
        new_pos = torch.cat((position_embeddings[:,:self.start_idx,:][b*[0],...], new_pos),
                            dim=1)

        return new_emb, new_pos
