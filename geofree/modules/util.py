import torch
import torch.nn as nn


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class SOSProvider(AbstractEncoder):
    # for unconditional training
    def __init__(self, sos_token, quantize_interface=True):
        super().__init__()
        self.sos_token = sos_token
        self.quantize_interface = quantize_interface

    def encode(self, x):
        # get batch size from data and replicate sos_token
        c = torch.ones(x.shape[0], 1)*self.sos_token
        c = c.long().to(x.device)
        if self.quantize_interface:
            return None, None, [None, None, c]
        return c


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


class Embedder(nn.Module):
    """to replace the convolutional architecture entirely"""
    def __init__(self, n_positions, n_channels, n_embed, bias=False):
        super().__init__()
        self.n_positions = n_positions
        self.n_channels = n_channels
        self.n_embed = n_embed
        self.fc = nn.Linear(self.n_channels, self.n_embed, bias=bias)

    def forward(self, x):
        x = x.reshape(x.shape[0], self.n_positions, self.n_channels)
        x = self.fc(x)
        return x


class MultiEmbedder(nn.Module):
    def __init__(self, keys, n_positions, n_channels, n_embed, bias=False):
        super().__init__()
        self.keys = keys
        self.n_positions = n_positions
        self.n_channels = n_channels
        self.n_embed = n_embed
        self.fc = nn.Linear(self.n_channels, self.n_embed, bias=bias)

    def forward(self, **kwargs):
        values = [kwargs[k] for k in self.keys]
        inputs = list()
        for k in self.keys:
            entry = kwargs[k].reshape(kwargs[k].shape[0], -1, self.n_channels)
            inputs.append(entry)
        x = torch.cat(inputs, dim=1)
        assert x.shape[1] == self.n_positions, x.shape
        x = self.fc(x)
        return x


class SpatialEmbedder(nn.Module):
    def __init__(self, keys, n_channels, n_embed, bias=False, shape=[13, 23]):
        # here, n_channels = dim(params)
        super().__init__()
        self.shape = shape
        self.keys = keys
        self.n_channels = n_channels
        self.n_embed = n_embed
        self.linear = nn.Conv2d(self.n_channels, self.n_embed, 1, bias=bias)

    def forward(self, **kwargs):
        inputs = list()
        for k in self.keys:
            entry = kwargs[k].reshape(kwargs[k].shape[0], -1, 1, 1)
            inputs.append(entry)
        x = torch.cat(inputs, dim=1) # b, n_channels, 1, 1
        assert x.shape[1] == self.n_channels, f"expecting {self.n_channels} channels but got {x.shape[1]}"
        x = x.repeat(1, 1, self.shape[0], self.shape[1])  # duplicate spatially
        x = self.linear(x)
        return x


def to_rgb(model, x):
    if not hasattr(model, "colorize"):
        model.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
    x = nn.functional.conv2d(x, weight=model.colorize)
    x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
    return x
