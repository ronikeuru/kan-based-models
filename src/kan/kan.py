import math

import torch
import torch.nn as nn

import torch.nn.functional as F

from .utils.splines import b_splines
from .utils.curve2coeff import curve2coeff

class KANLayer(nn.Module):
    def __init__(
            self,
            in_dim,
            out_dim,
            grid_size=5,
            k=3,
            noise_scale=0.1,
            noise_scale_base=0.1,
            scale_spline=None,
            base_fun=nn.SiLU(),
            grid_range=[-1, 1],
            sp_trainable=True,
            sb_trainable=True,
            device="cpu",
        ):
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.k = k
        self.base_fun = base_fun

        step = (grid_range[1] - grid_range[0]) / grid_size
        points = torch.arange(-k, grid_size + k + 1, device=device)
        grid = (points * step + grid_range[0]).repeat(self.in_dim, 1)
        self.register_buffer("grid", grid)

        if scale_spline is not None:
            self.scale_spline = nn.Parameter(
                torch.full((out_dim, in_dim), fill_value=scale_spline, device=device),
                requires_grad=sp_trainable,
            )
        else:
            self.register_buffer("scale_spline", torch.Tensor([1.0], device=device))

        rand_noise = torch.rand(grid_size + 1, in_dim, out_dim, device=device) - 1 / 2
        noise = rand_noise * noise_scale / self.grid_size
        self.coeff = nn.Parameter(curve2coeff(self.grid.T[k:-k], noise, self.grid, k).contiguous())

        self.scale_base = nn.Parameter(
            (
                1 / (in_dim**0.5)
                + (torch.randn(self.out_dim, self.in_dim, device=device) * 2 - 1)
                * noise_scale_base
            ),
            requires_grad=sb_trainable,
        )

    def forward(self, x: torch.Tensor):
        shape = x.shape[:-1]
        x = x.view(-1, self.in_dim)

        splines = b_splines(x, self.grid, self.k)

        batch_size = x.shape[0]
        y_b = F.linear(self.base_fun(x), self.scale_base)

        y_spline = F.linear(
            splines.view(batch_size, -1),
            (self.coeff * self.scale_spline.unsqueeze(-1)).view(self.out_dim, -1),
        )

        # /sigma(x) = w(b(x) + spline(x))
        y = y_b + y_spline
        y = y.view(*shape, self.out_dim)

        return y

class KAN(nn.Module):
    def __init__(
        self,
        layer_dims=[3, 2],
        grid=5,
        k=3,
        noise_scale=0.1,
        scale_spline=1.0,
        base_fun=nn.SiLU(),
        grid_range=[-1, 1],
        sp_trainable=True,
        device="cpu",
    ):
        super(KAN, self).__init__()
        
        self.layers = nn.ModuleList()

        for in_dim, out_dim in zip(layer_dims, layer_dims[1:]):
            self.layers.append(
                KANLayer(
                    in_dim,
                    out_dim,
                    grid_size=grid,
                    k=k,
                    noise_scale=noise_scale,
                    scale_spline=scale_spline,
                    base_fun=base_fun,
                    grid_range=grid_range,
                    sp_trainable=sp_trainable,
                    device=device,
                )
            )

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x

class KANConv2d(nn.Module):
    def __init__(
        self,
        in_chan,
        out_chan,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        device="cpu",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.kernels = nn.ModuleList()
        for _ in range(in_chan):
            self.kernels.append(KANLayer(kernel_size * kernel_size, out_chan, device=device))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_chan), requires_grad=True)
        else:
            self.bias = bias

        self.unfold = nn.Unfold(
            kernel_size,
            dilation,
            padding,
            stride,
        )

    def forward(self, x):
        shape = x.shape

        x = x.view(-1, shape[-3], shape[-2], shape[-1])
        x = self.unfold(x).permute(0, 2, 1).view(
            x.shape[0],
            -1,
            self.in_chan,
            self.kernel_size**2,
        )
        x = torch.stack(
            [
                kernel(x[:, :, i, :].contiguous())
                for i, kernel in enumerate(self.kernels)],
            dim=2,
        ).sum(dim=2)

        if self.bias is not False:
            x = x + self.bias[None, None, :]

        shift = 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1
        h = math.floor((shape[-2] + shift) / self.stride + 1)
        w = math.floor((shape[-1] + shift) / self.stride + 1)

        x = x.permute(0, 2, 1).view(*shape[:-3], self.out_chan, h, w)

        return x


class KAN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            KANConv2d(1, 4, 5, 1, 2),
            nn.MaxPool2d(2),
            KANConv2d(4, 16, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            KANLayer(16 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.layers(x)
