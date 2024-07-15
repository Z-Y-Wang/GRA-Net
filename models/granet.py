import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

# from .utils import LayerNorm2D, GRA, MGRN


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class GRA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        p = x ** 2
        p = p / (p.mean(dim=-1, keepdim=True) + 1e-6)
        p = torch.log2(1 + p) # optional
        return (self.gamma * p + 1) * x + self.beta


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        g = x.norm(p=2, dim=(1, 2), keepdim=True)
        g = g / (g.mean(dim=-1, keepdim=True) + 1e-6)
        return (self.gamma * g + 1) * x + self.beta


class Block(nn.Module):
    def __init__(self, dim, drop_path=0.,):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = GRA(4 * dim)
        self.grn = GRN(4 * dim)

        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # with torch.cuda.amp.autocast(enabled=False):
        x = self.dwconv(x.float())
        # print(x.max())
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x


class GRA_Net(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6)
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)


        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = self.norm(x.mean([-2, -1]))
        x = self.head(x)
        return x


def granet_atto(**kwargs):
    model = GRA_Net(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model



def granet_femto(**kwargs):
    model = GRA_Net(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def granet_pico(**kwargs):
    model = GRA_Net(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def granet_nano(**kwargs):
    model = GRA_Net(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def granet_tiny(**kwargs):
    model = GRA_Net(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def granet_base(**kwargs):
    model = GRA_Net(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model
