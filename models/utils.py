import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2D(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        std, mean = torch.std_mean(x, dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        return self.gamma * x + self.beta

# class LayerNorm2D(nn.LayerNorm):
#     def forward(self, x):
#         x = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps)
#         x = x.permute(0, 3, 1, 2)
#         return x


class GRA(nn.Module):
    def __init__(self, channels, dim=(1,)):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.dim = dim

    def forward(self, x):
        p = x ** 2
        p = p / (p.mean(dim=self.dim, keepdim=True) + 1e-6)
        return (self.gamma * p + self.beta) * x


class MGRN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, x):
        g = x.norm(p=2, dim=(2, 3), keepdim=True)
        g = g / (g.mean(dim=1, keepdim=True) + 1e-6)
        return (self.gamma * g + self.beta) * x
