from .block import C3k, Bottleneck, DropPath, Conv

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmcv.cnn import build_norm_layer
from torch import Tensor
from math import log

# ========== Sobel + Gaussian 分支保持不变 ==========
class GaussianBranch(nn.Module):
    def __init__(self, dim, k, sigma, norm_layer, act_layer):
        super().__init__()
        kernel = self.gaussian_kernel(k, sigma)
        kernel = nn.Parameter(kernel, requires_grad=False)
        self.gaussian = nn.Conv2d(dim, dim, k, padding=k // 2,
                                  groups=dim, bias=False)  # depthwise
        self.gaussian.weight.data = kernel.repeat(dim, 1, 1, 1)
        self.norm = build_norm_layer(norm_layer, dim)[1]
        self.act = act_layer()

    def gaussian_kernel(self, k, sigma):
        kernel = torch.FloatTensor([
            [(1 / (2 * math.pi * sigma ** 2)) *
             math.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
             for x in range(-k // 2 + 1, k // 2 + 1)]
            for y in range(-k // 2 + 1, k // 2 + 1)
        ]).unsqueeze(0).unsqueeze(0)
        return kernel / kernel.sum()

    def forward(self, x):
        g = self.gaussian(x)
        return self.act(self.norm(g))


def sobel_kernel():
    kx = torch.tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=torch.float32)
    ky = torch.tensor([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype=torch.float32)
    return kx.unsqueeze(0).unsqueeze(0), ky.unsqueeze(0).unsqueeze(0)


class SobelBranch(nn.Module):
    def __init__(self):
        super().__init__()
        kx, ky = sobel_kernel()
        self.kx = nn.Parameter(kx, requires_grad=False)
        self.ky = nn.Parameter(ky, requires_grad=False)
        self.refine = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        ex = F.conv2d(gray, self.kx, padding=1)
        ey = F.conv2d(gray, self.ky, padding=1)
        edge = torch.sqrt(ex ** 2 + ey ** 2 + 1e-6)
        edge = self.refine(edge)
        return edge.expand_as(x)


class SobelGaussianLite(nn.Module):
    """轻量化版 Sobel+Gaussian 融合，不再用 1×1 Conv"""
    def __init__(self, dim, k, sigma, norm_layer, act_layer):
        super().__init__()
        self.g_branch = GaussianBranch(dim, k, sigma, norm_layer, act_layer)
        self.s_branch = SobelBranch()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习权重

    def forward(self, x):
        g = self.g_branch(x)  # Gaussian 特征
        s = self.s_branch(x)  # Sobel 特征
        out = g + self.alpha * s
        return x + out  # 残差直连，不再有额外 1×1 conv


# ========== LFEA 精简 ==========
class LFEA_Lite(nn.Module):
    """轻量化 LFEA：DWConv + 简化权重生成"""
    def __init__(self, channel, norm_layer, act_layer):
        super(LFEA_Lite, self).__init__()
        # 用 DWConv 替代 3x3 全通道卷积
        self.conv2d = nn.Sequential(
            nn.Conv2d(channel, channel, 3, stride=1, padding=1,
                      groups=channel, bias=False),  # DWConv
            build_norm_layer(norm_layer, channel)[1],
            act_layer()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv1d(1, 1, kernel_size=1, bias=True)  # 代替 1D 卷积
        self.sigmoid = nn.Sigmoid()
        self.norm = build_norm_layer(norm_layer, channel)[1]

    def forward(self, c, att):
        att = self.conv2d(att)
        wei = self.avg_pool(att)  # [B,C,1,1]
        wei = self.fc(wei.squeeze(-1).transpose(-1, -2))  # [B,C,1]
        wei = self.sigmoid(wei).transpose(-1, -2).unsqueeze(-1)  # [B,C,1,1]
        x = self.norm(c + att * wei)
        return x
    
# Please refer to the manuscript for the core architectural logic.
    

class C2fG(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Initialize a CSP bottleneck with 2 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.ega = LFE_Module(self.c)

    def forward(self, x):
        """Forward pass through C2f layer."""
   
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y[0] = self.ega(y[0])
        return self.cv2(torch.cat(y, 1))   # [2, 128, 160, 160]

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y[0] = self.ega(y[0])
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class DKFE(C2fG):

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of blocks.
            c3k (bool): Whether to use C3k blocks.
            e (float): Expansion ratio.
            g (int): Groups for convolutions.
            shortcut (bool): Whether to use shortcut connections.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)
        )