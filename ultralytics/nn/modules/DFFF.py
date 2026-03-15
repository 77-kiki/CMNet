import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- SimAM (轻量空间注意力，高层特征增强) --------
class SimAM(nn.Module):
    def __init__(self, channels, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        x_mean = x.mean(dim=[2, 3], keepdim=True)
        x_centered = x - x_mean
        norm_squared = (x_centered ** 2).mean(dim=[2, 3], keepdim=True)
        attention = x_centered / (norm_squared + self.e_lambda)
        return x * torch.sigmoid(attention)


# -------- NAM (论文版) --------
class NAM(nn.Module):
    def __init__(self, channels, norm="BN"):
        super(NAM, self).__init__()
        if norm == "BN":
            self.norm = nn.BatchNorm2d(channels, affine=True)
        elif norm == "GN":
            self.norm = nn.GroupNorm(32, channels, affine=True)  # GN替代
        else:
            raise ValueError("norm must be 'BN' or 'GN'")

    def forward(self, x):
        if isinstance(self.norm, nn.BatchNorm2d):
            gamma = self.norm.weight.abs()
        # Please refer to the manuscript for the core architectural logic.
        Mc = torch.sigmoid(weight * x)
        return Mc


# -------- FAA-NAM-GAP (改进版) --------
class FAA_Module(nn.Module):
    def __init__(self, in_channels_low, in_channels_high, out_channels, reduction=16, norm="BN"):
        super().__init__()
        self.high_att = SimAM(in_channels_high)
        self.align_high = nn.Conv2d(in_channels_high, out_channels // 2, 1)
        self.align_low = nn.Conv2d(in_channels_low, out_channels // 2, 1)

        self.nam = NAM(out_channels, norm=norm)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, F):
        Fl, Fh = F[0], F[1]
        Fh_prime = self.high_att(Fh)

        Fh_prime_aligned = self.align_high(Fh_prime)
        Fl_up_aligned = self.align_low(Fl)

        FA = torch.sigmoid(Fh_prime_aligned) * Fl_up_aligned
        FlA = Fl_up_aligned + FA
        FhA = Fh_prime_aligned + FA

        fused = torch.cat([FlA, FhA], dim=1)

        # NAM 注意力
        out_nam = self.nam(fused)

        # GAP 辅助注意力
        gap_att = self.sigmoid(self.gap(fused))
        out_gap = fused * gap_att

        # 融合两种注意力 (带门控参数)
        out = self.alpha * out_nam + (1 - self.alpha) * out_gap
        return out
