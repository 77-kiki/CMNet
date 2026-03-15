import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedMultiScaleSpatialAttention(nn.Module):
    """使用膨胀卷积的多尺度空间注意力模块"""
    def __init__(self, expand_ratio=4):
        super().__init__()
        
        # 标准3x3卷积分支（局部细节）
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(1, expand_ratio, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_ratio, 1, kernel_size=1, bias=False)
        )
        
        # 膨胀卷积分支（更大的感受野）
        self.branch_dilated = nn.Sequential(
            nn.Conv2d(1, expand_ratio, kernel_size=3, 
                      padding=2, dilation=2, bias=False),  # 等效5x5感受野
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_ratio, 1, kernel_size=1, bias=False)
        )
        
        # 融合层
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(2, expand_ratio, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_ratio, 1, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    
    def forward(self, x):
        # 输入特征图的空间压缩
        spatial_map = torch.mean(x, dim=1, keepdim=True)
        
        # 多尺度特征提取
        b3 = self.branch3x3(spatial_map)
        b_dilated = self.branch_dilated(spatial_map)
        
        # 特征融合
        fused = torch.cat([b3, b_dilated], dim=1)
        att = self.fuse_conv(fused)
        
        # 残差增强
        res = self.res_conv(spatial_map)
        return att + res
        
    

class CoordAttention(nn.Module):
    """坐标注意力机制 - 使用更小的reduction比率(16)"""
    def __init__(self, in_channels, reduction=16):  # reduction从32改为16
        super().__init__()
        reduction_channels = max(8, in_channels // reduction)
        
        # 水平方向池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 垂直方向池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # 共享的特征变换层（更多通道）
        self.conv_reduction = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels, 1, bias=False),
            nn.BatchNorm2d(reduction_channels),
            nn.ReLU(inplace=True)
        )
        
        # 量化友好的激活函数
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, x):
        identity = x
        batch, _, height, width = x.size()
        
        # 水平方向注意力
        x_h = self.pool_h(x)
        # 垂直方向注意力
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        # 连接并变换特征
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv_reduction(y)
        
        # 分离特征
        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # 生成注意力图
        att_h = self.sigmoid(self.conv_h(x_h))
        att_w = self.sigmoid(self.conv_w(x_w))
        
        return identity * att_w * att_h
    

class EfficientLocalizationAttention(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super(EfficientLocalizationAttention, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        b, c, h, w = x.size()
 
        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
 
        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)
 
        # 在两个维度上应用注意力
        return x * x_h * x_w


class DFFG(nn.Module):
    """优化版位置感知注意力模块 - 膨胀卷积+更强的通道注意力"""
    def __init__(self, in_channels, reduction=16, expand_ratio=4):  # reduction默认16
        super().__init__()
        
        # 轻量特征变换
        self.feature_transform = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish(inplace=True)
        )
        
        # 更强的坐标注意力机制（reduction=16）
        self.coord_att = CoordAttention(in_channels, reduction)
        # self.coord_att = EfficientLocalizationAttention(in_channels)
        
        # 使用膨胀卷积的空间注意力
        self.spatial_att = DilatedMultiScaleSpatialAttention(expand_ratio)
        
        # 残差比例因子（可学习）
        self.res_factor = nn.Parameter(torch.tensor(0.5))

     
    def forward(self, x):
        # 特征变换
        x = self.feature_transform(x)
        
        # 坐标注意力增强
        x = self.coord_att(x)
        
        # 多尺度空间注意力
        sa = self.spatial_att(x)
        
        # 带权重的残差连接
        return self.res_factor * x * sa + x
