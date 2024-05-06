import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from utils import LayerNorm, GlobalResponseNormalization


class AdaptiveWeightModule(nn.Module):
    def __init__(self, C, K):
        super(AdaptiveWeightModule, self).__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(C, C // 4, 1)  # replace nn.linear by 1x1conv
        self.ln = LayerNorm(C // 4, eps=1e-6, data_format="channels_first")
        self.conv2 = nn.Conv2d(C // 4, C * K * K, 1)  # same replacement
        self.K = K

    def forward(self, x):
        B, C, _, _ = x.shape
        x = self.adaptive_pool(x)
        x = F.relu(self.conv1(x))
        x = self.ln(x)
        x = F.relu(self.conv2(x))
        x = x.view(B* C, 1, self.K, self.K)  # reshape shape of depth-wise kernel B*C, 1, K, K
        return x
    
    def flops(self, B, H, W):
        flops = 0
        # Adaptive Pool
        flops += B * self.conv1.in_channels * H * W
        # conv1
        flops += B * self.conv1.in_channels * self.conv1.out_channels
        # conv2
        flops += B * self.conv2.in_channels * self.conv2.out_channels
        return flops
    
class DynamicDepthwiseConv2d(nn.Module):
    def __init__(self, dim, kernel_size, bias=True, stride=1, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = self.kernel_size // 2
        self.adaptive_weight_module = AdaptiveWeightModule(dim, kernel_size)  # Initialize AdaptiveWeightModule
        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.adaptive_weight_module(x) # [B*C, 1, K, K]
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * c)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x
    
    def flops(self, B, C, H, W):
        flops = 0
        # Adaptive weight module flops
        flops += self.adaptive_weight_module.flops(B, H, W)
        # Depthwise convolution
        kernel_ops = C * self.kernel_size * self.kernel_size
        output_elements = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        output_elements *= (W - self.kernel_size + 2 * self.padding) // self.stride + 1
        flops += kernel_ops * output_elements * B
        return flops
    
# class DynamicDepthwiseConv2d(nn.Module):
#     def __init__(self, K):
#         super(DynamicDepthwiseConv2d, self).__init__()
#         # initialize kernel size
#         self.K = K

#     def forward(self, x):
#         B, C, H, W = x.shape  # acquire B,C,H,W from input
#         # generate Depthwise Conv Weight
#         adaptive_weight_module = AdaptiveWeightModule(C, self.K)
#         weight = adaptive_weight_module(x)  # [B*C, 1, K, K]

#         # Reshape input
#         x = x.view(1, B * C, H, W)
        
#         # apply generated depth-wise weight
#         padding = self.K // 2
#         out = F.conv2d(x, weight, None, stride=1, padding=padding, groups=B * C)

#         # Reshape output size
#         out = out.view(B, C, H, W)
#         return out
    
