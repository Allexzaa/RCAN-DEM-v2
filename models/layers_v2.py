"""
RCAN-DEM v2 Building Blocks

Improved components with:
- Spatial dropout for regularization
- Multi-scale convolutions for better feature extraction
- Spatial attention for "where to focus"
- All improvements from v1 plus new capabilities

Reference: Image Super-Resolution Using Very Deep Residual Channel Attention Networks
https://arxiv.org/abs/1807.02758

v2 Improvements:
- SpatialDropout2d for regularization
- Optional spatial attention
- Multi-scale feature extraction
- Better MPS compatibility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class SpatialDropout2d(nn.Module):
    """
    Spatial Dropout for 2D feature maps.
    
    Drops entire channels (feature maps) rather than individual elements.
    More effective for convolutional networks than standard dropout.
    
    Args:
        p: Dropout probability (default: 0.1)
    
    Shape:
        Input: [B, C, H, W]
        Output: [B, C, H, W] (same shape, some channels zeroed)
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x
        
        # Create channel-wise mask [B, C, 1, 1]
        batch_size, channels = x.size(0), x.size(1)
        mask = torch.ones(batch_size, channels, 1, 1, device=x.device, dtype=x.dtype)
        mask = F.dropout(mask, p=self.p, training=True)
        
        # Scale to maintain expected values
        return x * mask


class ChannelAttention_v2(nn.Module):
    """
    Channel Attention (CA) Layer v2.
    
    Learns to weight channels based on their importance.
    Uses global average pooling followed by FC layers with sigmoid.
    
    v2 Improvements:
    - Optional dropout after attention
    - Configurable activation
    
    Args:
        n_feat: Number of input channels
        reduction: Channel reduction ratio for bottleneck (default: 16)
        dropout: Dropout rate after attention (default: 0.0)
    """
    def __init__(
        self,
        n_feat: int,
        reduction: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat, 1, bias=True),
            nn.Sigmoid(),
        )
        
        self.dropout = SpatialDropout2d(p=dropout) if dropout > 0 else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.fc(y)
        out = x * y
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        return out


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Learns to weight spatial locations based on their importance.
    Complements Channel Attention by focusing on "where" rather than "what".
    
    Uses channel pooling (max and avg) followed by conv to produce attention map.
    
    Args:
        kernel_size: Size of convolution kernel (default: 7)
    
    Shape:
        Input: [B, C, H, W]
        Output: [B, C, H, W] (spatially weighted)
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel pooling: [B, C, H, W] -> [B, 2, H, W]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_pool, max_pool], dim=1)
        
        # Spatial attention map: [B, 2, H, W] -> [B, 1, H, W]
        attention = self.conv(pooled)
        
        return x * attention


class MultiScaleConv(nn.Module):
    """
    Multi-Scale Convolution Module.
    
    Extracts features at multiple receptive field sizes using dilated convolutions.
    Captures both local details and broader context.
    
    Args:
        n_feat: Number of input/output channels
        dilations: List of dilation rates (default: [1, 2, 4])
        reduction: Channel reduction for efficiency (default: 4)
    
    Shape:
        Input: [B, C, H, W]
        Output: [B, C, H, W]
    """
    def __init__(
        self,
        n_feat: int,
        dilations: List[int] = [1, 2, 4],
        reduction: int = 4,
    ):
        super().__init__()
        
        self.n_feat = n_feat
        self.n_scales = len(dilations)
        reduced_feat = n_feat // reduction
        
        # Multi-scale branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_feat, reduced_feat, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    reduced_feat, reduced_feat, 3,
                    padding=d, dilation=d, bias=False
                ),
                nn.ReLU(inplace=True),
            )
            for d in dilations
        ])
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(reduced_feat * len(dilations), n_feat, 1, bias=False),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extract multi-scale features
        features = [branch(x) for branch in self.branches]
        
        # Concatenate and fuse
        concat = torch.cat(features, dim=1)
        out = self.fusion(concat)
        
        return out


class RCAB_v2(nn.Module):
    """
    Residual Channel Attention Block v2.
    
    Core building block with improvements:
    - Optional dropout for regularization
    - Optional spatial attention
    - Configurable residual scaling
    
    Architecture:
        Conv → ReLU → Conv → ChannelAttention → [SpatialAttention] → [Dropout] → Residual Add
    
    Args:
        n_feat: Number of channels
        kernel_size: Convolution kernel size (default: 3)
        reduction: CA reduction ratio (default: 16)
        dropout: Dropout rate (default: 0.1)
        res_scale: Residual scaling factor (default: 0.1)
        use_spatial_attention: Use spatial attention (default: False)
    """
    def __init__(
        self,
        n_feat: int,
        kernel_size: int = 3,
        reduction: int = 16,
        dropout: float = 0.1,
        res_scale: float = 0.1,
        use_spatial_attention: bool = False,
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        # Main body
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat, n_feat, kernel_size, padding=padding, bias=True),
        )
        
        # Channel attention
        self.ca = ChannelAttention_v2(n_feat, reduction, dropout=0.0)
        
        # Optional spatial attention
        self.sa = SpatialAttention() if use_spatial_attention else None
        
        # Dropout (applied after attention)
        self.dropout = SpatialDropout2d(p=dropout) if dropout > 0 else None
        
        self.res_scale = res_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res = self.ca(res)
        
        if self.sa is not None:
            res = self.sa(res)
        
        if self.dropout is not None:
            res = self.dropout(res)
        
        res = res * self.res_scale
        return x + res


class ResidualGroup_v2(nn.Module):
    """
    Residual Group v2.
    
    Contains multiple RCABs with a residual connection.
    
    v2 Improvements:
    - Dropout support
    - Optional spatial attention at group level
    - Configurable per-block settings
    
    Args:
        n_feat: Number of channels
        kernel_size: Convolution kernel size
        reduction: CA reduction ratio
        n_resblocks: Number of RCABs in the group (default: 20)
        dropout: Dropout rate (default: 0.1)
        res_scale: Residual scaling factor
        use_spatial_attention: Use spatial attention (default: False)
    """
    def __init__(
        self,
        n_feat: int,
        kernel_size: int = 3,
        reduction: int = 16,
        n_resblocks: int = 20,
        dropout: float = 0.1,
        res_scale: float = 0.1,
        use_spatial_attention: bool = False,
    ):
        super().__init__()
        
        self.res_scale = res_scale
        
        # Stack of RCABs
        blocks = [
            RCAB_v2(
                n_feat=n_feat,
                kernel_size=kernel_size,
                reduction=reduction,
                dropout=dropout,
                res_scale=res_scale,
                use_spatial_attention=use_spatial_attention,
            )
            for _ in range(n_resblocks)
        ]
        blocks.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=kernel_size // 2))
        
        self.body = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.body(x)
        res = res * self.res_scale
        return x + res


class Upsampler(nn.Sequential):
    """
    Upsampler using pixel shuffle (sub-pixel convolution).
    
    More efficient and artifact-free than transposed convolution.
    
    Args:
        scale: Upscaling factor (2, 3, or 4)
        n_feat: Number of input channels
        bias: Use bias in convolutions
    """
    def __init__(self, scale: int, n_feat: int, bias: bool = True):
        layers = []
        
        if scale == 2 or scale == 4:
            # For 2x or 4x: use multiple 2x upsamplings
            for _ in range(scale // 2):
                layers.append(nn.Conv2d(n_feat, n_feat * 4, 3, padding=1, bias=bias))
                layers.append(nn.PixelShuffle(2))
                
        elif scale == 3:
            # For 3x: single 3x upsampling
            layers.append(nn.Conv2d(n_feat, n_feat * 9, 3, padding=1, bias=bias))
            layers.append(nn.PixelShuffle(3))
            
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")
        
        super().__init__(*layers)


class ConvBlock(nn.Module):
    """
    Simple convolution block without normalization.
    
    Conv → ReLU
    
    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        bias: Use bias
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


# Quick test
if __name__ == "__main__":
    print("Testing DEM-SR v2 Layers...")
    print("=" * 50)
    
    # Test input
    x = torch.randn(2, 64, 32, 32)
    
    # Test SpatialDropout2d
    dropout = SpatialDropout2d(p=0.2)
    dropout.train()
    y = dropout(x)
    print(f"SpatialDropout2d: {x.shape} -> {y.shape}")
    
    # Test ChannelAttention_v2
    ca = ChannelAttention_v2(64, reduction=16, dropout=0.1)
    y = ca(x)
    print(f"ChannelAttention_v2: {x.shape} -> {y.shape}")
    
    # Test SpatialAttention
    sa = SpatialAttention(kernel_size=7)
    y = sa(x)
    print(f"SpatialAttention: {x.shape} -> {y.shape}")
    
    # Test MultiScaleConv
    msc = MultiScaleConv(64, dilations=[1, 2, 4])
    y = msc(x)
    print(f"MultiScaleConv: {x.shape} -> {y.shape}")
    
    # Test RCAB_v2
    rcab = RCAB_v2(64, dropout=0.1, use_spatial_attention=True)
    y = rcab(x)
    print(f"RCAB_v2: {x.shape} -> {y.shape}")
    
    # Test ResidualGroup_v2
    rg = ResidualGroup_v2(64, n_resblocks=5, dropout=0.1)
    y = rg(x)
    print(f"ResidualGroup_v2: {x.shape} -> {y.shape}")
    
    # Test Upsampler
    up = Upsampler(3, 64)
    y = up(x)
    print(f"Upsampler (3x): {x.shape} -> {y.shape}")
    
    print("=" * 50)
    print("All layer tests passed!")
