"""
Multi-Scale Loss for DEM Super-Resolution

Ensures accuracy at multiple spatial resolutions by computing loss
at different scales (original, 2x downsampled, 4x downsampled).
Catches both local details and regional/global errors.

Key Features:
- Multi-resolution loss computation
- Weighted combination across scales
- Ensures consistent accuracy at all scales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class MultiScaleLoss(nn.Module):
    """
    Multi-Scale Loss for consistent accuracy across resolutions.
    
    Computes loss at multiple scales and combines them.
    Ensures the model captures both fine details and overall structure.
    
    Args:
        scales: List of scale factors (1 = original) (default: [1, 2, 4])
        weights: Weight for each scale (default: [1.0, 0.5, 0.25])
        base_loss: Base loss type ("l1" or "l2") (default: "l1")
        reduction: Loss reduction method (default: "mean")
    
    Benefits for DEM SR:
    - Consistent accuracy at local and global scales
    - Better global structure preservation
    - Catches regional errors that might be missed at full resolution
    """
    def __init__(
        self,
        scales: List[int] = [1, 2, 4],
        weights: List[float] = [1.0, 0.5, 0.25],
        base_loss: str = "l1",
        reduction: str = "mean",
    ):
        super().__init__()
        
        assert len(scales) == len(weights), "scales and weights must have same length"
        
        self.scales = scales
        self.weights = weights
        self.reduction = reduction
        
        if base_loss == "l1":
            self.base_loss_fn = F.l1_loss
        else:
            self.base_loss_fn = F.mse_loss
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute multi-scale loss.
        
        Args:
            pred: Predicted DEM [B, 1, H, W]
            target: Ground truth DEM [B, 1, H, W]
            mask: Optional validity mask [B, 1, H, W]
            
        Returns:
            Weighted multi-scale loss
        """
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale > 1:
                # Downsample using average pooling
                pred_scaled = F.avg_pool2d(pred, kernel_size=scale)
                target_scaled = F.avg_pool2d(target, kernel_size=scale)
                
                if mask is not None:
                    # Downsample mask (any invalid pixel makes pooled pixel invalid)
                    mask_scaled = F.avg_pool2d(mask.float(), kernel_size=scale)
                    mask_scaled = (mask_scaled == 1.0).float()
                else:
                    mask_scaled = None
            else:
                pred_scaled = pred
                target_scaled = target
                mask_scaled = mask
            
            # Compute loss at this scale
            if mask_scaled is not None:
                diff = torch.abs(pred_scaled - target_scaled) * mask_scaled
                scale_loss = diff.sum() / (mask_scaled.sum() + 1e-8)
            else:
                scale_loss = self.base_loss_fn(
                    pred_scaled, target_scaled, reduction=self.reduction
                )
            
            total_loss = total_loss + weight * scale_loss
        
        return total_loss


class PyramidLoss(nn.Module):
    """
    Laplacian Pyramid Loss.
    
    Decomposes images into Laplacian pyramid and computes loss at each level.
    More sophisticated multi-scale approach that focuses on details at each scale.
    
    Args:
        levels: Number of pyramid levels (default: 3)
        weights: Weight for each level (default: [1.0, 0.5, 0.25])
    """
    def __init__(
        self,
        levels: int = 3,
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        
        self.levels = levels
        self.weights = weights or [1.0 / (2**i) for i in range(levels)]
        
        # Gaussian kernel for downsampling
        kernel = torch.tensor([
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ], dtype=torch.float32) / 256.0
        self.register_buffer("gaussian_kernel", kernel.view(1, 1, 5, 5))
    
    def _gaussian_downsample(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample with Gaussian blur."""
        # Apply Gaussian blur
        blurred = F.conv2d(x, self.gaussian_kernel, padding=2)
        # Downsample by factor of 2
        return blurred[:, :, ::2, ::2]
    
    def _gaussian_upsample(self, x: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """Upsample with interpolation."""
        return F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
    
    def _laplacian_pyramid(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Build Laplacian pyramid."""
        pyramid = []
        current = x
        
        for i in range(self.levels - 1):
            down = self._gaussian_downsample(current)
            up = self._gaussian_upsample(down, current.shape[-2:])
            laplacian = current - up
            pyramid.append(laplacian)
            current = down
        
        # Last level is the residual (lowest frequency)
        pyramid.append(current)
        
        return pyramid
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Laplacian pyramid loss."""
        pred_pyramid = self._laplacian_pyramid(pred)
        target_pyramid = self._laplacian_pyramid(target)
        
        total_loss = 0.0
        for level, (pred_l, target_l, weight) in enumerate(
            zip(pred_pyramid, target_pyramid, self.weights)
        ):
            level_loss = F.l1_loss(pred_l, target_l, reduction="mean")
            total_loss = total_loss + weight * level_loss
        
        return total_loss


# Quick test
if __name__ == "__main__":
    print("Testing Multi-Scale Loss...")
    print("=" * 50)
    
    # Test data
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    
    # Test MultiScaleLoss
    ms_loss = MultiScaleLoss(scales=[1, 2, 4], weights=[1.0, 0.5, 0.25])
    loss = ms_loss(pred, target)
    print(f"MultiScaleLoss: {loss.item():.4f}")
    
    # Test PyramidLoss
    pyramid_loss = PyramidLoss(levels=3)
    loss = pyramid_loss(pred, target)
    print(f"PyramidLoss: {loss.item():.4f}")
    
    print("=" * 50)
    print("Multi-scale loss tests passed!")
