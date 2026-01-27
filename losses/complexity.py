"""
Terrain Complexity Weighting Loss for DEM Super-Resolution

Weights the loss by local terrain complexity, forcing the model
to work harder on challenging areas without sacrificing performance
on easier regions.

Key Features:
- Local variance-based complexity measure
- Higher weight in complex (rough) terrain
- Prevents "lazy learning" on easy (flat) terrain
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TerrainComplexityLoss(nn.Module):
    """
    Terrain Complexity Weighted Loss.
    
    Weights the loss higher in regions with high terrain complexity
    (measured by local variance/roughness).
    
    Args:
        complexity_weight: Maximum weight in complex areas (default: 2.0)
        window_size: Window for complexity computation (default: 5)
        base_loss: Base loss type ("l1" or "l2") (default: "l1")
        reduction: Loss reduction method (default: "mean")
    
    Benefits for DEM SR:
    - Better results on complex terrain
    - Maintains performance on simple terrain
    - Focuses learning on challenging areas
    - Prevents model from "ignoring" difficult regions
    """
    def __init__(
        self,
        complexity_weight: float = 2.0,
        window_size: int = 5,
        base_loss: str = "l1",
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.complexity_weight = complexity_weight
        self.window_size = window_size
        self.reduction = reduction
        
        if base_loss == "l1":
            self.base_loss_fn = nn.L1Loss(reduction='none')
        else:
            self.base_loss_fn = nn.MSELoss(reduction='none')
    
    def compute_complexity(self, dem: torch.Tensor) -> torch.Tensor:
        """
        Compute local terrain complexity using variance.
        
        High variance = complex terrain (rough, varied)
        Low variance = simple terrain (flat, smooth)
        """
        padding = self.window_size // 2
        
        # Compute local mean
        kernel_size = self.window_size
        mean_filter = torch.ones(
            1, 1, kernel_size, kernel_size, device=dem.device
        ) / (kernel_size * kernel_size)
        
        local_mean = F.conv2d(dem, mean_filter, padding=padding)
        local_sq_mean = F.conv2d(dem**2, mean_filter, padding=padding)
        
        # Variance = E[X^2] - E[X]^2
        local_variance = local_sq_mean - local_mean**2
        local_variance = torch.clamp(local_variance, min=0)  # Numerical stability
        
        # Normalize to [0, 1]
        complexity = local_variance / (local_variance.max() + 1e-8)
        
        return complexity
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute complexity-weighted loss.
        
        Args:
            pred: Predicted DEM [B, 1, H, W]
            target: Ground truth DEM [B, 1, H, W]
            mask: Optional validity mask [B, 1, H, W]
            
        Returns:
            Complexity-weighted loss value
        """
        # Compute terrain complexity from target
        complexity = self.compute_complexity(target)
        
        # Create weight map: 1 at simple terrain, complexity_weight at complex
        weight_map = 1 + (self.complexity_weight - 1) * complexity
        
        # Compute base pixel-wise loss
        pixel_loss = self.base_loss_fn(pred, target)
        
        # Apply complexity weights
        weighted_loss = pixel_loss * weight_map
        
        # Apply validity mask if provided
        if mask is not None:
            weighted_loss = weighted_loss * mask
            if self.reduction == "mean":
                return weighted_loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss


class RoughnessLoss(nn.Module):
    """
    Roughness-Aware Loss.
    
    Alternative complexity measure using standard deviation
    in a sliding window (surface roughness).
    
    Args:
        roughness_weight: Maximum weight in rough areas (default: 2.0)
        window_size: Window for roughness computation (default: 5)
    """
    def __init__(
        self,
        roughness_weight: float = 2.0,
        window_size: int = 5,
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.roughness_weight = roughness_weight
        self.window_size = window_size
        self.reduction = reduction
    
    def compute_roughness(self, dem: torch.Tensor) -> torch.Tensor:
        """Compute local roughness (standard deviation)."""
        padding = self.window_size // 2
        kernel_size = self.window_size
        
        mean_filter = torch.ones(
            1, 1, kernel_size, kernel_size, device=dem.device
        ) / (kernel_size * kernel_size)
        
        local_mean = F.conv2d(dem, mean_filter, padding=padding)
        local_sq_mean = F.conv2d(dem**2, mean_filter, padding=padding)
        
        local_variance = local_sq_mean - local_mean**2
        local_variance = torch.clamp(local_variance, min=0)
        local_std = torch.sqrt(local_variance + 1e-8)
        
        # Normalize to [0, 1]
        roughness = local_std / (local_std.max() + 1e-8)
        
        return roughness
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute roughness-weighted loss."""
        roughness = self.compute_roughness(target)
        weight_map = 1 + (self.roughness_weight - 1) * roughness
        
        pixel_loss = torch.abs(pred - target)
        weighted_loss = pixel_loss * weight_map
        
        if mask is not None:
            weighted_loss = weighted_loss * mask
            if self.reduction == "mean":
                return weighted_loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        return weighted_loss


# Quick test
if __name__ == "__main__":
    print("Testing Terrain Complexity Loss...")
    print("=" * 50)
    
    # Test data
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    
    # Test TerrainComplexityLoss
    complexity_loss = TerrainComplexityLoss(complexity_weight=2.0, window_size=5)
    loss = complexity_loss(pred, target)
    print(f"TerrainComplexityLoss: {loss.item():.4f}")
    
    # Test RoughnessLoss
    roughness_loss = RoughnessLoss(roughness_weight=2.0, window_size=5)
    loss = roughness_loss(pred, target)
    print(f"RoughnessLoss: {loss.item():.4f}")
    
    print("=" * 50)
    print("Complexity loss tests passed!")
