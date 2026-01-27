"""
Edge-Aware Loss for DEM Super-Resolution

Provides extra emphasis on terrain boundaries and discontinuities.
Critical for preserving cliffs, ridges, and sharp slope transitions
that represent obstacles for vehicle suspension systems.

Key Features:
- Sobel-based edge detection
- Adaptive weighting based on edge magnitude
- Preserves terrain boundaries
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EdgeAwareLoss(nn.Module):
    """
    Edge-Aware Loss for terrain boundary preservation.
    
    Weights the loss higher at terrain boundaries (detected via Sobel gradients).
    Ensures sharp features like cliffs and ridges are preserved.
    
    Args:
        edge_weight: Maximum weight multiplier at edges (default: 2.0)
        base_loss: Base loss type ("l1" or "l2") (default: "l1")
        reduction: Loss reduction method (default: "mean")
    
    Benefits for DEM SR:
    - Sharper terrain boundaries
    - Better cliff and ridge preservation
    - Prevents edge blurring
    - Preserves sudden elevation changes (obstacles)
    """
    def __init__(
        self,
        edge_weight: float = 2.0,
        base_loss: str = "l1",
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.edge_weight = edge_weight
        self.reduction = reduction
        
        if base_loss == "l1":
            self.base_loss_fn = nn.L1Loss(reduction='none')
        else:
            self.base_loss_fn = nn.MSELoss(reduction='none')
        
        # Sobel kernels for edge detection
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
    
    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel operators."""
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2 + 1e-8)
        return edge_magnitude
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute edge-aware loss.
        
        Args:
            pred: Predicted DEM [B, 1, H, W]
            target: Ground truth DEM [B, 1, H, W]
            mask: Optional validity mask [B, 1, H, W]
            
        Returns:
            Edge-weighted loss value
        """
        # Compute edge magnitude from target
        edge_magnitude = self.compute_edges(target)
        
        # Normalize edge magnitude to [0, 1]
        edge_norm = edge_magnitude / (edge_magnitude.max() + 1e-8)
        
        # Create weight map: 1 at flat areas, edge_weight at edges
        weight_map = 1 + (self.edge_weight - 1) * edge_norm
        
        # Compute base pixel-wise loss
        pixel_loss = self.base_loss_fn(pred, target)
        
        # Apply edge weights
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


class GradientEdgeLoss(nn.Module):
    """
    Alternative edge loss using gradient magnitude directly.
    
    Instead of weighting by edge magnitude, compares gradient magnitudes.
    Simpler but effective for edge preservation.
    
    Args:
        reduction: Loss reduction method (default: "mean")
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        
        # Sobel kernels
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute gradient magnitude loss."""
        # Compute gradients
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        
        # Gradient magnitudes
        pred_grad = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8)
        target_grad = torch.sqrt(target_gx**2 + target_gy**2 + 1e-8)
        
        # L1 loss on gradient magnitudes
        loss = torch.abs(pred_grad - target_grad)
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


# Quick test
if __name__ == "__main__":
    print("Testing Edge-Aware Loss...")
    print("=" * 50)
    
    # Test data
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    
    # Test EdgeAwareLoss
    edge_loss = EdgeAwareLoss(edge_weight=2.0)
    loss = edge_loss(pred, target)
    print(f"EdgeAwareLoss: {loss.item():.4f}")
    
    # Test GradientEdgeLoss
    grad_edge_loss = GradientEdgeLoss()
    loss = grad_edge_loss(pred, target)
    print(f"GradientEdgeLoss: {loss.item():.4f}")
    
    print("=" * 50)
    print("Edge-aware loss tests passed!")
