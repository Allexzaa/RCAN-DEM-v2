"""
Adaptive Loss Functions for DEM Super-Resolution v2

Implements learnable loss weighting based on homoscedastic uncertainty.
The model learns optimal loss weights automatically during training.

Reference: Multi-Task Learning Using Uncertainty to Weigh Losses
https://arxiv.org/abs/1705.07115

Key Features:
- Learnable log-variance parameters for automatic weight balancing
- No manual weight tuning needed
- Scheduled weight adjustment option for curriculum learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List


class ElevationLoss(nn.Module):
    """L1 loss for elevation accuracy."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = torch.abs(pred - target)
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GradientLoss(nn.Module):
    """Gradient loss for slope preservation."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Forward differences
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        
        loss_x = torch.abs(pred_dx - target_dx)
        loss_y = torch.abs(pred_dy - target_dy)
        
        if mask is not None:
            mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
            mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
            loss_x = loss_x * mask_x
            loss_y = loss_y * mask_y
            if self.reduction == "mean":
                return (loss_x.sum() + loss_y.sum()) / (mask_x.sum() + mask_y.sum() + 1e-8)
        
        if self.reduction == "mean":
            return (loss_x.mean() + loss_y.mean()) / 2
        elif self.reduction == "sum":
            return loss_x.sum() + loss_y.sum()
        return loss_x, loss_y


class CurvatureLoss(nn.Module):
    """Laplacian curvature loss for terrain shape preservation."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
        
        # Laplacian kernel
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0],
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("laplacian_kernel", kernel)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred_curv = F.conv2d(pred, self.laplacian_kernel, padding=1)
        target_curv = F.conv2d(target, self.laplacian_kernel, padding=1)
        
        loss = torch.abs(pred_curv - target_curv)
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AdaptiveLoss_v2(nn.Module):
    """
    Adaptive Physics Loss with Learnable Weights.
    
    Uses homoscedastic uncertainty to automatically balance loss components.
    The model learns the optimal weighting during training.
    
    Loss = sum_i (exp(-s_i) * L_i + s_i)
    
    where s_i = log(σ_i^2) is a learnable parameter.
    
    Args:
        use_elevation: Include elevation loss (default: True)
        use_gradient: Include gradient loss (default: True)
        use_curvature: Include curvature loss (default: True)
        init_weights: Initial weight estimates (optional)
    
    Example:
        >>> loss_fn = AdaptiveLoss_v2()
        >>> total_loss, components = loss_fn(pred, target)
        >>> # Weights are learned automatically!
    """
    def __init__(
        self,
        use_elevation: bool = True,
        use_gradient: bool = True,
        use_curvature: bool = True,
        init_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        self.use_elevation = use_elevation
        self.use_gradient = use_gradient
        self.use_curvature = use_curvature
        
        # Initialize learnable log-variance parameters
        # Initial values set so exp(-s) ≈ desired weight
        init_weights = init_weights or {
            "elevation": 1.0,
            "gradient": 0.5,
            "curvature": 0.2,
        }
        
        # s = -log(weight), so exp(-s) = weight
        if use_elevation:
            self.log_var_elev = nn.Parameter(
                torch.tensor(-torch.log(torch.tensor(init_weights.get("elevation", 1.0))))
            )
            self.elevation_loss = ElevationLoss()
        
        if use_gradient:
            self.log_var_grad = nn.Parameter(
                torch.tensor(-torch.log(torch.tensor(init_weights.get("gradient", 0.5))))
            )
            self.gradient_loss = GradientLoss()
        
        if use_curvature:
            self.log_var_curv = nn.Parameter(
                torch.tensor(-torch.log(torch.tensor(init_weights.get("curvature", 0.2))))
            )
            self.curvature_loss = CurvatureLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute adaptive physics loss.
        
        Args:
            pred: Predicted DEM [B, 1, H, W]
            target: Ground truth DEM [B, 1, H, W]
            mask: Optional validity mask [B, 1, H, W]
            return_components: Return individual loss components
            
        Returns:
            Tuple of (total_loss, components_dict)
        """
        total_loss = 0.0
        components = {}
        
        if self.use_elevation:
            l_elev = self.elevation_loss(pred, target, mask)
            w_elev = torch.exp(-self.log_var_elev)
            total_loss = total_loss + w_elev * l_elev + self.log_var_elev
            components["elevation"] = l_elev.detach()
            components["w_elevation"] = w_elev.detach().item()
        
        if self.use_gradient:
            l_grad = self.gradient_loss(pred, target, mask)
            w_grad = torch.exp(-self.log_var_grad)
            total_loss = total_loss + w_grad * l_grad + self.log_var_grad
            components["gradient"] = l_grad.detach()
            components["w_gradient"] = w_grad.detach().item()
        
        if self.use_curvature:
            l_curv = self.curvature_loss(pred, target, mask)
            w_curv = torch.exp(-self.log_var_curv)
            total_loss = total_loss + w_curv * l_curv + self.log_var_curv
            components["curvature"] = l_curv.detach()
            components["w_curvature"] = w_curv.detach().item()
        
        components["total"] = total_loss.detach()
        
        if return_components:
            return total_loss, components
        return total_loss, None
    
    def get_weights(self) -> Dict[str, float]:
        """Get current learned weights."""
        weights = {}
        if self.use_elevation:
            weights["elevation"] = torch.exp(-self.log_var_elev).item()
        if self.use_gradient:
            weights["gradient"] = torch.exp(-self.log_var_grad).item()
        if self.use_curvature:
            weights["curvature"] = torch.exp(-self.log_var_curv).item()
        return weights


class ScheduledLoss(nn.Module):
    """
    Loss with scheduled weight changes.
    
    Implements curriculum learning:
    - Early epochs: Focus on elevation (coarse structure)
    - Mid epochs: Balanced weights
    - Late epochs: Increase detail (gradient/curvature)
    
    Args:
        schedule: Dict mapping epoch -> weight updates
        base_weights: Starting weights
    
    Example:
        >>> schedule = {
        ...     0: {"elevation": 1.0, "gradient": 0.3, "curvature": 0.1},
        ...     50: {"elevation": 1.0, "gradient": 0.5, "curvature": 0.2},
        ...     150: {"elevation": 0.8, "gradient": 0.6, "curvature": 0.3},
        ... }
        >>> loss_fn = ScheduledLoss(schedule=schedule)
    """
    def __init__(
        self,
        schedule: Optional[Dict[int, Dict[str, float]]] = None,
        base_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        
        # Default curriculum schedule
        self.schedule = schedule or {
            0: {"elevation": 1.0, "gradient": 0.3, "curvature": 0.1},
            50: {"elevation": 1.0, "gradient": 0.5, "curvature": 0.2},
            100: {"elevation": 1.0, "gradient": 0.5, "curvature": 0.2},
            150: {"elevation": 0.8, "gradient": 0.6, "curvature": 0.3},
        }
        
        base_weights = base_weights or {"elevation": 1.0, "gradient": 0.5, "curvature": 0.2}
        
        self.w_elevation = base_weights.get("elevation", 1.0)
        self.w_gradient = base_weights.get("gradient", 0.5)
        self.w_curvature = base_weights.get("curvature", 0.2)
        
        self.current_epoch = 0
        
        self.elevation_loss = ElevationLoss()
        self.gradient_loss = GradientLoss()
        self.curvature_loss = CurvatureLoss()
    
    def update_epoch(self, epoch: int):
        """Update weights based on epoch schedule."""
        self.current_epoch = epoch
        
        # Find the most recent schedule entry <= current epoch
        applicable_epochs = [e for e in self.schedule.keys() if e <= epoch]
        if applicable_epochs:
            latest = max(applicable_epochs)
            weights = self.schedule[latest]
            self.w_elevation = weights.get("elevation", self.w_elevation)
            self.w_gradient = weights.get("gradient", self.w_gradient)
            self.w_curvature = weights.get("curvature", self.w_curvature)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Compute scheduled loss."""
        l_elev = self.elevation_loss(pred, target, mask)
        l_grad = self.gradient_loss(pred, target, mask)
        l_curv = self.curvature_loss(pred, target, mask)
        
        total = (
            self.w_elevation * l_elev +
            self.w_gradient * l_grad +
            self.w_curvature * l_curv
        )
        
        if return_components:
            components = {
                "elevation": l_elev.detach(),
                "gradient": l_grad.detach(),
                "curvature": l_curv.detach(),
                "total": total.detach(),
                "w_elevation": self.w_elevation,
                "w_gradient": self.w_gradient,
                "w_curvature": self.w_curvature,
                "epoch": self.current_epoch,
            }
            return total, components
        return total, None
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return {
            "elevation": self.w_elevation,
            "gradient": self.w_gradient,
            "curvature": self.w_curvature,
        }


# Quick test
if __name__ == "__main__":
    print("Testing Adaptive Loss v2...")
    print("=" * 50)
    
    # Test data
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    
    # Test AdaptiveLoss_v2
    adaptive_loss = AdaptiveLoss_v2()
    loss, components = adaptive_loss(pred, target)
    print(f"AdaptiveLoss_v2:")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Learned weights: {adaptive_loss.get_weights()}")
    print(f"  Components: elevation={components['elevation'].item():.4f}, "
          f"gradient={components['gradient'].item():.4f}, "
          f"curvature={components['curvature'].item():.4f}")
    
    # Test ScheduledLoss
    print("\nScheduledLoss:")
    scheduled_loss = ScheduledLoss()
    
    for epoch in [0, 50, 100, 150]:
        scheduled_loss.update_epoch(epoch)
        loss, components = scheduled_loss(pred, target)
        weights = scheduled_loss.get_weights()
        print(f"  Epoch {epoch}: weights={weights}, loss={loss.item():.4f}")
    
    print("\n" + "=" * 50)
    print("Adaptive loss tests passed!")
