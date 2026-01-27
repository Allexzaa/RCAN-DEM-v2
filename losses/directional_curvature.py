"""
Directional Curvature Loss for Vehicle Dynamics

For vehicle suspension stress calculations, directional curvatures are critical:
- Profile Curvature: Curvature in the direction of maximum slope (affects pitch dynamics)
- Planform Curvature: Curvature perpendicular to slope (affects roll dynamics)

These are more physically meaningful than the Laplacian for vehicle applications.

Reference: Terrain Analysis for Off-Road Vehicle Navigation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class DirectionalCurvatureLoss(nn.Module):
    """
    Directional Curvature Loss for vehicle dynamics.
    
    Computes and preserves:
    1. Profile Curvature: Rate of slope change in steepest direction
       - Positive = convex (hilltop), vehicle crests
       - Negative = concave (valley), vehicle compression
    
    2. Planform Curvature: Rate of slope change perpendicular to steepest
       - Affects lateral stability and roll dynamics
    
    3. Mean Curvature: Average of principal curvatures
       - Overall terrain shape indicator
    
    Args:
        w_profile: Weight for profile curvature (default: 0.5)
        w_planform: Weight for planform curvature (default: 0.3)
        w_mean: Weight for mean curvature (default: 0.2)
        cell_size: DEM cell size in meters (default: 10.0 for 10m DEM)
        reduction: Loss reduction method (default: "mean")
    
    Benefits for vehicle stress calculation:
    - Profile curvature → Pitch rate → Front/rear suspension load transfer
    - Planform curvature → Roll rate → Left/right suspension load transfer
    - Both critical for predicting suspension articulation and stress
    """
    def __init__(
        self,
        w_profile: float = 0.5,
        w_planform: float = 0.3,
        w_mean: float = 0.2,
        cell_size: float = 10.0,
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.w_profile = w_profile
        self.w_planform = w_planform
        self.w_mean = w_mean
        self.cell_size = cell_size
        self.reduction = reduction
        
        # Sobel kernels for first derivatives
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
        
        # Second derivative kernels
        kernel_xx = torch.tensor([
            [1, -2, 1],
            [1, -2, 1],
            [1, -2, 1],
        ], dtype=torch.float32).view(1, 1, 3, 3) / 3.0
        
        kernel_yy = torch.tensor([
            [1, 1, 1],
            [-2, -2, -2],
            [1, 1, 1],
        ], dtype=torch.float32).view(1, 1, 3, 3) / 3.0
        
        kernel_xy = torch.tensor([
            [-1, 0, 1],
            [0, 0, 0],
            [1, 0, -1],
        ], dtype=torch.float32).view(1, 1, 3, 3) / 4.0
        
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("kernel_xx", kernel_xx)
        self.register_buffer("kernel_yy", kernel_yy)
        self.register_buffer("kernel_xy", kernel_xy)
    
    def compute_curvatures(
        self,
        dem: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute directional curvatures from DEM.
        
        Returns:
            Tuple of (profile_curvature, planform_curvature, mean_curvature)
        """
        # First derivatives (slope)
        p = F.conv2d(dem, self.sobel_x, padding=1) / self.cell_size  # dz/dx
        q = F.conv2d(dem, self.sobel_y, padding=1) / self.cell_size  # dz/dy
        
        # Second derivatives
        r = F.conv2d(dem, self.kernel_xx, padding=1) / (self.cell_size ** 2)  # d2z/dx2
        s = F.conv2d(dem, self.kernel_xy, padding=1) / (self.cell_size ** 2)  # d2z/dxdy
        t = F.conv2d(dem, self.kernel_yy, padding=1) / (self.cell_size ** 2)  # d2z/dy2
        
        # Intermediate calculations
        p2 = p ** 2
        q2 = q ** 2
        pq = p * q
        
        # Denominator terms
        denom1 = p2 + q2
        denom2 = (1 + p2 + q2) ** 1.5
        
        # Avoid division by zero
        eps = 1e-8
        
        # Profile curvature (in direction of steepest slope)
        # Affects pitch dynamics - how the vehicle "crests" hills
        profile_curv = (
            (r * p2 + 2 * s * pq + t * q2) /
            (denom1 * torch.sqrt(1 + denom1) + eps)
        )
        
        # Planform curvature (perpendicular to steepest slope)
        # Affects roll dynamics - lateral stability
        planform_curv = (
            (r * q2 - 2 * s * pq + t * p2) /
            (denom1 ** 1.5 + eps)
        )
        
        # Mean curvature (average of principal curvatures)
        mean_curv = (
            ((1 + q2) * r - 2 * pq * s + (1 + p2) * t) /
            (2 * denom2 + eps)
        )
        
        # Handle flat areas (where denom1 ≈ 0)
        flat_mask = denom1 < eps
        profile_curv = torch.where(flat_mask, torch.zeros_like(profile_curv), profile_curv)
        planform_curv = torch.where(flat_mask, torch.zeros_like(planform_curv), planform_curv)
        
        return profile_curv, planform_curv, mean_curv
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute directional curvature loss.
        
        Args:
            pred: Predicted DEM [B, 1, H, W]
            target: Ground truth DEM [B, 1, H, W]
            mask: Optional validity mask [B, 1, H, W]
            
        Returns:
            Weighted directional curvature loss
        """
        # Compute curvatures for both
        pred_profile, pred_planform, pred_mean = self.compute_curvatures(pred)
        target_profile, target_planform, target_mean = self.compute_curvatures(target)
        
        # L1 losses for each curvature type
        loss_profile = torch.abs(pred_profile - target_profile)
        loss_planform = torch.abs(pred_planform - target_planform)
        loss_mean = torch.abs(pred_mean - target_mean)
        
        # Apply mask if provided
        if mask is not None:
            loss_profile = loss_profile * mask
            loss_planform = loss_planform * mask
            loss_mean = loss_mean * mask
            
            if self.reduction == "mean":
                count = mask.sum() + 1e-8
                loss_profile = loss_profile.sum() / count
                loss_planform = loss_planform.sum() / count
                loss_mean = loss_mean.sum() / count
        elif self.reduction == "mean":
            loss_profile = loss_profile.mean()
            loss_planform = loss_planform.mean()
            loss_mean = loss_mean.mean()
        
        # Weighted combination
        total_loss = (
            self.w_profile * loss_profile +
            self.w_planform * loss_planform +
            self.w_mean * loss_mean
        )
        
        return total_loss
    
    def get_weights(self) -> dict:
        """Get current loss weights."""
        return {
            "profile": self.w_profile,
            "planform": self.w_planform,
            "mean": self.w_mean,
        }


class TerrainRoughnessIndexLoss(nn.Module):
    """
    Terrain Ruggedness Index (TRI) Loss.
    
    TRI measures local elevation variability, which directly correlates
    with vehicle vibration and component fatigue.
    
    Formula: TRI = sqrt(sum((z_i - z_center)^2) / 8)
    
    Benefits for vehicle application:
    - High TRI = rough terrain = high-frequency vibrations
    - Preserving TRI accuracy ensures correct fatigue load prediction
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def compute_tri(self, dem: torch.Tensor) -> torch.Tensor:
        """Compute Terrain Ruggedness Index."""
        # Use a 3x3 neighborhood
        # TRI = sqrt(mean((z_neighbor - z_center)^2))
        
        # Get center values (original DEM)
        center = dem
        
        # Compute sum of squared differences using conv
        # Kernel that computes (z_i - z_center)^2 sum
        kernel = torch.ones(1, 1, 3, 3, device=dem.device, dtype=dem.dtype)
        kernel[0, 0, 1, 1] = 0  # Exclude center
        
        # Sum of neighbors
        neighbor_sum = F.conv2d(dem, kernel, padding=1)
        neighbor_count = 8.0
        neighbor_mean = neighbor_sum / neighbor_count
        
        # Sum of squared neighbors
        neighbor_sq_sum = F.conv2d(dem ** 2, kernel, padding=1)
        neighbor_sq_mean = neighbor_sq_sum / neighbor_count
        
        # Variance around center
        # E[(z_i - z_center)^2] = E[z_i^2] - 2*z_center*E[z_i] + z_center^2
        variance = neighbor_sq_mean - 2 * center * neighbor_mean + center ** 2
        
        # TRI = sqrt(variance)
        tri = torch.sqrt(variance + 1e-8)
        
        return tri
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute TRI loss."""
        pred_tri = self.compute_tri(pred)
        target_tri = self.compute_tri(target)
        
        loss = torch.abs(pred_tri - target_tri)
        
        if mask is not None:
            loss = loss * mask
            if self.reduction == "mean":
                return loss.sum() / (mask.sum() + 1e-8)
        
        if self.reduction == "mean":
            return loss.mean()
        return loss


# Quick test
if __name__ == "__main__":
    print("Testing Directional Curvature Loss...")
    print("=" * 60)
    
    # Test data
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    
    # Test DirectionalCurvatureLoss
    dir_curv_loss = DirectionalCurvatureLoss(
        w_profile=0.5,
        w_planform=0.3,
        w_mean=0.2,
        cell_size=10.0,
    )
    loss = dir_curv_loss(pred, target)
    print(f"DirectionalCurvatureLoss: {loss.item():.6f}")
    print(f"  Weights: {dir_curv_loss.get_weights()}")
    
    # Test curvature computation
    profile, planform, mean = dir_curv_loss.compute_curvatures(target)
    print(f"  Profile curvature range: [{profile.min():.4f}, {profile.max():.4f}]")
    print(f"  Planform curvature range: [{planform.min():.4f}, {planform.max():.4f}]")
    print(f"  Mean curvature range: [{mean.min():.4f}, {mean.max():.4f}]")
    
    # Test TRI Loss
    tri_loss = TerrainRoughnessIndexLoss()
    loss = tri_loss(pred, target)
    print(f"\nTerrainRoughnessIndexLoss: {loss.item():.6f}")
    
    print("\n" + "=" * 60)
    print("Directional curvature tests passed!")
