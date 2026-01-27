"""
Combined Physics Loss v2 for DEM Super-Resolution

Integrates all loss components:
- Elevation (L1)
- Gradient (slope preservation)
- Curvature (Laplacian)
- Spectral (frequency domain)
- Edge-aware
- Multi-scale
- Terrain complexity weighting

Supports both fixed and adaptive weighting.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List

from .adaptive_v2 import ElevationLoss, GradientLoss, CurvatureLoss, AdaptiveLoss_v2
from .spectral import SpectralLoss
from .edge_aware import EdgeAwareLoss
from .multiscale import MultiScaleLoss
from .complexity import TerrainComplexityLoss


class PhysicsLoss_v2(nn.Module):
    """
    Combined Physics-Aware Loss v2.
    
    Integrates all loss components with configurable weights.
    Supports fixed weights, adaptive (learnable) weights, or scheduled weights.
    
    Args:
        mode: "fixed", "adaptive", or "scheduled"
        
        # Fixed mode weights:
        w_elevation: Weight for elevation loss (default: 1.0)
        w_gradient: Weight for gradient loss (default: 0.5)
        w_curvature: Weight for curvature loss (default: 0.2)
        w_spectral: Weight for spectral loss (default: 0.1)
        w_edge: Weight for edge-aware loss (default: 0.2)
        w_multiscale: Weight for multi-scale loss (default: 0.0, disabled)
        w_complexity: Weight for complexity weighting (default: 0.0, disabled)
        
        # Component enable/disable:
        use_spectral: Enable spectral loss (default: True)
        use_edge: Enable edge-aware loss (default: True)
        use_multiscale: Enable multi-scale loss (default: False)
        use_complexity: Enable complexity weighting (default: True)
    
    Example:
        >>> # Fixed weights
        >>> loss_fn = PhysicsLoss_v2(mode="fixed", w_spectral=0.1, w_edge=0.2)
        >>> total_loss, components = loss_fn(pred, target)
        
        >>> # Adaptive weights
        >>> loss_fn = PhysicsLoss_v2(mode="adaptive")
        >>> total_loss, components = loss_fn(pred, target)
    """
    def __init__(
        self,
        mode: str = "fixed",
        # Fixed weights
        w_elevation: float = 1.0,
        w_gradient: float = 0.5,
        w_curvature: float = 0.2,
        w_spectral: float = 0.1,
        w_edge: float = 0.2,
        w_multiscale: float = 0.0,
        w_complexity: float = 0.0,
        # Component toggles
        use_spectral: bool = True,
        use_edge: bool = True,
        use_multiscale: bool = False,
        use_complexity: bool = True,
        # Component parameters
        spectral_weight_high: float = 0.5,
        edge_weight: float = 2.0,
        complexity_weight: float = 2.0,
    ):
        super().__init__()
        
        self.mode = mode
        
        # Store fixed weights
        self.w_elevation = w_elevation
        self.w_gradient = w_gradient
        self.w_curvature = w_curvature
        self.w_spectral = w_spectral
        self.w_edge = w_edge
        self.w_multiscale = w_multiscale
        self.w_complexity = w_complexity
        
        # Component toggles
        self.use_spectral = use_spectral and w_spectral > 0
        self.use_edge = use_edge and w_edge > 0
        self.use_multiscale = use_multiscale and w_multiscale > 0
        self.use_complexity = use_complexity and w_complexity > 0
        
        # Initialize loss functions
        if mode == "adaptive":
            # Adaptive mode uses learnable weights for core losses
            self.adaptive_loss = AdaptiveLoss_v2()
        else:
            # Fixed/scheduled mode uses separate losses
            self.elevation_loss = ElevationLoss()
            self.gradient_loss = GradientLoss()
            self.curvature_loss = CurvatureLoss()
            self.adaptive_loss = None
        
        # New loss components (always fixed weight)
        if self.use_spectral:
            self.spectral_loss = SpectralLoss(
                weight_low=1.0,
                weight_high=spectral_weight_high,
            )
        
        if self.use_edge:
            self.edge_loss = EdgeAwareLoss(edge_weight=edge_weight)
        
        if self.use_multiscale:
            self.multiscale_loss = MultiScaleLoss(
                scales=[1, 2, 4],
                weights=[1.0, 0.5, 0.25],
            )
        
        if self.use_complexity:
            self.complexity_loss = TerrainComplexityLoss(
                complexity_weight=complexity_weight,
            )
        
        # Store config
        self.config = {
            "mode": mode,
            "w_elevation": w_elevation,
            "w_gradient": w_gradient,
            "w_curvature": w_curvature,
            "w_spectral": w_spectral,
            "w_edge": w_edge,
            "w_multiscale": w_multiscale,
            "w_complexity": w_complexity,
            "use_spectral": use_spectral,
            "use_edge": use_edge,
            "use_multiscale": use_multiscale,
            "use_complexity": use_complexity,
        }
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_components: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Compute combined physics loss.
        
        Args:
            pred: Predicted DEM [B, 1, H, W]
            target: Ground truth DEM [B, 1, H, W]
            mask: Optional validity mask [B, 1, H, W]
            return_components: Return individual loss components
            
        Returns:
            Tuple of (total_loss, components_dict)
        """
        components = {}
        total_loss = torch.tensor(0.0, device=pred.device)
        
        # Core losses (elevation, gradient, curvature)
        if self.mode == "adaptive":
            # Adaptive mode: learnable weights
            core_loss, core_components = self.adaptive_loss(pred, target, mask)
            total_loss = total_loss + core_loss
            components.update(core_components)
        else:
            # Fixed mode: manual weights
            l_elev = self.elevation_loss(pred, target, mask)
            l_grad = self.gradient_loss(pred, target, mask)
            l_curv = self.curvature_loss(pred, target, mask)
            
            total_loss = total_loss + self.w_elevation * l_elev
            total_loss = total_loss + self.w_gradient * l_grad
            total_loss = total_loss + self.w_curvature * l_curv
            
            components["elevation"] = l_elev.detach()
            components["gradient"] = l_grad.detach()
            components["curvature"] = l_curv.detach()
            components["w_elevation"] = self.w_elevation
            components["w_gradient"] = self.w_gradient
            components["w_curvature"] = self.w_curvature
        
        # Additional loss components
        if self.use_spectral:
            l_spectral = self.spectral_loss(pred, target, mask)
            total_loss = total_loss + self.w_spectral * l_spectral
            components["spectral"] = l_spectral.detach()
            components["w_spectral"] = self.w_spectral
        
        if self.use_edge:
            l_edge = self.edge_loss(pred, target, mask)
            total_loss = total_loss + self.w_edge * l_edge
            components["edge"] = l_edge.detach()
            components["w_edge"] = self.w_edge
        
        if self.use_multiscale:
            l_multiscale = self.multiscale_loss(pred, target, mask)
            total_loss = total_loss + self.w_multiscale * l_multiscale
            components["multiscale"] = l_multiscale.detach()
            components["w_multiscale"] = self.w_multiscale
        
        if self.use_complexity:
            l_complexity = self.complexity_loss(pred, target, mask)
            total_loss = total_loss + self.w_complexity * l_complexity
            components["complexity"] = l_complexity.detach()
            components["w_complexity"] = self.w_complexity
        
        components["total"] = total_loss.detach()
        
        if return_components:
            return total_loss, components
        return total_loss, None
    
    def get_weights(self) -> Dict[str, float]:
        """Get current loss weights."""
        weights = {}
        
        if self.mode == "adaptive":
            weights.update(self.adaptive_loss.get_weights())
        else:
            weights["elevation"] = self.w_elevation
            weights["gradient"] = self.w_gradient
            weights["curvature"] = self.w_curvature
        
        if self.use_spectral:
            weights["spectral"] = self.w_spectral
        if self.use_edge:
            weights["edge"] = self.w_edge
        if self.use_multiscale:
            weights["multiscale"] = self.w_multiscale
        if self.use_complexity:
            weights["complexity"] = self.w_complexity
        
        return weights
    
    def get_config(self) -> Dict:
        """Get loss configuration."""
        return self.config.copy()


def create_loss_v2(
    loss_type: str = "physics_v2",
    **kwargs,
) -> nn.Module:
    """
    Factory function to create v2 loss functions.
    
    Args:
        loss_type: "physics_v2", "adaptive", or "fixed"
        **kwargs: Additional arguments for the loss
        
    Returns:
        Loss module
    """
    if loss_type == "physics_v2":
        return PhysicsLoss_v2(**kwargs)
    elif loss_type == "adaptive":
        return PhysicsLoss_v2(mode="adaptive", **kwargs)
    elif loss_type == "fixed":
        return PhysicsLoss_v2(mode="fixed", **kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Quick test
if __name__ == "__main__":
    print("Testing Combined Physics Loss v2...")
    print("=" * 60)
    
    # Test data
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    
    # Test fixed mode
    print("\n1. Fixed mode:")
    loss_fixed = PhysicsLoss_v2(
        mode="fixed",
        w_elevation=1.0,
        w_gradient=0.5,
        w_curvature=0.2,
        w_spectral=0.1,
        w_edge=0.2,
        use_spectral=True,
        use_edge=True,
        use_complexity=True,
    )
    loss, components = loss_fixed(pred, target)
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Weights: {loss_fixed.get_weights()}")
    print(f"   Components: {list(components.keys())}")
    
    # Test adaptive mode
    print("\n2. Adaptive mode:")
    loss_adaptive = PhysicsLoss_v2(
        mode="adaptive",
        use_spectral=True,
        use_edge=True,
        w_spectral=0.1,
        w_edge=0.2,
    )
    loss, components = loss_adaptive(pred, target)
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Learned weights: {loss_adaptive.get_weights()}")
    
    # Test factory function
    print("\n3. Factory function:")
    loss_factory = create_loss_v2("physics_v2", mode="fixed", w_spectral=0.15)
    loss, _ = loss_factory(pred, target)
    print(f"   Factory loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("All PhysicsLoss_v2 tests passed!")
