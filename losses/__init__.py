"""
DEM-SR Model v2 - Loss Functions

Improved physics-aware losses with:
- Adaptive/learnable loss weights
- Spectral (frequency domain) loss
- Edge-aware loss
- Multi-scale loss
- Terrain complexity weighting
"""

from .adaptive_v2 import (
    AdaptiveLoss_v2,
    ScheduledLoss,
)

from .spectral import SpectralLoss
from .edge_aware import EdgeAwareLoss
from .multiscale import MultiScaleLoss
from .complexity import TerrainComplexityLoss
from .directional_curvature import DirectionalCurvatureLoss, TerrainRoughnessIndexLoss
from .combined_v2 import PhysicsLoss_v2, create_loss_v2

# Re-export base losses from original (elevation, gradient, curvature)
# These will be imported from the parent project
__all__ = [
    # Adaptive
    "AdaptiveLoss_v2",
    "ScheduledLoss",
    # New components
    "SpectralLoss",
    "EdgeAwareLoss",
    "MultiScaleLoss",
    "TerrainComplexityLoss",
    # Combined
    "PhysicsLoss_v2",
    "create_loss_v2",
]
