"""
DEM-SR Model v2 - Neural Network Models

Improved architecture with:
- Spatial dropout for regularization
- Optional multi-scale convolutions
- Optional spatial attention
- Better generalization to unseen terrain
"""

from .layers_v2 import (
    SpatialDropout2d,
    ChannelAttention_v2,
    SpatialAttention,
    RCAB_v2,
    ResidualGroup_v2,
    MultiScaleConv,
    Upsampler,
)

from .rcan_v2 import (
    RCAN_DEM_v2,
    RCAN_DEM_v2_Light,
    create_model_v2,
)

__all__ = [
    # Layers
    "SpatialDropout2d",
    "ChannelAttention_v2",
    "SpatialAttention",
    "RCAB_v2",
    "ResidualGroup_v2",
    "MultiScaleConv",
    "Upsampler",
    # Models
    "RCAN_DEM_v2",
    "RCAN_DEM_v2_Light",
    "create_model_v2",
]
