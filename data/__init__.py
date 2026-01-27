"""
DEM-SR Model v2 - Data Loading and Augmentation

Enhanced data pipeline with:
- Extended augmentation (noise, elevation shift, scale jitter)
- Hard example mining support
- Backward compatibility with v1 data format
"""

from .augmentation import (
    GaussianNoise,
    ElevationShift,
    ScaleJitter,
    DEMAugmentation,
    create_augmentation_pipeline,
)

from .dataset_v2 import (
    DEMDataset_v2,
    create_dataloaders_v2,
)

__all__ = [
    # Augmentation
    "GaussianNoise",
    "ElevationShift",
    "ScaleJitter",
    "DEMAugmentation",
    "create_augmentation_pipeline",
    # Dataset
    "DEMDataset_v2",
    "create_dataloaders_v2",
]
