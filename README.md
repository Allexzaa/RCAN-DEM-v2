<div align="center">

<img src="images/demo.gif" alt="RCAN-DEM v2 Logo" width="400">

# RCAN-DEM v2: Enhanced DEM Super-Resolution

**30m → 10m DEM Super-Resolution for Vehicle Terrain Analysis**

<p>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Apple%20Silicon-MPS-000000?logo=apple&logoColor=white" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>
</div>

## Overview

**RCAN-DEM v2** is an enhanced deep learning model for Digital Elevation Model (DEM) super-resolution, converting 30m resolution DEMs to 10m resolution with physics-aware loss functions optimized for **vehicle terrain analysis**.

This is the improved v2 architecture with:
- 🎯 **Spatial Dropout** for better generalization
- 🔬 **Adaptive Loss Weights** (learnable)
- 📊 **Vehicle-Specific Losses** (TRI, directional curvature, edge-aware)
- ⚡ **Test-Time Augmentation** for improved accuracy
- 📈 **Early Stopping** and **LR Warmup**

## Key Features

### Architecture Improvements
| Feature | Description |
|---------|-------------|
| Spatial Dropout | Channel-wise dropout for regularization |
| Multi-Scale Conv | Dilated convolutions for multi-scale features |
| Spatial Attention | Learn "where to focus" in terrain |
| ~15.6M Parameters | Same as v1, with better regularization |

### Loss Functions
| Loss | Purpose | Vehicle Application |
|------|---------|---------------------|
| Elevation (L1) | Absolute height accuracy | Elevation range, gain |
| Gradient | Slope preservation | Brake/transmission stress |
| Curvature | Terrain shape | Suspension articulation |
| **TRI** | Roughness index | Suspension wear rate |
| **Edge-Aware** | Obstacle detection | Impact loading |
| **Spectral** | High-frequency detail | Small bumps/vibration |
| **Complexity** | Focus on rough terrain | Challenging routes |

### Training Improvements
- AdamW optimizer with weight decay
- Cosine annealing with linear warmup
- Gradient accumulation (effective batch size 16)
- Early stopping (patience: 30 epochs)

### Training Safety Checks
| Safety Check | Purpose | Action |
|--------------|---------|--------|
| NaN/Inf Detection | Catch numerical instability | Skip batch, log warning |
| Gradient Clipping | Prevent gradient explosion | Clip to max norm (1.0) |
| Checkpoint Validation | Ensure saved models are valid | Retry save, log error |
| Early Stopping | Prevent overfitting | Stop training |

### Inference Improvements
- Test-Time Augmentation (4 rotations)
- Monte Carlo Dropout for uncertainty estimation
- Improved sliding window blending

## Training Data: 13 US Regions

| Region | Terrain Type | Tiles |
|--------|--------------|-------|
| Kansas | Flat Plains | 225 |
| Appalachian | Moderate Hills | 225 |
| Colorado | Steep Mountains | 225 |
| Oregon | Coastal | 225 |
| Arizona | Valleys/Canyons | 225 |
| Death Valley, CA | Desert Basin | 208 |
| Glacier NP, MT | Glacial/Alpine | 273 |
| Utah Canyonlands | Mountains & Canyons | 182 |
| Grand Canyon, AZ | Deep Canyon | 104 |
| Zion NP, UT | Sandstone Cliffs | 36 |
| Bryce Canyon, UT | Hoodoos/Erosional | 24 |
| Monument Valley | Buttes & Mesas | 48 |
| Sedona, AZ | Red Rock Formations | 24 |
| **Total** | **13 Regions** | **2,024** |

## Installation

```bash
# Clone repository
git clone https://github.com/Allexzaa/RCAN-DEM-v2.git
cd RCAN-DEM-v2

# Create conda environment
conda create -n dem-sr-v2 python=3.10
conda activate dem-sr-v2

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy scipy matplotlib rasterio tqdm pyyaml tensorboard
```

## Quick Start

### Training

```bash
# With default config
python scripts/train_v2.py --data_dir processed/ --output_dir outputs_v2/

# With custom config
python scripts/train_v2.py --config configs/fracadapt_optimized.yaml --data_dir processed/

# Resume training
python scripts/train_v2.py --data_dir processed/ --resume outputs_v2/checkpoints/latest.pth
```

### Inference

```bash
# Basic inference
python inference_v2.py \
    --checkpoint outputs_v2/checkpoints/best_model.pth \
    --input dem_30m.tif \
    --output dem_10m_sr.tif \
    --device mps

# With Test-Time Augmentation (recommended)
python inference_v2.py \
    --checkpoint best_model.pth \
    --input dem_30m.tif \
    --output dem_10m.tif \
    --tta \
    --device mps

# With uncertainty estimation
python inference_v2.py \
    --checkpoint best_model.pth \
    --input dem_30m.tif \
    --output dem_10m.tif \
    --uncertainty \
    --uncertainty_output uncertainty_map.tif
```

## Project Structure

```
RCAN-DEM-v2/
├── configs/
│   ├── default_v2.yaml           # Default configuration
│   └── fracadapt_optimized.yaml  # FracAdapt-optimized config
├── models/
│   ├── __init__.py
│   ├── layers_v2.py              # Enhanced layers
│   └── rcan_v2.py                # RCAN v2 architecture
├── losses/
│   ├── __init__.py
│   ├── adaptive_v2.py            # Learnable loss weights
│   ├── combined_v2.py            # PhysicsLoss_v2
│   ├── complexity.py             # Terrain complexity
│   ├── directional_curvature.py  # Vehicle dynamics losses
│   ├── edge_aware.py             # Edge preservation
│   ├── multiscale.py             # Multi-resolution
│   └── spectral.py               # Frequency domain
├── data/
│   ├── __init__.py
│   ├── augmentation.py           # Enhanced augmentation
│   └── dataset_v2.py             # Dataset with augmentation
├── scripts/
│   └── train_v2.py               # Training script
├── docs/
│   └── DEM-SR-FracAdapt-Integration-Report.md
├── inference_v2.py               # Inference with TTA
├── SPECIFICATION.md              # Full specification
├── PROGRESS.md                   # Implementation progress
└── README.md                     # This file
```

## Configuration

### Default Configuration

```yaml
model:
  type: "rcan_v2"
  scale: 3
  dropout: 0.1
  use_spatial_attention: false

loss:
  mode: "fixed"  # or "adaptive"
  w_elevation: 1.0
  w_gradient: 0.5
  w_curvature: 0.2
  w_spectral: 0.1
  w_edge: 0.2

training:
  epochs: 200
  batch_size: 4
  accumulation_steps: 4
  optimizer: "adamw"
  early_stopping: true
  patience: 30

inference:
  use_tta: true
  tta_rotations: [0, 90, 180, 270]
```

## Vehicle Application Integration

This model is optimized for integration with vehicle terrain analysis systems:

| Backend Metric | DEM-SR v2 Loss |
|----------------|----------------|
| Elevation Range | `ElevationLoss` |
| Slope (brake/transmission stress) | `GradientLoss` |
| Roughness Index (suspension stress) | `TRI Loss` |
| Obstacle Density (impact loading) | `EdgeAwareLoss` |
| TTI Score | All losses combined |

See `docs/DEM-SR-FracAdapt-Integration-Report.md` for detailed integration documentation.

## Performance

### Accuracy Improvements vs v1

| Metric | v1 | v2 Expected |
|--------|----|----|
| Flat Terrain Slope MAE | 0.23° | <0.18° |
| Roughness Correlation | 0.81 | >0.88 |
| Generalization | Variable | Consistent |

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | Apple M1 (8GB) | Apple M3 Ultra (64GB+) |
| RAM | 16GB | 64GB+ |
| Storage | 10GB | 50GB+ |

## Related Projects

- [RCAN-DEM v1](https://github.com/Allexzaa/DEM-SR-Model) - Original model
- [FracAdapt](https://github.com/Allexzaa/FracAdapt) - Military fleet management system

## License

MIT License - See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{rcan_dem_v2,
  author = {Zare, Alex},
  title = {RCAN-DEM v2: Enhanced DEM Super-Resolution for Vehicle Terrain Analysis},
  year = {2026},
  url = {https://github.com/Allexzaa/RCAN-DEM-v2}
}
```

## Author

**Alex Zare**

---

*Built with PyTorch on Apple Silicon*
