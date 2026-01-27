# DEM-SR Model v2 - Improvement Specification

## Overview

This document specifies all improvements to be implemented in DEM-SR Model v2, based on thorough analysis of the v1 model's performance and identified limitations.

**Author:** Alex Zare  
**Date:** January 26, 2026  
**Base Model:** RCAN-DEM v1 (15.6M parameters)  
**Target:** Improved terrain derivative accuracy for vehicle suspension stress calculation

---

## Improvement Categories

| Category | Priority | Phases |
|----------|----------|--------|
| Architecture | High | Phase 1-2 |
| Loss Functions | High | Phase 3-4 |
| Data & Augmentation | Medium | Phase 5 |
| Training Process | Medium | Phase 6 |
| Inference | Low | Phase 7 |

---

## Phase 1: Architecture - Regularization & Dropout

### 1.1 Spatial Dropout

**File:** `models/layers_v2.py`

**Changes:**
- Add `SpatialDropout2d` class
- Add dropout after channel attention in RCAB blocks
- Configurable dropout rate (default: 0.1)

**Rationale:** 
- v1 has 15.6M params but only ~1,125 training tiles
- Risk of overfitting, especially on extreme terrain
- Dropout improves generalization

**Parameters:**
```python
dropout_rate: float = 0.1  # Applied after channel attention
```

### 1.2 Residual Scaling Improvement

**Changes:**
- Make `res_scale` configurable per-group
- Allow different scaling in early vs. late groups

---

## Phase 2: Architecture - Multi-Scale & Attention

### 2.1 Multi-Scale Feature Extraction

**File:** `models/layers_v2.py`

**Changes:**
- Add `MultiScaleConv` module with dilated convolutions
- Receptive field sizes: 3×3, 5×5, 7×7 (via dilation)
- Concatenate and reduce features

**Rationale:**
- Terrain features exist at multiple scales
- Better capture of small bumps AND large valleys

### 2.2 Spatial Attention (Optional)

**File:** `models/layers_v2.py`

**Changes:**
- Add lightweight `SpatialAttention` module
- Learn "where to focus" in addition to "what features"
- Applied after select residual groups

**Parameters:**
```python
use_spatial_attention: bool = True
attention_reduction: int = 8
```

### 2.3 Updated RCAN Architecture

**File:** `models/rcan_v2.py`

**Changes:**
- Integrate dropout into RCAB_v2
- Add optional multi-scale conv
- Add optional spatial attention
- Maintain backward compatibility with v1 checkpoints

---

## Phase 3: Loss Functions - Adaptive Weights

### 3.1 Uncertainty-Based Adaptive Loss

**File:** `losses/adaptive_v2.py`

**Changes:**
- Learnable log-variance parameters for each loss
- Automatic balancing based on homoscedastic uncertainty
- No manual weight tuning needed

**Formula:**
```
L_total = exp(-s_elev) * L_elev + s_elev + 
          exp(-s_grad) * L_grad + s_grad + 
          exp(-s_curv) * L_curv + s_curv
```

### 3.2 Scheduled Weight Adjustment

**Changes:**
- Optional curriculum: elevation-focused early, detail-focused late
- Schedule: epochs 1-50 (elev: 1.0), 50-150 (balanced), 150+ (detail: higher)

---

## Phase 4: Loss Functions - New Components

### 4.1 Spectral/Frequency Loss

**File:** `losses/spectral.py`

**Changes:**
- FFT-based loss comparing frequency magnitudes
- Separate weights for low and high frequencies
- High-frequency preservation for small obstacle detection

**Parameters:**
```python
weight_low: float = 1.0   # Overall shape
weight_high: float = 0.5  # Fine detail
cutoff: float = 0.25      # Frequency cutoff ratio
```

### 4.2 Edge-Aware Loss

**File:** `losses/edge_aware.py`

**Changes:**
- Sobel-based edge detection on ground truth
- Higher weights at terrain boundaries
- Preserves cliffs, ridges, sharp transitions

**Parameters:**
```python
edge_weight: float = 2.0  # Weight multiplier at edges
```

### 4.3 Multi-Scale Loss

**File:** `losses/multiscale.py`

**Changes:**
- Compute loss at multiple resolutions (1×, 2×, 4× downsampled)
- Ensures accuracy at local AND global scales
- Weighted combination

**Parameters:**
```python
scales: List[int] = [1, 2, 4]
weights: List[float] = [1.0, 0.5, 0.25]
```

### 4.4 Terrain Complexity Weighting

**File:** `losses/complexity.py`

**Changes:**
- Compute local terrain complexity (variance-based)
- Weight loss higher in complex regions
- Prevents "lazy learning" on easy terrain

**Parameters:**
```python
complexity_weight: float = 2.0  # Max weight in complex areas
window_size: int = 5            # Complexity computation window
```

### 4.5 Combined Physics Loss v2

**File:** `losses/combined_v2.py`

**Changes:**
- Integrate all new loss components
- Support for adaptive OR fixed weights
- Configurable component selection

---

## Phase 5: Data & Augmentation

### 5.1 Enhanced Augmentation

**File:** `data/augmentation.py`

**New Augmentations:**
| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Gaussian Noise | σ = 0.01-0.05 | Sensor noise robustness |
| Elevation Shift | ±5m | Elevation invariance |
| Scale Jitter | 0.9-1.1× | Scale invariance |
| Elastic Deformation | α=50, σ=5 | Terrain warping |

### 5.2 Hard Example Mining (Optional)

**File:** `data/dataset_v2.py`

**Changes:**
- Track per-tile loss history
- Oversample high-loss tiles
- Configurable mining ratio

---

## Phase 6: Training Process

### 6.1 Early Stopping

**File:** `train_v2.py`

**Changes:**
- Monitor validation loss
- Stop if no improvement for N epochs
- Save best model automatically

**Parameters:**
```python
early_stopping_patience: int = 30
min_delta: float = 1e-4
```

### 6.2 Learning Rate Warmup

**Changes:**
- Linear warmup from 0 to initial LR
- Warmup period: 5-10 epochs
- Prevents early instability

**Parameters:**
```python
warmup_epochs: int = 5
```

### 6.3 Gradient Accumulation

**Changes:**
- Accumulate gradients over multiple steps
- Effective batch size = batch_size × accumulation_steps

**Parameters:**
```python
accumulation_steps: int = 4  # Effective batch = 16
```

### 6.4 AdamW Optimizer

**Changes:**
- Switch from Adam to AdamW
- Decoupled weight decay for better regularization

**Parameters:**
```python
optimizer: str = "adamw"
weight_decay: float = 0.01
```

---

## Phase 7: Inference Improvements

### 7.1 Test-Time Augmentation (TTA)

**File:** `inference_v2.py`

**Changes:**
- Process each tile at multiple rotations
- Optionally add flips
- Average predictions

**Parameters:**
```python
use_tta: bool = True
tta_rotations: List[int] = [0, 90, 180, 270]
tta_flips: bool = False
```

### 7.2 Uncertainty Estimation (Optional)

**Changes:**
- Monte Carlo Dropout at inference
- Multiple forward passes
- Output prediction + uncertainty map

---

## Configuration File

**File:** `configs/default_v2.yaml`

```yaml
# Model
model:
  type: "rcan_v2"
  n_resgroups: 10
  n_resblocks: 20
  n_feats: 64
  dropout_rate: 0.1
  use_spatial_attention: true
  use_multiscale_conv: false

# Loss
loss:
  type: "adaptive"  # or "fixed"
  components:
    elevation: true
    gradient: true
    curvature: true
    spectral: true
    edge_aware: true
    multiscale: false
    complexity: true
  fixed_weights:  # Used if type="fixed"
    elevation: 1.0
    gradient: 0.5
    curvature: 0.2
    spectral: 0.1
    edge_aware: 0.2

# Training
training:
  epochs: 200
  batch_size: 4
  accumulation_steps: 4
  optimizer: "adamw"
  lr: 1e-4
  lr_min: 1e-6
  weight_decay: 0.01
  warmup_epochs: 5
  early_stopping_patience: 30

# Augmentation
augmentation:
  flip_horizontal: true
  flip_vertical: true
  rotate_90: true
  gaussian_noise: true
  noise_std: 0.02
  elevation_shift: true
  shift_range: 5.0

# Inference
inference:
  use_tta: true
  tta_rotations: [0, 90, 180, 270]
```

---

## Training Data: 13 Diverse US Regions

### Original Regions (v1)
| Region | State | Terrain Type | Tiles |
|--------|-------|--------------|-------|
| Kansas | KS | Flat Plains | 225 |
| Appalachian | VA/WV | Moderate Hills | 225 |
| Colorado | CO | Steep Mountains | 225 |
| Oregon | OR | Coastal | 225 |
| Arizona | AZ | Valleys/Canyons | 225 |

### Extended Extreme Regions (v2)
| Region | State | Terrain Type | Tiles |
|--------|-------|--------------|-------|
| Death Valley | CA | Desert Basin (Low) | 208 |
| Glacier NP | MT | Glacial/Alpine | 273 |
| Utah Canyonlands | UT | Mountains & Canyons | 182 |
| Grand Canyon | AZ | Deep Canyon | 104 |
| Zion NP | UT | Sandstone Cliffs | 36 |
| Bryce Canyon | UT | Hoodoos/Erosional | 24 |
| Monument Valley | AZ/UT | Buttes & Mesas | 48 |
| Sedona | AZ | Red Rock Formations | 24 |

### Data Summary
| Split | Tiles | Percentage |
|-------|-------|------------|
| **Train** | 1,354 | 67% |
| **Validation** | 291 | 14% |
| **Test** | 379 | 19% |
| **Total** | **2,024** | 100% |

---

## Expected Improvements

| Metric | v1 Performance | Expected v2 |
|--------|----------------|-------------|
| Florida Slope MAE | 0.23° | <0.18° |
| Florida Curvature MAE | 0.00094 | <0.00075 |
| Utah Slope MAE | 2.58° | <2.0° |
| Generalization (new terrain) | Variable | More consistent |

---

## File Structure

```
DEM-SR-Model-v2/
├── SPECIFICATION.md          # This file
├── PROGRESS.md               # Phase tracking
├── configs/
│   └── default_v2.yaml       # Configuration
├── models/
│   ├── __init__.py
│   ├── layers_v2.py          # Phase 1-2: New layers
│   └── rcan_v2.py            # Phase 2: Updated RCAN
├── losses/
│   ├── __init__.py
│   ├── adaptive_v2.py        # Phase 3: Adaptive weights
│   ├── spectral.py           # Phase 4: Frequency loss
│   ├── edge_aware.py         # Phase 4: Edge loss
│   ├── multiscale.py         # Phase 4: Multi-scale loss
│   ├── complexity.py         # Phase 4: Complexity weighting
│   └── combined_v2.py        # Phase 4: Combined loss
├── data/
│   ├── __init__.py
│   ├── augmentation.py       # Phase 5: Enhanced augmentation
│   └── dataset_v2.py         # Phase 5: Updated dataset
├── scripts/
│   └── train_v2.py           # Phase 6: Updated training
├── inference_v2.py           # Phase 7: Improved inference
└── docs/
    └── CHANGELOG.md          # Version changes
```

---

## Implementation Order

1. **Phase 1:** Architecture - Regularization (layers_v2.py)
2. **Phase 2:** Architecture - Attention (rcan_v2.py)
3. **Phase 3:** Loss - Adaptive Weights (adaptive_v2.py)
4. **Phase 4:** Loss - New Components (spectral, edge, etc.)
5. **Phase 5:** Data - Augmentation (augmentation.py, dataset_v2.py)
6. **Phase 6:** Training - Process (train_v2.py)
7. **Phase 7:** Inference - TTA (inference_v2.py)

Each phase will be marked complete in `PROGRESS.md` as it's finished.
