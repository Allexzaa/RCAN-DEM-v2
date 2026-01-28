# Future Improvements: Terrain-Aware Training Strategies

**Document:** Terrain-Aware Training for DEM Super-Resolution v3+  
**Date:** January 27, 2026  
**Author:** Alex Zare  
**Status:** Proposed for Future Implementation  
**Base Model:** RCAN-DEM v2 (15.6M parameters)

---

## Table of Contents

1. [Problem Analysis](#1-problem-analysis)
2. [Observed Training Behavior](#2-observed-training-behavior)
3. [Root Cause Analysis](#3-root-cause-analysis)
4. [Proposed Solutions](#4-proposed-solutions)
5. [Implementation Details](#5-implementation-details)
6. [Priority and Roadmap](#6-priority-and-roadmap)
7. [Expected Improvements](#7-expected-improvements)

---

## 1. Problem Analysis

### 1.1 Current Training Observation

During v2 training, we observed **loss spikes** that indicate the model struggles to simultaneously optimize for fundamentally different terrain types:

```
Epoch:  1     2     7     8    22    ...   87   100
Train: 2.67  3.33  3.65  3.78  0.91       0.60  0.58
            ↑     ↑     ↑     ↑
         Loss spikes when encountering "hard" terrain batches
```

### 1.2 The Core Issue

Different terrain types have **conflicting optimization targets**:

| Terrain Type | Desired Output | Loss Gradient Direction |
|--------------|----------------|------------------------|
| **Desert (Arizona, Grand Canyon)** | Sharp edges, bimodal slopes | "Preserve edges!" |
| **Glaciated (Glacier NP)** | Smooth curves, gradual transitions | "Smooth out!" |
| **Mixed (Colorado)** | Both features present | "Compromise..." |

When a batch contains predominantly one terrain type, the model adjusts strongly in that direction. The next batch with different terrain causes the loss to spike as the model readjusts.

### 1.3 Training Data Distribution

Current v2 training set (2,024 tiles):

| Category | Regions | Tiles | Percentage |
|----------|---------|-------|------------|
| **Flat** | Kansas | 225 | 11% |
| **Moderate** | Appalachian, Oregon | 450 | 22% |
| **Alpine/Mixed** | Colorado, Glacier | 498 | 25% |
| **Desert/Canyon** | AZ, Death Valley, Utah, Grand Canyon, Zion, Bryce, Monument, Sedona | 851 | **42%** |

**Observation:** Desert/canyon terrain is heavily represented (42%), which may bias the model toward sharp-edge features at the expense of smooth glacial terrain.

---

## 2. Observed Training Behavior

### 2.1 Loss Component Analysis

From v2 training history at epoch 100:

| Loss Component | Train | Val | Trend | Notes |
|----------------|-------|-----|-------|-------|
| **Total** | 0.580 | 0.645 | ↓ | Overall improving |
| **Elevation** | 0.062 | 0.074 | Stable | Core metric, good |
| **Gradient** | 0.008 | 0.011 | ↓ | Slope learning well |
| **Curvature** | 0.015 | 0.023 | ↓ | Shape learning well |
| **Spectral** | 4.98 | 5.47 | ↓ | Frequency detail improving |
| **Edge** | 0.063 | 0.075 | Stable | Edge preservation good |

### 2.2 Spike Pattern Analysis

```
Likely Spike Causes:

Epoch 2 spike (2.67 → 3.33):
  └─ Early training instability + warmup period
  
Epoch 7-8 spikes (3.65, 3.78):
  └─ Batch with extreme terrain (Grand Canyon, Bryce)
     after batches with moderate terrain
  
Epoch 22 spike (0.70 → 0.91):
  └─ Random shuffle presented difficult batch
     after model had stabilized on easier terrain
```

### 2.3 What the Model "Forgets"

```
When training primarily on desert terrain:
  Model learns: Sharp edges, cliff preservation, bimodal slopes
  Model weakens on: Smooth U-valley curves, gradual transitions
  
When training primarily on glaciated terrain:
  Model learns: Smooth curves, gradual slopes, flowing features
  Model weakens on: Sharp cliff edges, discrete elevation steps
```

---

## 3. Root Cause Analysis

### 3.1 Statistical Distribution Conflict

```
DESERT TERRAIN (e.g., Grand Canyon)
────────────────────────────────────
Slope Distribution:      Elevation Profile:
                        
Freq │ ■■■                    │      
     │ ■■■           ■■■      │      ┌────┐
     │ ■■■■          ■■■      │      │    │
     │ ■■■■■■■■■■■■■■■■■      │  ────┘    └────
     └───────────────────     └─────────────────
       0°  15°  30°  60°      Sharp steps, cliffs
       
     Bimodal: flat OR cliff


GLACIATED TERRAIN (e.g., Glacier NP)
────────────────────────────────────
Slope Distribution:      Elevation Profile:
                        
Freq │       ■■■■                ╭────────╮
     │      ■■■■■■              ╱          ╲
     │    ■■■■■■■■■           ╱            ╲
     │ ■■■■■■■■■■■■■■       ──╯              ╰──
     └───────────────────     
       0°  15°  30°  60°      Smooth U-valleys
       
     Normal: centered mid-range
```

### 3.2 Gradient Conflict in Training

When optimizing for both terrain types simultaneously:

```python
# Simplified gradient update scenario

# Batch 1: Desert terrain (sharp edges needed)
gradient_desert = compute_gradient(pred, target_desert)
# gradient_desert says: "Increase edge sharpness weights!"

# Batch 2: Glaciated terrain (smooth curves needed)
gradient_glaciated = compute_gradient(pred, target_glaciated)
# gradient_glaciated says: "Decrease edge sharpness, smooth more!"

# Model receives conflicting signals
# Result: Oscillating loss, slower convergence
```

### 3.3 Current Data Loader Behavior

```python
# Current: Random shuffling
train_loader = DataLoader(train_ds, shuffle=True)

# What happens:
# Batch 1: [Kansas, Arizona, Grand_Canyon, Colorado]  # Mixed difficulty
# Batch 2: [Bryce, Monument, Sedona, Death_Valley]    # ALL extreme desert!
# Batch 3: [Kansas, Kansas, Appalachian, Oregon]      # All easy/moderate
# Batch 4: [Glacier, Colorado, Zion, Grand_Canyon]    # Mixed

# The random shuffle can create batches with very uneven terrain representation
```

---

## 4. Proposed Solutions

### Solution A: Stratified Batching (Recommended - Low Effort)

**Concept:** Ensure each batch contains a balanced mix of terrain types.

```python
class StratifiedBatchSampler:
    """
    Ensures each batch has representation from all terrain categories.
    """
    def __init__(self, dataset, batch_size, terrain_categories):
        self.dataset = dataset
        self.batch_size = batch_size
        self.categories = terrain_categories
        
        # Group tiles by terrain type
        self.category_indices = {cat: [] for cat in terrain_categories}
        for idx, tile_id in enumerate(dataset.tile_ids):
            category = self.get_terrain_category(tile_id)
            self.category_indices[category].append(idx)
    
    def __iter__(self):
        # Create batches with one tile from each category (round-robin)
        iterators = {
            cat: iter(np.random.permutation(indices))
            for cat, indices in self.category_indices.items()
        }
        
        while True:
            batch = []
            for cat in self.categories:
                try:
                    batch.append(next(iterators[cat]))
                except StopIteration:
                    # Restart this category's iterator
                    iterators[cat] = iter(np.random.permutation(
                        self.category_indices[cat]
                    ))
                    batch.append(next(iterators[cat]))
                
                if len(batch) >= self.batch_size:
                    break
            
            yield batch
```

**Terrain Categories:**
| Category | Regions | Count |
|----------|---------|-------|
| `flat` | Kansas | 225 |
| `moderate` | Appalachian, Oregon | 450 |
| `alpine` | Colorado, Glacier | 498 |
| `desert` | Arizona, Death Valley | 433 |
| `extreme` | Grand Canyon, Zion, Bryce, Monument, Sedona | 418 |

**Expected Result:**
```
# With stratified batching:
# Batch 1: [Kansas, Appalachian, Colorado, Arizona]     # Balanced!
# Batch 2: [Kansas, Oregon, Glacier, Death_Valley]      # Balanced!
# Batch 3: [Kansas, Appalachian, Colorado, Grand_Canyon] # Balanced!
```

---

### Solution B: Curriculum Learning (Medium Effort)

**Concept:** Train on easy terrain first, progressively add harder terrain.

```python
class CurriculumScheduler:
    """
    Controls which terrain types are available during training.
    """
    def __init__(self, total_epochs=200):
        self.total_epochs = total_epochs
        
        # Curriculum phases
        self.phases = [
            # (start_epoch, end_epoch, terrain_types)
            (1, 50, ["flat", "moderate"]),           # Easy terrain
            (50, 100, ["flat", "moderate", "alpine"]), # Add alpine
            (100, 150, ["flat", "moderate", "alpine", "desert"]), # Add desert
            (150, 200, ["all"]),                      # Full dataset
        ]
    
    def get_active_terrain(self, epoch):
        """Return list of active terrain types for current epoch."""
        for start, end, terrains in self.phases:
            if start <= epoch < end:
                return terrains
        return ["all"]
    
    def get_data_subset(self, dataset, epoch):
        """Return indices of tiles to use for current epoch."""
        active_terrains = self.get_active_terrain(epoch)
        
        if "all" in active_terrains:
            return list(range(len(dataset)))
        
        indices = []
        for idx, tile_id in enumerate(dataset.tile_ids):
            terrain = self.get_terrain_type(tile_id)
            if terrain in active_terrains:
                indices.append(idx)
        
        return indices
```

**Phase Structure:**

```
Curriculum Training Schedule:
═══════════════════════════════════════════════════════════════════

Phase 1 (Epochs 1-50): FOUNDATION
├── Terrain: Kansas (flat), Appalachian, Oregon (moderate)
├── Purpose: Learn basic elevation reconstruction
├── Expected loss: Rapid decrease (2.5 → 0.8)
└── Tiles: 675 (33%)

Phase 2 (Epochs 50-100): COMPLEXITY
├── Terrain: + Colorado, Glacier (alpine)
├── Purpose: Learn slope variation, moderate complexity
├── Expected loss: Moderate decrease (0.8 → 0.65)
└── Tiles: 1,173 (58%)

Phase 3 (Epochs 100-150): CHALLENGE
├── Terrain: + Arizona, Death Valley (desert)
├── Purpose: Learn edge preservation, canyon features
├── Expected loss: Slower decrease (0.65 → 0.58)
└── Tiles: 1,606 (79%)

Phase 4 (Epochs 150-200): MASTERY
├── Terrain: ALL (+ Grand Canyon, Zion, Bryce, Monument, Sedona)
├── Purpose: Fine-tune on extreme terrain
├── Expected loss: Minimal decrease (0.58 → 0.55)
└── Tiles: 2,024 (100%)
```

---

### Solution C: Terrain-Aware Loss Weighting (Medium Effort)

**Concept:** Apply different loss weights based on terrain type.

```python
class TerrainAwareLoss(nn.Module):
    """
    Adjusts loss weights based on terrain characteristics.
    """
    def __init__(self):
        super().__init__()
        
        # Base loss functions
        self.elevation_loss = ElevationLoss()
        self.gradient_loss = GradientLoss()
        self.curvature_loss = CurvatureLoss()
        self.edge_loss = EdgeAwareLoss()
        self.spectral_loss = SpectralLoss()
        
        # Terrain-specific weight profiles
        self.terrain_weights = {
            "flat": {
                "elevation": 1.0, "gradient": 0.3, "curvature": 0.1,
                "edge": 0.1, "spectral": 0.1
            },
            "moderate": {
                "elevation": 1.0, "gradient": 0.5, "curvature": 0.2,
                "edge": 0.2, "spectral": 0.1
            },
            "alpine": {
                "elevation": 1.0, "gradient": 0.6, "curvature": 0.3,
                "edge": 0.2, "spectral": 0.15
            },
            "desert": {
                "elevation": 1.0, "gradient": 0.5, "curvature": 0.2,
                "edge": 0.4,  # Higher for edge preservation
                "spectral": 0.15
            },
            "extreme": {
                "elevation": 1.0, "gradient": 0.7, "curvature": 0.3,
                "edge": 0.5,  # Highest for extreme terrain
                "spectral": 0.2
            },
        }
    
    def forward(self, pred, target, terrain_type, mask=None):
        """
        Compute loss with terrain-specific weights.
        """
        weights = self.terrain_weights.get(terrain_type, self.terrain_weights["moderate"])
        
        loss = (
            weights["elevation"] * self.elevation_loss(pred, target, mask) +
            weights["gradient"] * self.gradient_loss(pred, target, mask) +
            weights["curvature"] * self.curvature_loss(pred, target, mask) +
            weights["edge"] * self.edge_loss(pred, target, mask) +
            weights["spectral"] * self.spectral_loss(pred, target, mask)
        )
        
        return loss
```

**Weight Rationale:**

| Terrain | Edge Weight | Why |
|---------|-------------|-----|
| Flat | 0.1 | Few edges to preserve |
| Moderate | 0.2 | Some ridgelines |
| Alpine | 0.2 | Mountain peaks need some edge |
| Desert | 0.4 | Cliffs and mesas need sharp edges |
| Extreme | 0.5 | Hoodoos, canyons need maximum edge preservation |

---

### Solution D: Domain-Specific Batch Normalization (High Effort)

**Concept:** Maintain separate normalization statistics for each terrain type.

```python
class TerrainAdaptiveBatchNorm(nn.Module):
    """
    Batch normalization with terrain-specific statistics.
    """
    def __init__(self, num_features, num_domains=5):
        super().__init__()
        
        # One BatchNorm per terrain domain
        self.bns = nn.ModuleDict({
            "flat": nn.BatchNorm2d(num_features),
            "moderate": nn.BatchNorm2d(num_features),
            "alpine": nn.BatchNorm2d(num_features),
            "desert": nn.BatchNorm2d(num_features),
            "extreme": nn.BatchNorm2d(num_features),
        })
        
        # Shared affine transformation
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
    
    def forward(self, x, terrain_type="moderate"):
        """
        Apply terrain-specific normalization.
        """
        bn = self.bns[terrain_type]
        
        # Get normalized output (without affine)
        bn.affine = False
        x_norm = bn(x)
        
        # Apply shared affine transformation
        x_out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        
        return x_out
```

**Architecture Integration:**

```
RCAN-DEM v3 with Domain-Specific BatchNorm:

Input (30m DEM) + Terrain Label
        │
        ▼
┌───────────────────────────────────────────┐
│  Residual Group 1                          │
│  ┌─────────┐  ┌─────────────────────────┐ │
│  │  Conv   │──│  TerrainAdaptiveBatchNorm│ │
│  └─────────┘  │  ├── BN_flat            │ │
│               │  ├── BN_moderate        │ │
│               │  ├── BN_alpine          │ │
│               │  ├── BN_desert          │ │
│               │  └── BN_extreme ←(select)│ │
│               └─────────────────────────┘ │
│                         │                  │
│                         ▼                  │
│              Channel Attention             │
└───────────────────────────────────────────┘
        │
       ... (repeat for all groups)
        │
        ▼
Output (10m DEM)
```

---

### Solution E: Gradient Accumulation by Terrain (Low Effort)

**Concept:** Accumulate gradients separately for each terrain type, then combine.

```python
def train_epoch_terrain_balanced(model, train_loader, optimizer, criterion):
    """
    Training with terrain-balanced gradient accumulation.
    """
    # Gradient accumulators per terrain type
    terrain_gradients = {
        "flat": [], "moderate": [], "alpine": [], "desert": [], "extreme": []
    }
    
    for batch in train_loader:
        lr, hr, terrain_types = batch["lr"], batch["hr"], batch["terrain"]
        
        # Forward pass
        pred = model(lr)
        loss = criterion(pred, hr)
        
        # Backward pass
        loss.backward()
        
        # Store gradients by terrain type
        for i, terrain in enumerate(terrain_types):
            terrain_gradients[terrain].append(
                {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
            )
        
        optimizer.zero_grad()
    
    # Average gradients across terrain types (equal contribution)
    balanced_gradients = {}
    for name in model.named_parameters():
        grads = []
        for terrain, grad_list in terrain_gradients.items():
            if grad_list:
                terrain_avg = torch.stack([g[name] for g in grad_list]).mean(0)
                grads.append(terrain_avg)
        
        if grads:
            balanced_gradients[name] = torch.stack(grads).mean(0)
    
    # Apply balanced gradients
    for name, param in model.named_parameters():
        if name in balanced_gradients:
            param.grad = balanced_gradients[name]
    
    optimizer.step()
```

---

## 5. Implementation Details

### 5.1 Terrain Classification Mapping

Add terrain metadata to the dataset:

```python
# File: data/terrain_metadata.py

TERRAIN_CLASSIFICATION = {
    # Region -> Terrain Category
    "kansas": "flat",
    "appalachian": "moderate",
    "oregon": "moderate",
    "colorado": "alpine",
    "glacier": "alpine",
    "arizona": "desert",
    "death_valley": "desert",
    "utah_canyonlands": "desert",
    "grand_canyon": "extreme",
    "zion": "extreme",
    "bryce_canyon": "extreme",
    "monument_valley": "extreme",
    "sedona": "extreme",
}

def get_terrain_category(tile_id: str) -> str:
    """
    Get terrain category from tile ID.
    
    Tile IDs follow pattern: region_tile_XXXXX
    """
    for region, category in TERRAIN_CLASSIFICATION.items():
        if region in tile_id.lower():
            return category
    return "moderate"  # Default fallback
```

### 5.2 Dataset Updates

```python
# Add to DEMDataset_v2.__getitem__()

def __getitem__(self, idx):
    tile_id = self.tile_ids[idx]
    
    # ... existing code ...
    
    # Add terrain category
    terrain_category = get_terrain_category(tile_id)
    
    output = {
        "lr": lr_tensor,
        "hr": hr_tensor,
        "tile_id": tile_id,
        "terrain": terrain_category,  # NEW
        "idx": idx,
    }
    
    return output
```

### 5.3 Configuration Updates

```yaml
# configs/default_v3.yaml

# Terrain-Aware Training
terrain_aware:
  enabled: true
  strategy: "stratified"  # "stratified", "curriculum", or "weighted_loss"
  
  # For stratified batching
  stratified:
    categories: ["flat", "moderate", "alpine", "desert", "extreme"]
    samples_per_category: 1  # Per batch
  
  # For curriculum learning
  curriculum:
    phases:
      - epochs: [1, 50]
        terrains: ["flat", "moderate"]
      - epochs: [50, 100]
        terrains: ["flat", "moderate", "alpine"]
      - epochs: [100, 150]
        terrains: ["flat", "moderate", "alpine", "desert"]
      - epochs: [150, 200]
        terrains: ["all"]
  
  # For terrain-aware loss
  terrain_loss_weights:
    flat:
      elevation: 1.0
      gradient: 0.3
      edge: 0.1
    extreme:
      elevation: 1.0
      gradient: 0.7
      edge: 0.5
```

---

## 6. Priority and Roadmap

### 6.1 Implementation Priority

| Solution | Effort | Impact | Priority | Phase |
|----------|--------|--------|----------|-------|
| **A. Stratified Batching** | Low | Medium-High | **1 (First)** | v3.0 |
| **E. Balanced Gradient Accumulation** | Low | Medium | 2 | v3.0 |
| **B. Curriculum Learning** | Medium | High | 3 | v3.1 |
| **C. Terrain-Aware Loss** | Medium | Medium-High | 4 | v3.1 |
| **D. Domain-Specific BatchNorm** | High | Medium | 5 | v3.2+ |

### 6.2 Implementation Roadmap

```
DEM-SR Model v3 Development:
════════════════════════════════════════════════════════════════

v3.0 - Terrain Balancing (Quick Wins)
├── Stratified batch sampler
├── Terrain metadata in dataset
├── Balanced gradient accumulation
└── Expected improvement: 5-10% smoother training, fewer spikes

v3.1 - Terrain Intelligence
├── Curriculum learning scheduler
├── Terrain-aware loss weights
├── Per-terrain validation metrics
└── Expected improvement: 10-15% better generalization

v3.2 - Advanced Domain Adaptation
├── Domain-specific batch normalization
├── Terrain embedding layer
├── Multi-task terrain prediction head
└── Expected improvement: 15-20% improvement on extreme terrain
```

### 6.3 Files to Create/Modify

| File | Changes | Solution |
|------|---------|----------|
| `data/terrain_metadata.py` | **New file** - Terrain classification | All |
| `data/dataset_v3.py` | Add terrain labels, stratified sampler | A, E |
| `data/curriculum.py` | **New file** - Curriculum scheduler | B |
| `losses/terrain_aware.py` | **New file** - Terrain-weighted loss | C |
| `models/layers_v3.py` | Add TerrainAdaptiveBatchNorm | D |
| `scripts/train_v3.py` | Integrate terrain-aware training | All |
| `configs/default_v3.yaml` | Add terrain configuration | All |

---

## 7. Expected Improvements

### 7.1 Training Stability

```
BEFORE (v2):                    AFTER (v3 with stratified batching):

Loss                            Loss
  │                               │
  │╲                              │╲
  │ ╲ ↑spike                      │ ╲
  │  ╲↑                           │  ╲__
  │   ╲__                         │     ╲___
  │      ╲↑spike                  │         ╲____
  │       ╲___                    │              ╲_____
  │           ╲_____              │
  └─────────────────              └─────────────────────
  
  Oscillating, spiky             Smooth, consistent descent
```

### 7.2 Performance Metrics (Projected)

| Metric | v2 | v3 (Projected) | Improvement |
|--------|----|----|-------------|
| Training stability (spike frequency) | ~5-10 per 100 epochs | <2 per 100 epochs | 70% fewer spikes |
| Convergence speed | 100 epochs to 0.58 | 80 epochs to 0.58 | 20% faster |
| Flat terrain RMSE | 0.23m | 0.20m | 13% better |
| Extreme terrain RMSE | 2.5m | 2.0m | 20% better |
| Cross-terrain generalization | Variable | Consistent | More predictable |

### 7.3 Per-Terrain Performance

| Terrain | v2 Expected | v3 with Curriculum | v3 with Full Stack |
|---------|-------------|-------------------|-------------------|
| Flat (Kansas) | Excellent | Excellent | Excellent |
| Moderate (Appalachian) | Good | Very Good | Excellent |
| Alpine (Glacier) | Good | Good | Very Good |
| Desert (Arizona) | Very Good | Very Good | Excellent |
| Extreme (Grand Canyon) | Moderate | Good | Very Good |

---

## Summary

The current v2 model shows healthy convergence but exhibits loss spikes due to conflicting optimization targets from different terrain types. The proposed terrain-aware training strategies address this by:

1. **Stratified Batching** - Ensures balanced terrain representation in each batch
2. **Curriculum Learning** - Progressively introduces harder terrain
3. **Terrain-Aware Loss** - Applies appropriate loss weights per terrain
4. **Domain-Specific BatchNorm** - Maintains separate statistics per terrain
5. **Balanced Gradient Accumulation** - Equal contribution from all terrain types

**Recommended first steps for v3:**
1. Implement terrain classification metadata
2. Add stratified batch sampler
3. Monitor per-terrain validation metrics

These improvements will result in smoother training, faster convergence, and better generalization across all terrain types - particularly important for the FracAdapt backend where routes may traverse multiple terrain types in a single mission.

---

*Document created: January 27, 2026*  
*For implementation in DEM-SR Model v3*
