# AAIGrid Integration Guide for FracAdapt Backend

**Document:** AAIGrid Output Format Integration  
**Date:** January 27, 2026  
**Author:** Alex Zare  
**Related:** DEM-SR Model v2, FracAdapt Backend

---

## Table of Contents

1. [Overview](#1-overview)
2. [Why AAIGrid?](#2-why-aaigrid)
3. [Changes Made](#3-changes-made)
4. [AAIGrid Format Specification](#4-aaigrid-format-specification)
5. [Usage Examples](#5-usage-examples)
6. [Integration Architecture](#6-integration-architecture)
7. [Technical Details](#7-technical-details)

---

## 1. Overview

The `inference_v2.py` script has been updated to output **AAIGrid (Arc ASCII Grid)** format by default, enabling seamless integration with the FracAdapt backend. This change ensures that the enhanced 10m DEM data can be directly consumed by the existing backend terrain analysis pipeline without any modifications.

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Zero Backend Changes** | FracAdapt backend parser works unchanged |
| **API Compatibility** | Same format as OpenTopography API output |
| **Human Readable** | ASCII format is easy to inspect and debug |
| **Geographic Accuracy** | Preserves lat/lng coordinates precisely |

---

## 2. Why AAIGrid?

### Current FracAdapt Data Flow

```
OpenTopography API ──────► AAIGrid (.asc) ──────► Backend Parser
     (30m SRTM)              Format                    │
                                                       ▼
                                              Terrain Analysis
                                              (slope, roughness,
                                               component stress)
```

### The Problem

The DEM-SR model was outputting **GeoTIFF** format, which requires either:
1. Adding a new GeoTIFF parser to the backend, OR
2. Converting GeoTIFF to AAIGrid after inference

### The Solution

**Output AAIGrid directly from the model inference script.**

```
OpenTopography API ──► AAIGrid ──► DEM-SR Model ──► AAIGrid ──► Backend Parser
     (30m)              (30m)         (3× SR)         (10m)          │
                                                                      ▼
                                                              Terrain Analysis
                                                              (9× more detail!)
```

---

## 3. Changes Made

### Modified File: `inference_v2.py`

#### 3.1 New Functions Added

| Function | Purpose |
|----------|---------|
| `load_aaigrid()` | Load DEM from AAIGrid format |
| `save_aaigrid()` | Save DEM to AAIGrid format |
| `save_geotiff()` | Save DEM to GeoTIFF format (renamed from `save_dem`) |

#### 3.2 Updated Functions

| Function | Change |
|----------|--------|
| `load_dem()` | Now auto-detects format (.asc/.txt vs .tif/.tiff) |
| `save_dem()` | Now dispatches to `save_aaigrid()` or `save_geotiff()` based on format |
| `main()` | Added `--format`, `--also_geotiff`, `--precision` arguments |

#### 3.3 New Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--format` | `aaigrid` | Output format: `aaigrid` or `geotiff` |
| `--also_geotiff` | None | Also save GeoTIFF to this path |
| `--precision` | 2 | Decimal places for AAIGrid values |

---

## 4. AAIGrid Format Specification

### Header Structure (6 lines)

```
ncols         1200
nrows         1200
xllcorner     -105.500000000
yllcorner     39.000000000
cellsize      0.000092592592593
nodata_value  -9999
```

| Field | Type | Description |
|-------|------|-------------|
| `ncols` | Integer | Number of columns (pixels) |
| `nrows` | Integer | Number of rows (pixels) |
| `xllcorner` | Float | West boundary longitude (lower-left X) |
| `yllcorner` | Float | South boundary latitude (lower-left Y) |
| `cellsize` | Float | Cell size in degrees |
| `nodata_value` | Float | Missing data sentinel value |

### Grid Data

- One row of space-separated elevation values per line
- Values start from **top-left** corner (north-west)
- Rows proceed **top to bottom** (north to south)
- Values are in **meters** (elevation above sea level)

### Example (4×4 grid)

```
ncols         4
nrows         4
xllcorner     -105.5
yllcorner     39.0
cellsize      0.00027778
nodata_value  -9999
1523.45 1524.12 1525.67 1526.89
1522.34 1523.01 1524.56 1525.78
1521.23 1521.90 1523.45 1524.67
1520.12 1520.79 1522.34 1523.56
```

---

## 5. Usage Examples

### Basic Inference (AAIGrid Output)

```bash
# Default: outputs AAIGrid format
python inference_v2.py \
    --checkpoint outputs_v2/checkpoints/best_model.pth \
    --input route_dem_30m.tif \
    --output route_dem_10m.asc \
    --device mps
```

### With Test-Time Augmentation

```bash
python inference_v2.py \
    --checkpoint best_model.pth \
    --input dem_30m.asc \
    --output dem_10m.asc \
    --tta \
    --device mps
```

### Output GeoTIFF Instead

```bash
python inference_v2.py \
    --checkpoint best_model.pth \
    --input dem_30m.tif \
    --output dem_10m.tif \
    --format geotiff
```

### Output Both Formats

```bash
python inference_v2.py \
    --checkpoint best_model.pth \
    --input dem_30m.tif \
    --output dem_10m.asc \
    --also_geotiff dem_10m.tif
```

### Higher Precision for Scientific Use

```bash
python inference_v2.py \
    --checkpoint best_model.pth \
    --input dem_30m.tif \
    --output dem_10m.asc \
    --precision 4
```

### Full Pipeline with Uncertainty

```bash
python inference_v2.py \
    --checkpoint best_model.pth \
    --input dem_30m.asc \
    --output dem_10m.asc \
    --tta \
    --uncertainty \
    --n_samples 10 \
    --uncertainty_output dem_10m_uncertainty.asc \
    --device mps
```

---

## 6. Integration Architecture

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FRACADAPT TERRAIN INTELLIGENCE                       │
│                         WITH DEM-SR MODEL INTEGRATION                        │
└─────────────────────────────────────────────────────────────────────────────┘

USER REQUEST: "Calculate route terrain from Denver to Aspen"
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: FETCH 30M DEM (OpenTopography API)                                 │
│  ─────────────────────────────────────────────                              │
│  GET /API/globaldem?demtype=SRTMGL1&south=39.0&north=39.5&west=-105.5...   │
│                                                                              │
│  Response: AAIGrid format                                                    │
│  ┌─────────────────────────────────┐                                        │
│  │ ncols         400               │                                        │
│  │ nrows         400               │                                        │
│  │ xllcorner     -105.5            │                                        │
│  │ yllcorner     39.0              │                                        │
│  │ cellsize      0.000277778       │ ← 30m resolution                       │
│  │ nodata_value  -9999             │                                        │
│  │ 2450.12 2451.34 2452.56 ...    │                                        │
│  └─────────────────────────────────┘                                        │
│  Save as: /tmp/route_30m.asc                                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: ENHANCE WITH DEM-SR MODEL                                          │
│  ────────────────────────────────────                                       │
│                                                                              │
│  python inference_v2.py \                                                   │
│      --checkpoint best_model.pth \                                          │
│      --input /tmp/route_30m.asc \                                          │
│      --output /tmp/route_10m.asc \                                         │
│      --tta --device mps                                                     │
│                                                                              │
│  ┌─────────────────┐              ┌─────────────────┐                       │
│  │  route_30m.asc  │    ────▶     │  route_10m.asc  │                       │
│  │  400 × 400      │   RCAN-DEM   │  1200 × 1200    │                       │
│  │  ~30m/pixel     │     v2       │  ~10m/pixel     │                       │
│  │  160K samples   │              │  1.44M samples  │                       │
│  └─────────────────┘              └─────────────────┘                       │
│                                                                              │
│  Output: AAIGrid format (backend-compatible!)                               │
│  ┌─────────────────────────────────┐                                        │
│  │ ncols         1200              │ ← 3× more columns                      │
│  │ nrows         1200              │ ← 3× more rows                         │
│  │ xllcorner     -105.5            │ ← Same geographic extent               │
│  │ yllcorner     39.0              │ ← Same geographic extent               │
│  │ cellsize      0.000092593       │ ← 10m resolution (1/3 of 30m)          │
│  │ nodata_value  -9999             │                                        │
│  │ 2450.12 2450.45 2450.78 ...    │ ← 9× more elevation samples            │
│  └─────────────────────────────────┘                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: PARSE WITH EXISTING BACKEND (NO CHANGES!)                          │
│  ────────────────────────────────────────────────────                       │
│                                                                              │
│  // elevationService.js - UNCHANGED CODE                                    │
│  parseAAIGrid(aaigridContent) {                                             │
│      const lines = aaigridContent.split('\n');                              │
│      const header = {};                                                      │
│      for (let i = 0; i < 6; i++) {                                          │
│          const [key, value] = lines[i].split(/\s+/);                        │
│          header[key.toLowerCase()] = parseFloat(value);                     │
│      }                                                                       │
│      // ... parse grid data                                                  │
│      return { header, grid };  // Works with 10m data too!                  │
│  }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: ENHANCED TERRAIN ANALYSIS                                          │
│  ──────────────────────────────────                                         │
│                                                                              │
│  With 10m resolution instead of 30m:                                        │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  METRIC              30m ACCURACY        10m ACCURACY               │    │
│  ├─────────────────────────────────────────────────────────────────────┤    │
│  │  Slope detection     >30m features       >10m features              │    │
│  │  Roughness           90m² windows        10m² windows               │    │
│  │  Fractal dimension   Lower accuracy      Higher accuracy            │    │
│  │  Suspension stress   Misses small bumps  Detects 10-30m bumps      │    │
│  │  Brake stress        Averaged slopes     Precise steep sections     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Result: More accurate component stress predictions!                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Resolution Comparison

```
30m RESOLUTION                          10m RESOLUTION
(Original SRTM)                         (DEM-SR Enhanced)

  ┌───┬───┬───┐                         ┌─┬─┬─┬─┬─┬─┬─┬─┬─┐
  │   │   │   │                         │ │ │ │ │ │ │ │ │ │
  ├───┼───┼───┤                         ├─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   │   │   │                         │ │ │ │ │ │ │ │ │ │
  ├───┼───┼───┤                         ├─┼─┼─┼─┼─┼─┼─┼─┼─┤
  │   │   │   │                         │ │ │ │ │ │ │ │ │ │
  └───┴───┴───┘                         ├─┼─┼─┼─┼─┼─┼─┼─┼─┤
                                        │ │ │ │ │ │ │ │ │ │
  9 elevation samples                   ├─┼─┼─┼─┼─┼─┼─┼─┼─┤
  in same area                          │ │ │ │ │ │ │ │ │ │
                                        ├─┼─┼─┼─┼─┼─┼─┼─┼─┤
                                        │ │ │ │ │ │ │ │ │ │
                                        ├─┼─┼─┼─┼─┼─┼─┼─┼─┤
                                        │ │ │ │ │ │ │ │ │ │
                                        ├─┼─┼─┼─┼─┼─┼─┼─┼─┤
                                        │ │ │ │ │ │ │ │ │ │
                                        └─┴─┴─┴─┴─┴─┴─┴─┴─┘
                                        
                                        81 elevation samples
                                        in same area (9× more!)
```

---

## 7. Technical Details

### Coordinate Preservation

The model preserves geographic coordinates by:

1. **Reading bounds** from input file (GeoTIFF or AAIGrid)
2. **Keeping same bounds** for output (same lat/lng extent)
3. **Calculating new cellsize** as `input_cellsize / scale`

```python
# Input: 30m resolution
cellsize_in = 0.000277778  # ~30m

# Output: 10m resolution (scale=3)
cellsize_out = cellsize_in / 3  # = 0.000092593 (~10m)

# Geographic extent UNCHANGED
bounds_in = bounds_out = {
    "left": -105.5,    # West longitude
    "right": -105.0,   # East longitude
    "bottom": 39.0,    # South latitude
    "top": 39.5        # North latitude
}
```

### Cellsize Calculation

For any pixel at row `i`, column `j`:

```
longitude = xllcorner + (j + 0.5) × cellsize
latitude  = yllcorner + (nrows - i - 0.5) × cellsize
```

### Memory Considerations

| Resolution | Grid Size | File Size (approx) |
|------------|-----------|-------------------|
| 30m | 400×400 | ~5 MB |
| 10m | 1200×1200 | ~45 MB |

The 10m file is ~9× larger but still manageable for typical route bounding boxes.

### Precision Recommendations

| Use Case | Recommended Precision | Example |
|----------|----------------------|---------|
| FracAdapt backend | 2 (default) | 1523.45 |
| Scientific analysis | 4 | 1523.4567 |
| Storage optimization | 1 | 1523.5 |
| Uncertainty maps | 4 | 0.1234 |

---

## Summary

The `inference_v2.py` script now outputs AAIGrid format by default, enabling seamless integration with the FracAdapt backend. The geographic coordinates (latitude/longitude) are precisely preserved, and the enhanced 10m resolution provides 9× more terrain data points for improved component stress predictions.

### Quick Reference

```bash
# For FracAdapt backend (default)
python inference_v2.py --checkpoint model.pth --input dem_30m.tif --output dem_10m.asc

# For GIS software
python inference_v2.py --checkpoint model.pth --input dem_30m.tif --output dem_10m.tif --format geotiff

# For both
python inference_v2.py --checkpoint model.pth --input dem_30m.tif --output dem_10m.asc --also_geotiff dem_10m.tif
```

---

*Document created: January 27, 2026*
