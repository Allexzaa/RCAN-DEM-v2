# International Test Results: Afghanistan, Iraq, Vietnam

DEM-SR v2 was run on 30m Copernicus GLO-30 test samples from three regions, with TTA and AAIGrid output.

## Test regions

| Region     | Area (approx)        | Bounds (lat/lon)     | Terrain          |
|-----------|----------------------|----------------------|------------------|
| Afghanistan | Kabul region (0.25°×0.25°) | 34–34.25°N, 69–69.25°E | Mountainous      |
| Iraq      | Baghdad region       | 33–33.25°N, 44–44.25°E | Flat / low relief |
| Vietnam   | Hanoi region         | 21–21.25°N, 105–105.25°E | Mixed / coastal  |

## Results summary

| Region     | Input (30m) | Output (10m) | Elev min (m) | Elev max (m) | Elev mean (m) |
|-----------|-------------|--------------|--------------|--------------|---------------|
| Afghanistan | 901×901    | 2703×2703    | -1337.8      | 2743.2       | 2024.1        |
| Iraq      | 901×901     | 2703×2703    | -29.4        | 68.6         | 37.7          |
| Vietnam   | 901×901     | 2703×2703    | -491.7       | 1046.4       | 184.8         |

- **Model:** best checkpoint (epoch 189), 3× super-resolution, TTA (4 rotations).
- **Output format:** AAIGrid (`.asc`) for FracAdapt backend compatibility.

## Files in this folder

- **`*_10m_sr.asc`** – Super-resolved 10m DEM (AAIGrid).
- **`*_comparison.png`** – Side-by-side 30m input vs 10m SR and hillshades.
- **`*_elevation_profile.png`** – Elevation profile (center row): 30m vs 10m.
- **`summary_chart.png`** – Bar charts: elevation stats and output size by region.
- **`summary_results.json`** – Machine-readable summary and bounds.

## How to reproduce

From `DEM-SR-Model-v2`:

```bash
conda activate dem-sr
python scripts/run_international_test.py
```

Requires: AWS CLI (Copernicus DEM), GDAL (`gdal_translate`). To skip download and use existing 30m inputs: `--no-download`.
