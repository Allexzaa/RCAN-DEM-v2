#!/usr/bin/env python3
"""Create 30m DEM sample. Output: testing/sample_data/dem_30m_sample.tif. Run from repo root: python testing/create_sample_dem.py"""
import sys
from pathlib import Path
import numpy as np
TESTING_DIR = Path(__file__).resolve().parent
OUT_DIR = TESTING_DIR / "sample_data"
OUT_PATH = OUT_DIR / "dem_30m_sample.tif"

def main():
    try:
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds
    except ImportError:
        print("rasterio is required: pip install rasterio")
        sys.exit(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    width, height = 256, 256
    left, bottom = -106.0, 39.0
    cellsize_deg = 30.0 / 111320
    right = left + width * cellsize_deg
    top = bottom + height * cellsize_deg
    np.random.seed(42)
    y = np.linspace(0, 1, height)
    x = np.linspace(0, 1, width)
    ramp = 200 + 300 * (y[:, None] + x[None, :]) / 2
    noise = 20 * np.random.randn(height, width).astype(np.float32)
    dem = np.clip(ramp + noise, 0, 600).astype(np.float32)
    transform = from_bounds(left, bottom, right, top, width, height)
    crs = CRS.from_epsg(4326)
    with rasterio.open(OUT_PATH, "w", driver="GTiff", height=height, width=width, count=1, dtype=dem.dtype, crs=crs, transform=transform, nodata=-9999) as dst:
        dst.write(dem, 1)
    print(f"Created {OUT_PATH}")
    return 0
if __name__ == "__main__":
    sys.exit(main())
