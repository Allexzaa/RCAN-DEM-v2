#!/usr/bin/env python3
"""
Download 30m DEM test samples from Afghanistan, Iraq, and Vietnam,
run DEM-SR v2 inference, and generate visualizations and results.

Requires: AWS CLI (for Copernicus DEM), GDAL (gdal_translate), conda env dem-sr.

Usage (from DEM-SR-Model-v2):
    conda run -n dem-sr python scripts/run_international_test.py
    conda run -n dem-sr python scripts/run_international_test.py --no-download  # skip download, use existing
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# Add parent for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Regions: small tiles (0.25° x 0.25°) for manageable inference
REGIONS = {
    "afghanistan": {
        "name": "Afghanistan (Kabul region)",
        "bounds": {"south": 34.0, "north": 34.25, "west": 69.0, "east": 69.25},
    },
    "iraq": {
        "name": "Iraq (Baghdad region)",
        "bounds": {"south": 33.0, "north": 33.25, "west": 44.0, "east": 44.25},
    },
    "vietnam": {
        "name": "Vietnam (Hanoi region)",
        "bounds": {"south": 21.0, "north": 21.25, "west": 105.0, "east": 105.25},
    },
}


def get_copernicus_tile_name(bounds):
    """Single 1° tile name for bounds (SW corner)."""
    lat = int(bounds["south"])
    lon = int(bounds["west"])
    lat_hem = "N" if lat >= 0 else "S"
    lon_hem = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{lat_hem}{abs(lat):02d}_00_{lon_hem}{abs(lon):03d}_00_DEM"


def download_region(region_id: str, base_dir: Path) -> Path | None:
    """Download 30m Copernicus DEM for one region. Returns path to clipped GeoTIFF or None."""
    info = REGIONS[region_id]
    bounds = info["bounds"]
    out_dir = base_dir / "30m"
    out_dir.mkdir(parents=True, exist_ok=True)
    tile_name = get_copernicus_tile_name(bounds)
    local_tif = out_dir / f"{tile_name}.tif"
    merged = out_dir / f"{region_id}_30m.tif"

    if merged.exists():
        return merged

    # Download from AWS (no sign-request)
    s3_path = f"s3://copernicus-dem-30m/{tile_name}/{tile_name}.tif"
    try:
        subprocess.run(
            ["aws", "s3", "cp", s3_path, str(local_tif), "--no-sign-request", "--quiet"],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  Download failed for {region_id}: {e}")
        return None

    # Clip to exact bounds
    subprocess.run(
        [
            "gdal_translate",
            "-projwin",
            str(bounds["west"]),
            str(bounds["north"]),
            str(bounds["east"]),
            str(bounds["south"]),
            str(local_tif),
            str(merged),
        ],
        check=True,
        capture_output=True,
    )
    if local_tif.exists() and local_tif != merged:
        local_tif.unlink()
    return merged


def run_inference(checkpoint: Path, input_tif: Path, output_asc: Path, device: str = "mps"):
    """Run inference_v2.py on input_tif, write AAIGrid to output_asc."""
    script_dir = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(script_dir / "inference_v2.py"),
        "--checkpoint", str(checkpoint),
        "--input", str(input_tif),
        "--output", str(output_asc),
        "--format", "aaigrid",
        "--tta",
        "--device", device,
    ]
    r = subprocess.run(cmd, cwd=str(script_dir))
    return r.returncode == 0


def load_geotiff(path: Path) -> tuple[np.ndarray, dict]:
    import rasterio
    with rasterio.open(str(path)) as src:
        dem = src.read(1)
        meta = {
            "bounds": src.bounds,
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
        }
    return dem, meta


def load_aaigrid(path: Path) -> tuple[np.ndarray, dict]:
    """Load AAIGrid; return array and simple meta (bounds from header)."""
    with open(path) as f:
        lines = [f.readline().strip() for _ in range(6)]
        header = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                k, v = parts[0].lower(), parts[1]
                header[k] = int(v) if k in ("ncols", "nrows") else float(v)
        ncols, nrows = header["ncols"], header["nrows"]
        xll, yll = header["xllcorner"], header["yllcorner"]
        cs = header["cellsize"]
        data = []
        for _ in range(nrows):
            line = f.readline()
            if not line:
                break
            row = [float(x) for x in line.strip().split()]
            if row:
                data.append(row)
    dem = np.array(data, dtype=np.float32)
    meta = {
        "bounds": (xll, yll, xll + ncols * cs, yll + nrows * cs),
        "height": nrows,
        "width": ncols,
    }
    return dem, meta


def hillshade(dem: np.ndarray, azimuth: float = 315, altitude: float = 45) -> np.ndarray:
    ls = LightSource(azdeg=azimuth, altdeg=altitude)
    return ls.hillshade(np.nan_to_num(dem, nan=0), vert_exag=1)


def visualize_region(
    region_id: str,
    lr_dem: np.ndarray,
    sr_dem: np.ndarray,
    out_dir: Path,
    title: str,
):
    """Create comparison figure and elevation profile."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Downsample SR to same grid as LR for same-extent comparison (optional: show SR at full res in center crop)
    scale = sr_dem.shape[0] // lr_dem.shape[0]
    sr_low = sr_dem[::scale, ::scale][: lr_dem.shape[0], : lr_dem.shape[1]]

    # Crop to center 50% for detail
    hy, hx = lr_dem.shape[0] // 2, lr_dem.shape[1] // 2
    size = min(hy, hx)  # square crop
    lr_c = lr_dem[hy - size : hy + size, hx - size : hx + size]
    sr_c = sr_dem[(hy - size) * scale : (hy + size) * scale, (hx - size) * scale : (hx + size) * scale]

    vmin = float(np.nanmin(lr_dem))
    vmax = float(np.nanmax(lr_dem))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{title}\nDEM-SR v2 International Test", fontsize=14, fontweight="bold")

    axes[0, 0].imshow(lr_c, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"Input 30m DEM\n{lr_c.shape[0]}×{lr_c.shape[1]} px")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sr_c, cmap="terrain", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"Super-Resolved 10m DEM\n{sr_c.shape[0]}×{sr_c.shape[1]} px")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(hillshade(lr_c), cmap="gray")
    axes[1, 0].set_title("30m Hillshade")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(hillshade(sr_c), cmap="gray")
    axes[1, 1].set_title("10m Hillshade")
    axes[1, 1].axis("off")

    plt.tight_layout()
    fig.savefig(out_dir / f"{region_id}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Elevation profile (middle row)
    mid_row_lr = lr_dem[lr_dem.shape[0] // 2, :]
    mid_row_sr = sr_dem[sr_dem.shape[0] // 2, :]
    x_lr = np.linspace(0, 1, len(mid_row_lr))
    x_sr = np.linspace(0, 1, len(mid_row_sr))

    fig2, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(x_lr, mid_row_lr, "b-", label="30m input", linewidth=1.5)
    ax.plot(x_sr, mid_row_sr, "r-", label="10m SR output", linewidth=0.8, alpha=0.9)
    ax.set_xlabel("Normalized distance")
    ax.set_ylabel("Elevation (m)")
    ax.set_title(f"{title} – Elevation profile (center row)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig2.savefig(out_dir / f"{region_id}_elevation_profile.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="International test: download, infer, visualize")
    parser.add_argument("--no-download", action="store_true", help="Skip download; use existing 30m files")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (default: outputs_v2/checkpoints/best_model.pth)")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_dir = base / "test_regions_international"
    out_dir = base / "outputs_v2" / "inference_results" / "international"
    checkpoint = Path(args.checkpoint) if args.checkpoint else base / "outputs_v2" / "checkpoints" / "best_model.pth"

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        return 1

    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for region_id, info in REGIONS.items():
        print(f"\n{'='*60}\n{info['name']}\n{'='*60}")

        # Download
        input_30m = data_dir / "30m" / f"{region_id}_30m.tif"
        if not args.no_download:
            path_30m = download_region(region_id, data_dir)
            if path_30m is None:
                print(f"  Skipping {region_id} (download failed)")
                results.append({"region": region_id, "success": False, "error": "download_failed"})
                continue
            input_30m = path_30m
        else:
            if not input_30m.exists():
                print(f"  Skip {region_id}: {input_30m} not found (run without --no-download)")
                results.append({"region": region_id, "success": False, "error": "no_input"})
                continue

        # Inference
        output_asc = out_dir / f"{region_id}_10m_sr.asc"
        print(f"  Running inference...")
        if not run_inference(checkpoint, input_30m, output_asc, args.device):
            print(f"  Inference failed for {region_id}")
            results.append({"region": region_id, "success": False, "error": "inference_failed"})
            continue

        # Load and visualize
        try:
            lr_dem, _ = load_geotiff(input_30m)
            sr_dem, _ = load_aaigrid(output_asc)
        except Exception as e:
            print(f"  Load failed: {e}")
            results.append({"region": region_id, "success": False, "error": str(e)})
            continue

        visualize_region(region_id, lr_dem, sr_dem, out_dir, info["name"])

        stats = {
            "region": region_id,
            "success": True,
            "input_shape": list(lr_dem.shape),
            "output_shape": list(sr_dem.shape),
            "elev_min_m": float(np.nanmin(sr_dem)),
            "elev_max_m": float(np.nanmax(sr_dem)),
            "elev_mean_m": float(np.nanmean(sr_dem)),
        }
        results.append(stats)
        print(f"  Output: {sr_dem.shape[0]}×{sr_dem.shape[1]} px, elev range {stats['elev_min_m']:.1f}–{stats['elev_max_m']:.1f} m")

    # Summary JSON
    summary_path = out_dir / "summary_results.json"
    with open(summary_path, "w") as f:
        json.dump({"regions": REGIONS, "results": results}, f, indent=2)

    # Summary bar chart (elevation stats across regions)
    success_results = [r for r in results if r.get("success") and "elev_min_m" in r]
    if success_results:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        regions = [r["region"].replace("_", " ").title() for r in success_results]
        elev_min = [r["elev_min_m"] for r in success_results]
        elev_max = [r["elev_max_m"] for r in success_results]
        elev_mean = [r["elev_mean_m"] for r in success_results]
        x = np.arange(len(regions))
        w = 0.25
        axes[0].bar(x - w, elev_min, w, label="Min (m)", color="steelblue")
        axes[0].bar(x, elev_mean, w, label="Mean (m)", color="green", alpha=0.8)
        axes[0].bar(x + w, elev_max, w, label="Max (m)", color="brown", alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(regions)
        axes[0].set_ylabel("Elevation (m)")
        axes[0].set_title("Elevation statistics by region")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis="y")
        axes[1].bar(regions, [r["output_shape"][0] for r in success_results], color="teal", alpha=0.8)
        axes[1].set_ylabel("Output size (px)")
        axes[1].set_title("Super-resolved grid size")
        axes[1].tick_params(axis="x", rotation=15)
        plt.suptitle("DEM-SR v2 International Test Results", fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(out_dir / "summary_chart.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\nResults written to {out_dir}")
    print(f"Summary: {summary_path}")
    print(f"Figures: {out_dir}/*.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
