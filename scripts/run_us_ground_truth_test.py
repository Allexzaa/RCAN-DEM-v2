#!/usr/bin/env python3
"""
Run DEM-SR v2 on US regions that have real 10m (USGS 3DEP) ground truth.
Downloads or uses existing raw_data (30m Copernicus + 10m USGS), clips to 0.25°,
runs inference, and computes metrics vs ground truth and reconstruction fidelity.

Requires: raw_data/30m/{region}_30m.tif and raw_data/10m/{region}_10m.tif.
Create them by running from DEM-SR-Model: python scripts/download_dem_data.py --regions kansas colorado appalachian
(Optionally --test_mode for smaller areas; this script clips to 0.25° for consistency with international.)

Usage (from DEM-SR-Model-v2):
    conda run -n dem-sr python scripts/run_us_ground_truth_test.py
    conda run -n dem-sr python scripts/run_us_ground_truth_test.py --data_dir /path/to/raw_data
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# Add parent for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 0.25° x 0.25° tiles (same scale as international test) within US region bounds
US_REGIONS = {
    "kansas": {
        "name": "Great Plains, Kansas",
        "terrain_type": "Flat",
        "bounds": {"south": 38.0, "north": 38.25, "west": -100.0, "east": -99.75},
    },
    "colorado": {
        "name": "Rocky Mountains, Colorado",
        "terrain_type": "Steep Mountains",
        "bounds": {"south": 39.0, "north": 39.25, "west": -106.0, "east": -105.75},
    },
    "appalachian": {
        "name": "Appalachian Mountains, West Virginia",
        "terrain_type": "Moderate Hills",
        "bounds": {"south": 38.0, "north": 38.25, "west": -80.0, "east": -79.75},
    },
}


def clip_geotiff(input_path: Path, output_path: Path, bounds: dict) -> bool:
    """Clip GeoTIFF to bounds (west, north, east, south)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdal_translate",
        "-projwin",
        str(bounds["west"]),
        str(bounds["north"]),
        str(bounds["east"]),
        str(bounds["south"]),
        str(input_path),
        str(output_path),
    ]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0


def run_inference(checkpoint: Path, input_tif: Path, output_asc: Path, device: str = "cpu") -> bool:
    """Run inference_v2.py; write AAIGrid to output_asc."""
    script_dir = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(script_dir / "inference_v2.py"),
        "--checkpoint",
        str(checkpoint),
        "--input",
        str(input_tif),
        "--output",
        str(output_asc),
        "--format",
        "aaigrid",
        "--tta",
        "--device",
        device,
    ]
    r = subprocess.run(cmd, cwd=str(script_dir))
    return r.returncode == 0


def load_geotiff(path: Path):
    import rasterio
    with rasterio.open(str(path)) as src:
        return src.read(1).astype(np.float32), src.bounds, (src.height, src.width)


def load_aaigrid(path: Path):
    """Load AAIGrid; return array and meta."""
    with open(path) as f:
        lines = [f.readline().strip() for _ in range(6)]
        header = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                k, v = parts[0].lower(), parts[1]
                header[k] = int(v) if k in ("ncols", "nrows") else float(v)
        ncols, nrows = header["ncols"], header["nrows"]
        data = []
        for _ in range(nrows):
            line = f.readline()
            if not line:
                break
            row = [float(x) for x in line.strip().split()]
            if row:
                data.append(row)
    dem = np.array(data, dtype=np.float32)
    return dem, (dem.shape[0], dem.shape[1])


def resample_to_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Resample array to target (height, width) with linear interpolation."""
    from scipy.ndimage import zoom
    sy = target_shape[0] / arr.shape[0]
    sx = target_shape[1] / arr.shape[1]
    return zoom(arr, (sy, sx), order=1, mode="nearest").astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="US ground-truth test: inference + metrics")
    parser.add_argument("--data_dir", type=str, default=None, help="raw_data dir (default: ../raw_data)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best checkpoint")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--regions", nargs="+", default=list(US_REGIONS.keys()))
    parser.add_argument("--skip_inference", action="store_true", help="Only compute metrics from existing outputs")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else (base.parent / "raw_data")
    out_dir = base / "outputs_v2" / "inference_results" / "us_ground_truth"
    out_dir.mkdir(parents=True, exist_ok=True)
    clip_30_dir = out_dir / "30m_clipped"
    clip_10_dir = out_dir / "10m_ref_clipped"

    checkpoint = args.checkpoint
    if not checkpoint:
        ckpt_dir = base / "outputs_v2" / "checkpoints"
        best = ckpt_dir / "best_model.pth"
        if best.exists():
            checkpoint = str(best)
        else:
            for p in sorted(ckpt_dir.glob("epoch_*.pth"), reverse=True):
                checkpoint = str(p)
                break
    if not checkpoint or not Path(checkpoint).exists():
        print("ERROR: No checkpoint found. Set --checkpoint or ensure outputs_v2/checkpoints/best_model.pth exists.")
        return 1

    results = {"regions": {}, "checkpoint": checkpoint}
    for region_id in args.regions:
        if region_id not in US_REGIONS:
            continue
        info = US_REGIONS[region_id]
        bounds = info["bounds"]
        path_30m_full = data_dir / "30m" / f"{region_id}_30m.tif"
        path_10m_full = data_dir / "10m" / f"{region_id}_10m.tif"

        if not path_30m_full.exists():
            print(f"  Skip {region_id}: missing {path_30m_full}")
            results["regions"][region_id] = {"error": "missing_30m"}
            continue
        if not path_10m_full.exists():
            print(f"  Skip {region_id}: missing {path_10m_full}")
            results["regions"][region_id] = {"error": "missing_10m"}
            continue

        clip_30 = clip_30_dir / f"{region_id}_30m.tif"
        clip_10_ref = clip_10_dir / f"{region_id}_10m_ref.tif"
        sr_asc = out_dir / f"{region_id}_10m_sr.asc"

        if not args.skip_inference:
            if not clip_30.exists():
                if not clip_geotiff(path_30m_full, clip_30, bounds):
                    print(f"  Failed to clip 30m for {region_id}")
                    results["regions"][region_id] = {"error": "clip_30m_failed"}
                    continue
            if not clip_10_ref.exists():
                if not clip_geotiff(path_10m_full, clip_10_ref, bounds):
                    print(f"  Failed to clip 10m ref for {region_id}")
                    results["regions"][region_id] = {"error": "clip_10m_failed"}
                    continue
            print(f"  Running inference for {region_id}...")
            if not run_inference(Path(checkpoint), clip_30, sr_asc, args.device):
                results["regions"][region_id] = {"error": "inference_failed"}
                continue

        if not sr_asc.exists() or not clip_10_ref.exists():
            results["regions"][region_id] = {"error": "missing_sr_or_ref"}
            continue

        # Compute metrics
        sr_arr, sr_shape = load_aaigrid(sr_asc)
        ref_arr, ref_bounds, ref_shape = load_geotiff(clip_10_ref)
        lr_arr, _, lr_shape = load_geotiff(clip_30)

        # Align ref to SR grid (same extent; ref may have different pixel count)
        if ref_arr.shape != sr_arr.shape:
            ref_arr = resample_to_shape(ref_arr, sr_arr.shape)

        valid = np.isfinite(sr_arr) & np.isfinite(ref_arr) & (np.abs(sr_arr) < 1e6) & (np.abs(ref_arr) < 1e6)
        if valid.sum() < 100:
            results["regions"][region_id] = {"error": "insufficient_valid_pixels"}
            continue

        sr_flat = sr_arr[valid]
        ref_flat = ref_arr[valid]
        diff = sr_flat - ref_flat
        rmse_gt = float(np.sqrt(np.mean(diff**2)))
        mae_gt = float(np.mean(np.abs(diff)))
        corr_gt = float(np.corrcoef(sr_flat, ref_flat)[0, 1]) if valid.sum() > 1 else 0.0

        # Reconstruction fidelity (downsampled SR vs 30m input)
        def downsample_3x(dem):
            h, w = dem.shape
            h3, w3 = h // 3, w // 3
            out = np.zeros((h3, w3), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    out += dem[i : h3 * 3 : 3, j : w3 * 3 : 3]
            return out / 9.0

        sr_ds = downsample_3x(sr_arr)
        min_h = min(lr_arr.shape[0], sr_ds.shape[0])
        min_w = min(lr_arr.shape[1], sr_ds.shape[1])
        lr_c = lr_arr[:min_h, :min_w]
        sr_ds_c = sr_ds[:min_h, :min_w]
        mask_r = np.isfinite(lr_c) & np.isfinite(sr_ds_c) & (np.abs(lr_c) < 1e6) & (np.abs(sr_ds_c) < 1e6)
        if mask_r.sum() < 10:
            rmse_recon = mae_recon = corr_recon = mean_diff_recon = None
        else:
            rmse_recon = float(np.sqrt(np.mean((lr_c[mask_r] - sr_ds_c[mask_r]) ** 2)))
            mae_recon = float(np.mean(np.abs(lr_c[mask_r] - sr_ds_c[mask_r])))
            corr_recon = float(np.corrcoef(lr_c[mask_r].ravel(), sr_ds_c[mask_r].ravel())[0, 1])
            mean_diff_recon = float(np.mean(sr_ds_c[mask_r]) - np.mean(lr_c[mask_r]))

        results["regions"][region_id] = {
            "name": info["name"],
            "terrain_type": info["terrain_type"],
            "vs_ground_truth_10m": {
                "rmse_m": rmse_gt,
                "mae_m": mae_gt,
                "correlation": corr_gt,
            },
            "reconstruction_fidelity": {
                "rmse_30m_recon_m": rmse_recon,
                "mae_30m_recon_m": mae_recon,
                "correlation_input_vs_downsampled_sr": corr_recon,
                "mean_elevation_difference_m": mean_diff_recon,
            },
            "elevation": {
                "sr_10m_mean_m": float(np.nanmean(sr_flat)),
                "ref_10m_mean_m": float(np.nanmean(ref_flat)),
                "mean_bias_sr_minus_ref_m": float(np.nanmean(diff)),
            },
        }

    out_path = out_dir / "us_ground_truth_metrics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
