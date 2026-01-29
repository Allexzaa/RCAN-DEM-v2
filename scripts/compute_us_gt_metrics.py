#!/usr/bin/env python3
"""
Compute US ground-truth metrics from existing 10m SR and 10m reference DEMs.
Writes outputs_v2/inference_results/us_ground_truth/us_ground_truth_metrics.json.

Run after run_us_ground_truth_test.py has produced *_10m_sr.asc and 10m_ref_clipped/*.tif.

Usage (from DEM-SR-Model-v2):
    conda run -n dem-sr python scripts/compute_us_gt_metrics.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

US_REGIONS = ["kansas", "colorado", "appalachian"]
REGION_NAMES = {
    "kansas": ("Great Plains, Kansas", "Flat"),
    "colorado": ("Rocky Mountains, Colorado", "Steep Mountains"),
    "appalachian": ("Appalachian Mountains, West Virginia", "Moderate Hills"),
}


def load_geotiff(path):
    import rasterio
    with rasterio.open(str(path)) as src:
        return src.read(1).astype(np.float32), (src.height, src.width)


def load_aaigrid(path):
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


def resample_to_shape(arr, target_shape):
    from scipy.ndimage import zoom
    sy = target_shape[0] / arr.shape[0]
    sx = target_shape[1] / arr.shape[1]
    return zoom(arr, (sy, sx), order=1, mode="nearest").astype(np.float32)


def downsample_3x(dem):
    h, w = dem.shape
    h3, w3 = h // 3, w // 3
    out = np.zeros((h3, w3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += dem[i : h3 * 3 : 3, j : w3 * 3 : 3]
    return out / 9.0


def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "outputs_v2" / "inference_results" / "us_ground_truth"
    clip_30_dir = out_dir / "30m_clipped"
    clip_10_dir = out_dir / "10m_ref_clipped"

    results = {"regions": {}}
    for region_id in US_REGIONS:
        sr_asc = out_dir / f"{region_id}_10m_sr.asc"
        clip_10_ref = clip_10_dir / f"{region_id}_10m_ref.tif"
        clip_30 = clip_30_dir / f"{region_id}_30m.tif"

        if not sr_asc.exists() or not clip_10_ref.exists():
            results["regions"][region_id] = {"error": "missing_sr_or_ref"}
            continue
        if not clip_30.exists():
            results["regions"][region_id] = {"error": "missing_30m_clipped"}
            continue

        sr_arr, _ = load_aaigrid(sr_asc)
        ref_arr, ref_shape = load_geotiff(clip_10_ref)
        lr_arr, _ = load_geotiff(clip_30)

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
        corr_gt = float(np.corrcoef(sr_flat, ref_flat)[0, 1])

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

        name, terrain_type = REGION_NAMES.get(region_id, (region_id, ""))

        results["regions"][region_id] = {
            "name": name,
            "terrain_type": terrain_type,
            "vs_ground_truth_10m": {"rmse_m": rmse_gt, "mae_m": mae_gt, "correlation": corr_gt},
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
