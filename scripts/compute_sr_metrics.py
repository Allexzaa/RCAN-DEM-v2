#!/usr/bin/env python3
"""
Compute detailed metrics for 10m SR results vs 30m input.
Outputs a JSON of metrics for report generation.
"""
import json
import sys
from pathlib import Path

import numpy as np

# Add parent for rasterio if needed
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def load_geotiff(path):
    import rasterio
    with rasterio.open(str(path)) as src:
        return src.read(1), src.bounds

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
    return np.array(data, dtype=np.float32)

def downsample_3x(dem):
    """Average 3x3 blocks to get 30m-equivalent from 10m."""
    h, w = dem.shape
    h3, w3 = h // 3, w // 3
    out = np.zeros((h3, w3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            out += dem[i:h3*3:3, j:w3*3:3]
    return out / 9.0

def gradient_magnitude(dem):
    """Simple central-difference gradient magnitude."""
    gx = np.zeros_like(dem)
    gy = np.zeros_like(dem)
    gx[:, 1:-1] = (dem[:, 2:] - dem[:, :-2]) / 2.0
    gy[1:-1, :] = (dem[2:, :] - dem[:-2, :]) / 2.0
    return np.sqrt(gx**2 + gy**2)

def main():
    base = Path(__file__).resolve().parent.parent
    data_dir = base / "test_regions_international" / "30m"
    out_dir = base / "outputs_v2" / "inference_results" / "international"
    regions = ["afghanistan", "iraq", "vietnam"]
    metrics_all = {}
    for region in regions:
        lr_path = data_dir / f"{region}_30m.tif"
        sr_path = out_dir / f"{region}_10m_sr.asc"
        if not lr_path.exists() or not sr_path.exists():
            metrics_all[region] = {"error": "missing_file"}
            continue
        lr, _ = load_geotiff(lr_path)
        sr = load_aaigrid(sr_path)
        # Crop SR to multiple of 3
        scale = sr.shape[0] // lr.shape[0]
        sr_h, sr_w = lr.shape[0] * scale, lr.shape[1] * scale
        sr = sr[:sr_h, :sr_w]
        lr_valid = np.isfinite(lr) & (np.abs(lr) < 1e6)
        sr_valid = np.isfinite(sr) & (np.abs(sr) < 1e6)
        lr_flat = lr[lr_valid]
        sr_flat = sr[sr_valid]
        sr_ds = downsample_3x(sr)
        # Align shapes (downsample may be 1 pixel smaller)
        min_h = min(lr.shape[0], sr_ds.shape[0])
        min_w = min(lr.shape[1], sr_ds.shape[1])
        lr_c = lr[:min_h, :min_w]
        sr_ds_c = sr_ds[:min_h, :min_w]
        mask = np.isfinite(lr_c) & np.isfinite(sr_ds_c) & (np.abs(lr_c) < 1e6) & (np.abs(sr_ds_c) < 1e6)
        if mask.sum() < 10:
            metrics_all[region] = {"error": "insufficient_valid"}
            continue
        rmse_recon = np.sqrt(np.mean((lr_c[mask] - sr_ds_c[mask])**2))
        mae_recon = np.mean(np.abs(lr_c[mask] - sr_ds_c[mask]))
        corr = np.corrcoef(lr_c[mask].ravel(), sr_ds_c[mask].ravel())[0, 1] if mask.sum() > 1 else 0
        mean_lr = float(np.nanmean(lr_flat))
        mean_sr = float(np.nanmean(sr_flat))
        mean_diff = mean_sr - mean_lr
        grad_lr = gradient_magnitude(lr)
        grad_sr = gradient_magnitude(sr)
        grad_lr_valid = grad_lr[np.isfinite(grad_lr)]
        grad_sr_valid = grad_sr[np.isfinite(grad_sr)]
        metrics_all[region] = {
            "input_30m": {
                "shape": list(lr.shape),
                "elev_min_m": float(np.nanmin(lr_flat)),
                "elev_max_m": float(np.nanmax(lr_flat)),
                "elev_mean_m": mean_lr,
                "elev_std_m": float(np.nanstd(lr_flat)),
            },
            "output_10m": {
                "shape": list(sr.shape),
                "elev_min_m": float(np.nanmin(sr_flat)),
                "elev_max_m": float(np.nanmax(sr_flat)),
                "elev_mean_m": mean_sr,
                "elev_std_m": float(np.nanstd(sr_flat)),
            },
            "reconstruction_fidelity": {
                "rmse_30m_recon_m": float(rmse_recon),
                "mae_30m_recon_m": float(mae_recon),
                "correlation_input_vs_downsampled_sr": float(corr),
                "mean_elevation_difference_m": float(mean_diff),
            },
            "gradient": {
                "input_mean_grad_m_per_px": float(np.nanmean(grad_lr_valid)),
                "output_mean_grad_m_per_px": float(np.nanmean(grad_sr_valid)),
                "input_std_grad": float(np.nanstd(grad_lr_valid)),
                "output_std_grad": float(np.nanstd(grad_sr_valid)),
            },
        }
    out_path = out_dir / "sr_quality_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics_all, f, indent=2)
    print(f"Wrote {out_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
