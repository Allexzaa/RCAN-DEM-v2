#!/usr/bin/env python3
"""
Phase 7: Inference & Deployment v2 for DEM Super-Resolution

Enhanced inference with:
- Test-Time Augmentation (TTA)
- Uncertainty estimation (Monte Carlo Dropout)
- Improved sliding window blending

Usage:
    # Basic inference
    python inference_v2.py --checkpoint outputs_v2/checkpoints/best_model.pth --input dem_30m.tif --output dem_10m_sr.tif
    
    # With TTA
    python inference_v2.py --checkpoint best_model.pth --input dem_30m.tif --output dem_10m.tif --tta
    
    # With uncertainty estimation
    python inference_v2.py --checkpoint best_model.pth --input dem_30m.tif --output dem_10m.tif --uncertainty
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.transform import from_bounds
from tqdm import tqdm

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from models import create_model_v2


def load_model(checkpoint_path: str, device: str = "cpu") -> Tuple[torch.nn.Module, dict]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    
    model = create_model_v2(
        model_type=model_config.get("type", "rcan_v2"),
        scale=model_config.get("scale", 3),
        n_resgroups=model_config.get("n_resgroups", 10),
        n_resblocks=model_config.get("n_resblocks", 20),
        n_feats=model_config.get("n_feats", 64),
        dropout=model_config.get("dropout", 0.1),
        use_spatial_attention=model_config.get("use_spatial_attention", False),
        use_multiscale=model_config.get("use_multiscale", False),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    return model, config


def create_blend_weights(h: int, w: int, overlap: int, device: str = "cpu") -> torch.Tensor:
    """Create 2D weight map for blending overlapping tiles."""
    y_weights = torch.ones(h, device=device)
    x_weights = torch.ones(w, device=device)
    
    if overlap > 0:
        ramp = torch.linspace(0.01, 1.0, overlap, device=device)
        y_weights[:overlap] = ramp
        y_weights[-overlap:] = ramp.flip(0)
        x_weights[:overlap] = ramp
        x_weights[-overlap:] = ramp.flip(0)
    
    return y_weights.unsqueeze(1) * x_weights.unsqueeze(0)


class TTAAugmentation:
    """Test-Time Augmentation handler."""
    
    def __init__(
        self,
        rotations: List[int] = [0, 90, 180, 270],
        use_flips: bool = False,
    ):
        self.rotations = rotations
        self.use_flips = use_flips
    
    def augment(self, x: torch.Tensor) -> List[Tuple[torch.Tensor, dict]]:
        """Generate augmented versions of input."""
        augmented = []
        
        for rot in self.rotations:
            x_rot = self._rotate(x, rot)
            augmented.append((x_rot, {"rotation": rot, "flip": None}))
            
            if self.use_flips:
                # Horizontal flip
                x_flip_h = torch.flip(x_rot, dims=[-1])
                augmented.append((x_flip_h, {"rotation": rot, "flip": "h"}))
                
                # Vertical flip
                x_flip_v = torch.flip(x_rot, dims=[-2])
                augmented.append((x_flip_v, {"rotation": rot, "flip": "v"}))
        
        return augmented
    
    def reverse(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Reverse augmentation."""
        # Reverse flip first
        if params.get("flip") == "h":
            x = torch.flip(x, dims=[-1])
        elif params.get("flip") == "v":
            x = torch.flip(x, dims=[-2])
        
        # Reverse rotation
        rot = params.get("rotation", 0)
        if rot != 0:
            x = self._rotate(x, -rot)
        
        return x
    
    def _rotate(self, x: torch.Tensor, angle: int) -> torch.Tensor:
        """Rotate tensor by angle degrees (90, 180, 270)."""
        k = (angle // 90) % 4
        if k == 0:
            return x
        return torch.rot90(x, k, dims=[-2, -1])


def inference_with_tta(
    model: torch.nn.Module,
    tile: torch.Tensor,
    tta: TTAAugmentation,
    device: str = "cpu",
) -> torch.Tensor:
    """Run inference with test-time augmentation."""
    augmented = tta.augment(tile)
    
    predictions = []
    for aug_tile, params in augmented:
        with torch.no_grad():
            pred = model(aug_tile.to(device))
        pred = tta.reverse(pred, params)
        predictions.append(pred)
    
    # Average predictions
    stacked = torch.stack(predictions, dim=0)
    return stacked.mean(dim=0)


def inference_with_uncertainty(
    model: torch.nn.Module,
    tile: torch.Tensor,
    n_samples: int = 10,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run inference with Monte Carlo Dropout for uncertainty estimation.
    
    Returns:
        Tuple of (mean prediction, uncertainty map)
    """
    model.train()  # Enable dropout
    
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            pred = model(tile.to(device))
        predictions.append(pred)
    
    model.eval()  # Disable dropout
    
    stacked = torch.stack(predictions, dim=0)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    
    return mean, std


def inference_sliding_window_v2(
    model: torch.nn.Module,
    lr_dem: np.ndarray,
    tile_size: int = 256,
    overlap: int = 32,
    scale: int = 3,
    device: str = "cpu",
    norm_mode: str = "global",
    use_tta: bool = False,
    tta_rotations: List[int] = [0, 90, 180, 270],
    use_uncertainty: bool = False,
    n_uncertainty_samples: int = 10,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Perform inference with sliding window, TTA, and optional uncertainty.
    
    Returns:
        Tuple of (super-resolved DEM, uncertainty map or None)
    """
    lr_h, lr_w = lr_dem.shape
    sr_h, sr_w = lr_h * scale, lr_w * scale
    
    # Handle small inputs
    if lr_h < tile_size or lr_w < tile_size:
        pad_h = max(0, tile_size - lr_h)
        pad_w = max(0, tile_size - lr_w)
        lr_dem = np.pad(lr_dem, ((0, pad_h), (0, pad_w)), mode='reflect')
        lr_h, lr_w = lr_dem.shape
        sr_h_orig, sr_w_orig = (lr_h - pad_h) * scale, (lr_w - pad_w) * scale
    else:
        sr_h_orig, sr_w_orig = sr_h, sr_w
    
    # Initialize output arrays
    output_sum = np.zeros((lr_h * scale, lr_w * scale), dtype=np.float64)
    weight_sum = np.zeros((lr_h * scale, lr_w * scale), dtype=np.float64)
    
    if use_uncertainty:
        uncertainty_sum = np.zeros((lr_h * scale, lr_w * scale), dtype=np.float64)
    
    # Create blend weights
    weights = create_blend_weights(
        tile_size * scale, tile_size * scale, overlap * scale, device
    ).cpu().numpy()
    
    # Global normalization
    if norm_mode == "global":
        global_mean = lr_dem.mean()
        global_std = lr_dem.std() + 1e-6
        lr_norm = (lr_dem - global_mean) / global_std
    else:
        lr_norm = lr_dem.copy()
    
    # TTA setup
    if use_tta:
        tta = TTAAugmentation(rotations=tta_rotations, use_flips=False)
    
    # Calculate tile positions
    stride = tile_size - overlap
    
    y_positions = list(range(0, lr_h - tile_size + 1, stride))
    x_positions = list(range(0, lr_w - tile_size + 1, stride))
    
    if y_positions and y_positions[-1] + tile_size < lr_h:
        y_positions.append(lr_h - tile_size)
    if x_positions and x_positions[-1] + tile_size < lr_w:
        x_positions.append(lr_w - tile_size)
    
    if not y_positions:
        y_positions = [0]
    if not x_positions:
        x_positions = [0]
    
    total_tiles = len(y_positions) * len(x_positions)
    
    # Process tiles
    pbar = tqdm(total=total_tiles, desc="Processing tiles")
    
    for y in y_positions:
        for x in x_positions:
            # Extract tile
            if norm_mode == "per_tile":
                tile_raw = lr_dem[y:y+tile_size, x:x+tile_size]
                tile_mean = tile_raw.mean()
                tile_std = tile_raw.std() + 1e-6
                tile = (tile_raw - tile_mean) / tile_std
            else:
                tile = lr_norm[y:y+tile_size, x:x+tile_size]
            
            # To tensor
            tile_tensor = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float()
            
            # Inference
            if use_uncertainty:
                sr_tile, uncertainty = inference_with_uncertainty(
                    model, tile_tensor, n_uncertainty_samples, device
                )
                sr_tile = sr_tile.squeeze().cpu().numpy()
                uncertainty = uncertainty.squeeze().cpu().numpy()
            elif use_tta:
                sr_tile = inference_with_tta(model, tile_tensor, tta, device)
                sr_tile = sr_tile.squeeze().cpu().numpy()
                uncertainty = None
            else:
                with torch.no_grad():
                    sr_tile = model(tile_tensor.to(device))
                sr_tile = sr_tile.squeeze().cpu().numpy()
                uncertainty = None
            
            # Denormalize
            if norm_mode == "per_tile":
                sr_tile = sr_tile * tile_std + tile_mean
            
            # Accumulate
            y_out, x_out = y * scale, x * scale
            output_sum[y_out:y_out+tile_size*scale, x_out:x_out+tile_size*scale] += sr_tile * weights
            weight_sum[y_out:y_out+tile_size*scale, x_out:x_out+tile_size*scale] += weights
            
            if use_uncertainty and uncertainty is not None:
                uncertainty_sum[y_out:y_out+tile_size*scale, x_out:x_out+tile_size*scale] += uncertainty * weights
            
            pbar.update(1)
    
    pbar.close()
    
    # Normalize by weights
    sr_dem = output_sum / (weight_sum + 1e-10)
    
    # Denormalize (global mode)
    if norm_mode == "global":
        sr_dem = sr_dem * global_std + global_mean
    
    # Crop to original size
    sr_dem = sr_dem[:sr_h_orig, :sr_w_orig]
    
    if use_uncertainty:
        uncertainty_map = uncertainty_sum / (weight_sum + 1e-10)
        uncertainty_map = uncertainty_map[:sr_h_orig, :sr_w_orig]
        return sr_dem.astype(np.float32), uncertainty_map.astype(np.float32)
    
    return sr_dem.astype(np.float32), None


def load_dem(input_path: str) -> Tuple[np.ndarray, dict]:
    """Load DEM from GeoTIFF file."""
    with rasterio.open(input_path) as src:
        dem = src.read(1)
        metadata = {
            "crs": src.crs,
            "transform": src.transform,
            "bounds": src.bounds,
            "nodata": src.nodata,
            "width": src.width,
            "height": src.height,
        }
    return dem, metadata


def save_dem(
    dem: np.ndarray,
    output_path: str,
    crs,
    bounds,
    nodata: Optional[float] = None,
):
    """Save DEM to GeoTIFF file."""
    height, width = dem.shape
    transform = from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top,
        width, height
    )
    
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height, width=width,
        count=1, dtype=dem.dtype,
        crs=crs, transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(dem, 1)
    
    print(f"Saved: {output_path} ({height} x {width})")


def main():
    parser = argparse.ArgumentParser(
        description="DEM Super-Resolution Inference v2",
    )
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Input LR DEM")
    parser.add_argument("--output", type=str, required=True, help="Output SR DEM")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size")
    parser.add_argument("--overlap", type=int, default=32, help="Tile overlap")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--norm_mode", type=str, default="global", choices=["global", "per_tile"])
    
    # TTA options
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--tta_rotations", type=str, default="0,90,180,270", help="TTA rotations")
    
    # Uncertainty options
    parser.add_argument("--uncertainty", action="store_true", help="Enable uncertainty estimation")
    parser.add_argument("--n_samples", type=int, default=10, help="MC dropout samples")
    parser.add_argument("--uncertainty_output", type=str, default=None, help="Uncertainty map output")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DEM Super-Resolution Inference v2")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"TTA: {args.tta}")
    print(f"Uncertainty: {args.uncertainty}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model, config = load_model(args.checkpoint, args.device)
    scale = config.get("model", {}).get("scale", 3)
    print(f"Model loaded. Scale: {scale}x")
    
    # Load input
    print(f"\nLoading input: {args.input}")
    lr_dem, metadata = load_dem(args.input)
    print(f"Input shape: {lr_dem.shape}")
    
    # Handle nodata
    nodata = metadata.get("nodata")
    if nodata is not None:
        nodata_mask = lr_dem == nodata
        if nodata_mask.any():
            print(f"NoData pixels: {nodata_mask.sum()}")
            lr_dem = np.where(nodata_mask, np.nan, lr_dem)
            lr_dem = np.nan_to_num(lr_dem, nan=np.nanmean(lr_dem))
    
    # Parse TTA rotations
    tta_rotations = [int(r) for r in args.tta_rotations.split(",")]
    
    # Run inference
    print("\nRunning inference...")
    sr_dem, uncertainty_map = inference_sliding_window_v2(
        model=model,
        lr_dem=lr_dem,
        tile_size=args.tile_size,
        overlap=args.overlap,
        scale=scale,
        device=args.device,
        norm_mode=args.norm_mode,
        use_tta=args.tta,
        tta_rotations=tta_rotations,
        use_uncertainty=args.uncertainty,
        n_uncertainty_samples=args.n_samples,
    )
    
    print(f"\nOutput shape: {sr_dem.shape}")
    print(f"Output range: [{sr_dem.min():.2f}, {sr_dem.max():.2f}]")
    
    # Save output
    print(f"\nSaving output: {args.output}")
    save_dem(sr_dem, args.output, metadata["crs"], metadata["bounds"], nodata)
    
    # Save uncertainty map if requested
    if uncertainty_map is not None and args.uncertainty_output:
        print(f"Saving uncertainty: {args.uncertainty_output}")
        save_dem(uncertainty_map, args.uncertainty_output, metadata["crs"], metadata["bounds"])
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
