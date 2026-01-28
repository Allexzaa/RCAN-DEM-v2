#!/usr/bin/env python3
"""
Phase 7: Inference & Deployment v2 for DEM Super-Resolution

Enhanced inference with:
- Test-Time Augmentation (TTA)
- Uncertainty estimation (Monte Carlo Dropout)
- Improved sliding window blending
- AAIGrid output format for FracAdapt backend compatibility

Usage:
    # Basic inference (outputs AAIGrid format by default)
    python inference_v2.py --checkpoint outputs_v2/checkpoints/best_model.pth --input dem_30m.tif --output dem_10m_sr.asc
    
    # With TTA
    python inference_v2.py --checkpoint best_model.pth --input dem_30m.tif --output dem_10m.asc --tta
    
    # With uncertainty estimation
    python inference_v2.py --checkpoint best_model.pth --input dem_30m.tif --output dem_10m.asc --uncertainty
    
    # Output GeoTIFF format instead
    python inference_v2.py --checkpoint best_model.pth --input dem_30m.tif --output dem_10m.tif --format geotiff
    
    # Output both formats
    python inference_v2.py --checkpoint best_model.pth --input dem_30m.tif --output dem_10m.asc --also_geotiff dem_10m.tif
"""

import argparse
import sys
from collections import namedtuple
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
    """
    Load DEM from GeoTIFF or AAIGrid file.
    
    Automatically detects format based on file extension.
    
    Args:
        input_path: Path to input DEM (.tif, .tiff, .asc, or .txt)
        
    Returns:
        Tuple of (dem_array, metadata_dict)
    """
    input_path = Path(input_path)
    ext = input_path.suffix.lower()
    
    if ext in ['.asc', '.txt']:
        # Load AAIGrid format
        return load_aaigrid(str(input_path))
    else:
        # Load GeoTIFF format (default)
        with rasterio.open(str(input_path)) as src:
            dem = src.read(1)
            metadata = {
                "crs": src.crs,
                "transform": src.transform,
                "bounds": src.bounds,
                "nodata": src.nodata,
                "width": src.width,
                "height": src.height,
                "format": "geotiff",
            }
        return dem, metadata


def load_aaigrid(input_path: str) -> Tuple[np.ndarray, dict]:
    """
    Load DEM from AAIGrid (Arc ASCII Grid) format.
    
    This is the format used by OpenTopography API and FracAdapt backend.
    
    Args:
        input_path: Path to AAIGrid file (.asc or .txt)
        
    Returns:
        Tuple of (dem_array, metadata_dict)
    """
    header = {}
    
    with open(input_path, 'r') as f:
        # Read header (6 lines)
        for _ in range(6):
            line = f.readline().strip()
            key, value = line.split()
            key = key.lower()
            if key in ['ncols', 'nrows']:
                header[key] = int(value)
            else:
                header[key] = float(value)
        
        # Read grid data
        grid_data = []
        for line in f:
            row = [float(v) for v in line.strip().split()]
            if row:  # Skip empty lines
                grid_data.append(row)
    
    dem = np.array(grid_data, dtype=np.float32)
    
    # Replace nodata with NaN
    nodata = header.get('nodata_value', -9999)
    dem = np.where(dem == nodata, np.nan, dem)
    
    # Calculate bounds from header
    xllcorner = header['xllcorner']
    yllcorner = header['yllcorner']
    cellsize = header['cellsize']
    ncols = header['ncols']
    nrows = header['nrows']
    
    # Create bounds object compatible with rasterio
    Bounds = namedtuple('Bounds', ['left', 'bottom', 'right', 'top'])
    bounds = Bounds(
        left=xllcorner,
        bottom=yllcorner,
        right=xllcorner + (ncols * cellsize),
        top=yllcorner + (nrows * cellsize)
    )
    
    metadata = {
        "crs": "EPSG:4326",  # Assume WGS84 for geographic coordinates
        "transform": None,
        "bounds": bounds,
        "nodata": nodata,
        "width": ncols,
        "height": nrows,
        "cellsize": cellsize,
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
        "format": "aaigrid",
    }
    
    return dem, metadata


def save_dem(
    dem: np.ndarray,
    output_path: str,
    crs,
    bounds,
    nodata: Optional[float] = None,
    output_format: str = "aaigrid",
    precision: int = 2,
):
    """
    Save DEM to file (AAIGrid or GeoTIFF format).
    
    Args:
        dem: 2D numpy array of elevation values
        output_path: Output file path
        crs: Coordinate reference system
        bounds: Geographic bounds (left, bottom, right, top)
        nodata: NoData sentinel value
        output_format: "aaigrid" (default) or "geotiff"
        precision: Decimal precision for AAIGrid values (default: 2)
    """
    if output_format == "aaigrid":
        save_aaigrid(dem, output_path, bounds, nodata, precision)
    else:
        save_geotiff(dem, output_path, crs, bounds, nodata)


def save_geotiff(
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
    
    print(f"Saved GeoTIFF: {output_path}")
    print(f"  Dimensions: {width} x {height}")
    print(f"  CRS: {crs}")


def save_aaigrid(
    dem: np.ndarray,
    output_path: str,
    bounds,
    nodata: Optional[float] = -9999,
    precision: int = 2,
):
    """
    Save DEM in AAIGrid (Arc ASCII Grid) format.
    
    This format is directly compatible with the FracAdapt backend parser.
    The backend expects this exact format from OpenTopography API.
    
    AAIGrid Format:
        ncols         <number of columns>
        nrows         <number of rows>
        xllcorner     <west boundary longitude>
        yllcorner     <south boundary latitude>
        cellsize      <cell size in degrees>
        nodata_value  <missing data sentinel>
        <space-separated elevation values, row by row>
    
    Args:
        dem: 2D numpy array of elevation values [H, W]
        output_path: Output .asc file path
        bounds: Geographic bounds (left, bottom, right, top)
        nodata: NoData sentinel value (default: -9999)
        precision: Decimal places for elevation values (default: 2)
    """
    height, width = dem.shape
    
    # Calculate cellsize from bounds and dimensions
    # cellsize = (east - west) / ncols
    cellsize_x = (bounds.right - bounds.left) / width
    cellsize_y = (bounds.top - bounds.bottom) / height
    
    # AAIGrid assumes square cells - use average if slightly different
    cellsize = (cellsize_x + cellsize_y) / 2
    
    # xllcorner and yllcorner are the LOWER-LEFT corner coordinates
    xllcorner = bounds.left
    yllcorner = bounds.bottom
    
    # Replace NaN with nodata value
    if nodata is None:
        nodata = -9999
    dem_clean = np.where(np.isnan(dem), nodata, dem)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write header (exactly matches OpenTopography format)
        f.write(f"ncols         {width}\n")
        f.write(f"nrows         {height}\n")
        f.write(f"xllcorner     {xllcorner:.10f}\n")
        f.write(f"yllcorner     {yllcorner:.10f}\n")
        f.write(f"cellsize      {cellsize:.15f}\n")
        f.write(f"nodata_value  {nodata}\n")
        
        # Write grid data (row by row, top to bottom)
        # AAIGrid stores data from TOP-LEFT corner
        for row in dem_clean:
            row_str = " ".join(f"{v:.{precision}f}" for v in row)
            f.write(row_str + "\n")
    
    # Calculate approximate resolution in meters
    resolution_m = cellsize * 111320  # degrees to meters at equator
    
    print(f"Saved AAIGrid: {output_path}")
    print(f"  Dimensions: {width} x {height}")
    print(f"  Cellsize: {cellsize:.10f} deg (~{resolution_m:.1f}m)")
    print(f"  Bounds: [{xllcorner:.6f}, {yllcorner:.6f}] to [{bounds.right:.6f}, {bounds.top:.6f}]")


def main():
    parser = argparse.ArgumentParser(
        description="DEM Super-Resolution Inference v2 with FracAdapt Backend Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic inference (AAIGrid output for FracAdapt backend)
    python inference_v2.py --checkpoint best_model.pth --input dem_30m.tif --output dem_10m.asc
    
    # With TTA for better quality
    python inference_v2.py --checkpoint best_model.pth --input dem_30m.asc --output dem_10m.asc --tta
    
    # Output GeoTIFF instead
    python inference_v2.py --checkpoint best_model.pth --input dem_30m.tif --output dem_10m.tif --format geotiff
    
    # Output both formats
    python inference_v2.py --checkpoint best_model.pth --input dem_30m.tif --output dem_10m.asc --also_geotiff dem_10m.tif
        """
    )
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--input", type=str, required=True, help="Input LR DEM (.tif or .asc)")
    parser.add_argument("--output", type=str, required=True, help="Output SR DEM path")
    parser.add_argument("--tile_size", type=int, default=256, help="Tile size for processing (default: 256)")
    parser.add_argument("--overlap", type=int, default=32, help="Tile overlap in pixels (default: 32)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "mps", "cuda"], help="Inference device")
    parser.add_argument("--norm_mode", type=str, default="global", choices=["global", "per_tile"], help="Normalization mode")
    
    # Output format options
    parser.add_argument("--format", type=str, default="aaigrid", choices=["aaigrid", "geotiff"],
                        help="Output format: 'aaigrid' (default, for FracAdapt backend) or 'geotiff'")
    parser.add_argument("--also_geotiff", type=str, default=None,
                        help="Also save GeoTIFF to this path (in addition to primary output)")
    parser.add_argument("--precision", type=int, default=2,
                        help="Decimal precision for AAIGrid elevation values (default: 2)")
    
    # TTA options
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--tta_rotations", type=str, default="0,90,180,270", help="TTA rotations (comma-separated)")
    
    # Uncertainty options
    parser.add_argument("--uncertainty", action="store_true", help="Enable uncertainty estimation (MC Dropout)")
    parser.add_argument("--n_samples", type=int, default=10, help="MC dropout samples (default: 10)")
    parser.add_argument("--uncertainty_output", type=str, default=None, help="Uncertainty map output path")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("DEM Super-Resolution Inference v2")
    print("FracAdapt Backend Compatible")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Format: {args.format.upper()}")
    print(f"TTA: {args.tta}")
    print(f"Uncertainty: {args.uncertainty}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model, config = load_model(args.checkpoint, args.device)
    scale = config.get("model", {}).get("scale", 3)
    print(f"Model loaded. Scale: {scale}x")
    
    # Load input (supports both GeoTIFF and AAIGrid)
    print(f"\nLoading input: {args.input}")
    lr_dem, metadata = load_dem(args.input)
    input_format = metadata.get("format", "unknown")
    print(f"Input shape: {lr_dem.shape}")
    print(f"Input format: {input_format}")
    
    # Handle nodata
    nodata = metadata.get("nodata")
    if nodata is not None:
        nodata_mask = np.isnan(lr_dem) if np.isnan(nodata) else (lr_dem == nodata)
        if nodata_mask.any():
            print(f"NoData pixels: {nodata_mask.sum()}")
            lr_dem = np.where(nodata_mask, np.nan, lr_dem)
            lr_dem = np.nan_to_num(lr_dem, nan=np.nanmean(lr_dem))
    
    # Handle NaN values that might come from AAIGrid loading
    if np.isnan(lr_dem).any():
        nan_count = np.isnan(lr_dem).sum()
        print(f"Replacing {nan_count} NaN values with mean elevation")
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
    print(f"Output range: [{sr_dem.min():.2f}, {sr_dem.max():.2f}] meters")
    print(f"Resolution improvement: {lr_dem.shape} -> {sr_dem.shape} ({scale}x)")
    
    # Calculate output resolution
    input_cellsize = metadata.get("cellsize")
    if input_cellsize:
        output_cellsize = input_cellsize / scale
        input_res_m = input_cellsize * 111320
        output_res_m = output_cellsize * 111320
        print(f"Resolution: ~{input_res_m:.0f}m -> ~{output_res_m:.0f}m")
    
    # Save primary output
    print(f"\nSaving primary output: {args.output}")
    save_dem(
        sr_dem, args.output, metadata["crs"], metadata["bounds"],
        nodata=-9999 if nodata is None else nodata,
        output_format=args.format,
        precision=args.precision
    )
    
    # Save additional GeoTIFF if requested
    if args.also_geotiff:
        print(f"\nSaving additional GeoTIFF: {args.also_geotiff}")
        save_dem(
            sr_dem, args.also_geotiff, metadata["crs"], metadata["bounds"],
            nodata=-9999 if nodata is None else nodata,
            output_format="geotiff"
        )
    
    # Save uncertainty map if requested
    if uncertainty_map is not None and args.uncertainty_output:
        print(f"\nSaving uncertainty map: {args.uncertainty_output}")
        save_dem(
            uncertainty_map, args.uncertainty_output, metadata["crs"], metadata["bounds"],
            nodata=-9999,
            output_format=args.format,
            precision=4  # Higher precision for uncertainty values
        )
    
    print("\n" + "=" * 60)
    print("Inference complete!")
    print(f"Output ready for FracAdapt backend: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
