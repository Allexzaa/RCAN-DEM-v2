#!/usr/bin/env python3
"""
DEM Super-Resolution Training Script v2

Enhanced training with:
- Early stopping
- Learning rate warmup
- Gradient accumulation
- AdamW optimizer
- New loss functions
- Extended augmentation

Usage:
    python scripts/train_v2.py --data_dir processed/ --output_dir outputs_v2/
    
    # With config file
    python scripts/train_v2.py --config configs/default_v2.yaml --data_dir processed/
    
    # Resume training
    python scripts/train_v2.py --data_dir processed/ --resume outputs_v2/checkpoints/latest.pth

MPS Constraints:
    - NO AMP (gradient stability issues)
    - NO BatchNorm (unstable with small batches)
    - Use float32 precision throughout
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model_v2
from losses import create_loss_v2
from data import DEMDataset_v2, create_augmentation_pipeline


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from YAML file or use defaults."""
    default_config = {
        "model": {
            "type": "rcan_v2",
            "scale": 3,
            "n_resgroups": 10,
            "n_resblocks": 20,
            "n_feats": 64,
            "dropout": 0.1,
            "use_spatial_attention": False,
            "use_multiscale": False,
        },
        "loss": {
            "mode": "fixed",
            "w_elevation": 1.0,
            "w_gradient": 0.5,
            "w_curvature": 0.2,
            "w_spectral": 0.1,
            "w_edge": 0.2,
            "use_spectral": True,
            "use_edge": True,
            "use_complexity": True,
        },
        "training": {
            "epochs": 200,
            "batch_size": 4,
            "accumulation_steps": 4,
            "optimizer": "adamw",
            "lr": 1e-4,
            "lr_min": 1e-6,
            "weight_decay": 0.01,
            "warmup_epochs": 5,
            "early_stopping": True,
            "patience": 30,
            "min_delta": 1e-4,
            "clip_grad_norm": 1.0,
            "save_every": 10,
            "log_every": 10,
        },
        "augmentation": {
            "mode": "full",
            "noise_std": 0.02,
            "shift_range": 0.1,
            "scale_range": [0.9, 1.1],
        },
        "dataloader": {
            "num_workers": 4,
            "pin_memory": False,
            "persistent_workers": True,
        },
        "device": "mps",
        "seed": 42,
    }
    
    if config_path and Path(config_path).exists():
        try:
            import yaml
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
            # Deep merge
            for key, value in file_config.items():
                if isinstance(value, dict) and key in default_config:
                    default_config[key].update(value)
                else:
                    default_config[key] = value
        except ImportError:
            print("Warning: PyYAML not installed, using default config")
    
    return default_config


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get torch device with validation."""
    if device_str == "mps":
        if not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")
    elif device_str == "cuda":
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def setup_logging(output_dir: Path) -> logging.Logger:
    """Set up logging to file and console."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_v2_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    return logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping handler."""
    def __init__(self, patience: int = 30, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Improved
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return True  # No improvement


def create_warmup_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    lr_min: float,
):
    """Create learning rate scheduler with warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return lr_min / optimizer.defaults["lr"] + (1 - lr_min / optimizer.defaults["lr"]) * (
                0.5 * (1 + np.cos(np.pi * progress))
            )
    
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict,
    logger: logging.Logger,
    accumulation_steps: int = 1,
) -> Dict[str, float]:
    """Train for one epoch with gradient accumulation."""
    model.train()
    
    # Meters for tracking losses
    loss_meter = AverageMeter()
    elev_meter = AverageMeter()
    grad_meter = AverageMeter()
    curv_meter = AverageMeter()
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Get data
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        mask = batch.get("hr_mask")
        if mask is not None:
            mask = mask.to(device)
        
        # Forward pass
        sr = model(lr)
        
        # Compute loss
        loss, components = criterion(sr, hr, mask)
        
        # Scale loss for accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            if config["training"].get("clip_grad_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config["training"]["clip_grad_norm"],
                )
            
            optimizer.step()
            optimizer.zero_grad()
        
        # Update meters (use unscaled loss)
        batch_size = lr.size(0)
        loss_meter.update(loss.item() * accumulation_steps, batch_size)
        
        if "elevation" in components:
            elev_meter.update(components["elevation"].item(), batch_size)
        if "gradient" in components:
            grad_meter.update(components["gradient"].item(), batch_size)
        if "curvature" in components:
            curv_meter.update(components["curvature"].item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_meter.avg:.4f}",
            "elev": f"{elev_meter.avg:.4f}",
        })
    
    # Final gradient update if needed
    if len(dataloader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return {
        "loss": loss_meter.avg,
        "elevation": elev_meter.avg,
        "gradient": grad_meter.avg,
        "curvature": curv_meter.avg,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model on validation set."""
    model.eval()
    
    loss_meter = AverageMeter()
    elev_meter = AverageMeter()
    grad_meter = AverageMeter()
    curv_meter = AverageMeter()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating", leave=False):
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            mask = batch.get("hr_mask")
            if mask is not None:
                mask = mask.to(device)
            
            sr = model(lr)
            loss, components = criterion(sr, hr, mask)
            
            batch_size = lr.size(0)
            loss_meter.update(loss.item(), batch_size)
            
            if "elevation" in components:
                elev_meter.update(components["elevation"].item(), batch_size)
            if "gradient" in components:
                grad_meter.update(components["gradient"].item(), batch_size)
            if "curvature" in components:
                curv_meter.update(components["curvature"].item(), batch_size)
    
    return {
        "loss": loss_meter.avg,
        "elevation": elev_meter.avg,
        "gradient": grad_meter.avg,
        "curvature": curv_meter.avg,
    }


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_loss: float,
    config: Dict,
    output_dir: Path,
    is_best: bool = False,
):
    """Save training checkpoint."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_loss": best_loss,
        "config": config,
    }
    
    # Save regular checkpoint
    path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
    torch.save(checkpoint, path)
    
    # Save best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pth"
        torch.save(checkpoint, best_path)
    
    # Save latest
    latest_path = checkpoint_dir / "latest.pth"
    torch.save(checkpoint, latest_path)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(
    data_dirs: List[str],
    output_dir: str,
    config: Dict,
    resume: Optional[str] = None,
):
    """Main training function with all v2 improvements."""
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_path)
    logger.info("=" * 60)
    logger.info("DEM Super-Resolution Training v2")
    logger.info("=" * 60)
    
    # Set seed
    set_seed(config.get("seed", 42))
    logger.info(f"Random seed: {config.get('seed', 42)}")
    
    # Setup device
    device = get_device(config.get("device", "mps"))
    logger.info(f"Device: {device}")
    
    # MPS optimization
    if device.type == "mps":
        torch.set_float32_matmul_precision("high")
        logger.info("MPS: Set float32 matmul precision to 'high'")
    
    # Create datasets
    logger.info(f"Loading data from {len(data_dirs)} directories...")
    
    train_datasets = []
    val_datasets = []
    
    aug_config = config.get("augmentation", {})
    
    for data_dir in data_dirs:
        train_ds = DEMDataset_v2(
            data_dir=data_dir,
            split="train",
            augment=True,
            augment_mode=aug_config.get("mode", "full"),
            return_mask=True,
            noise_std=aug_config.get("noise_std", 0.02),
            shift_range=aug_config.get("shift_range", 0.1),
        )
        val_ds = DEMDataset_v2(
            data_dir=data_dir,
            split="val",
            augment=False,
            return_mask=True,
        )
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
        logger.info(f"  {data_dir}: {len(train_ds)} train, {len(val_ds)} val")
    
    # Combine datasets
    train_dataset = ConcatDataset(train_datasets)
    val_dataset = ConcatDataset(val_datasets)
    
    logger.info(f"Total: {len(train_dataset)} train, {len(val_dataset)} val")
    
    # Create dataloaders
    dl_config = config.get("dataloader", {})
    train_config = config.get("training", {})
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.get("batch_size", 4),
        shuffle=True,
        num_workers=dl_config.get("num_workers", 4),
        pin_memory=dl_config.get("pin_memory", False),
        persistent_workers=dl_config.get("persistent_workers", True),
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config.get("batch_size", 4),
        shuffle=False,
        num_workers=dl_config.get("num_workers", 4),
        pin_memory=dl_config.get("pin_memory", False),
        persistent_workers=dl_config.get("persistent_workers", True),
    )
    
    # Create model
    model_config = config.get("model", {})
    logger.info(f"Creating model: {model_config.get('type', 'rcan_v2')}")
    
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
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    
    # Create loss function
    loss_config = config.get("loss", {})
    criterion = create_loss_v2(
        "physics_v2",
        mode=loss_config.get("mode", "fixed"),
        w_elevation=loss_config.get("w_elevation", 1.0),
        w_gradient=loss_config.get("w_gradient", 0.5),
        w_curvature=loss_config.get("w_curvature", 0.2),
        w_spectral=loss_config.get("w_spectral", 0.1),
        w_edge=loss_config.get("w_edge", 0.2),
        use_spectral=loss_config.get("use_spectral", True),
        use_edge=loss_config.get("use_edge", True),
        use_complexity=loss_config.get("use_complexity", True),
    ).to(device)
    
    logger.info(f"Loss weights: {criterion.get_weights()}")
    
    # Create optimizer
    if train_config.get("optimizer", "adamw") == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=train_config.get("lr", 1e-4),
            weight_decay=train_config.get("weight_decay", 0.01),
        )
    else:
        optimizer = Adam(
            model.parameters(),
            lr=train_config.get("lr", 1e-4),
        )
    
    logger.info(f"Optimizer: {train_config.get('optimizer', 'adamw')}")
    
    # Create scheduler with warmup
    warmup_epochs = train_config.get("warmup_epochs", 5)
    total_epochs = train_config.get("epochs", 200)
    
    if warmup_epochs > 0:
        scheduler = create_warmup_scheduler(
            optimizer,
            warmup_epochs,
            total_epochs,
            train_config.get("lr_min", 1e-6),
        )
        logger.info(f"Scheduler: Warmup ({warmup_epochs} epochs) + Cosine Annealing")
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=train_config.get("lr_min", 1e-6),
        )
        logger.info("Scheduler: Cosine Annealing")
    
    # Early stopping
    if train_config.get("early_stopping", True):
        early_stopping = EarlyStopping(
            patience=train_config.get("patience", 30),
            min_delta=train_config.get("min_delta", 1e-4),
        )
        logger.info(f"Early stopping: patience={train_config.get('patience', 30)}")
    else:
        early_stopping = None
    
    # Resume from checkpoint
    start_epoch = 1
    best_loss = float("inf")
    
    if resume:
        logger.info(f"Resuming from: {resume}")
        checkpoint = torch.load(resume, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_loss = checkpoint.get("best_loss", float("inf"))
        logger.info(f"Resumed at epoch {start_epoch}, best_loss={best_loss:.4f}")
    
    # Save config
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # TensorBoard writer
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = output_path / "tensorboard"
        writer = SummaryWriter(log_dir=str(tb_dir))
        logger.info(f"TensorBoard logs: {tb_dir}")
    except ImportError:
        writer = None
        logger.info("TensorBoard not available")
    
    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_elevation": [], "val_elevation": [],
        "train_gradient": [], "val_gradient": [],
        "train_curvature": [], "val_curvature": [],
        "lr": [],
    }
    
    # Gradient accumulation steps
    accumulation_steps = train_config.get("accumulation_steps", 1)
    logger.info(f"Gradient accumulation: {accumulation_steps} steps")
    logger.info(f"Effective batch size: {train_config.get('batch_size', 4) * accumulation_steps}")
    
    # Training loop
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, config, logger, accumulation_steps
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Log epoch results
        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch {epoch}/{total_epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Update history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_elevation"].append(train_metrics["elevation"])
        history["val_elevation"].append(val_metrics["elevation"])
        history["train_gradient"].append(train_metrics["gradient"])
        history["val_gradient"].append(val_metrics["gradient"])
        history["train_curvature"].append(train_metrics["curvature"])
        history["val_curvature"].append(val_metrics["curvature"])
        history["lr"].append(current_lr)
        
        # TensorBoard logging
        if writer:
            writer.add_scalars("Loss", {
                "train": train_metrics["loss"],
                "val": val_metrics["loss"],
            }, epoch)
            writer.add_scalar("Learning_Rate", current_lr, epoch)
            writer.flush()
        
        # Check for best model
        is_best = val_metrics["loss"] < best_loss
        if is_best:
            best_loss = val_metrics["loss"]
            logger.info(f"  ✓ New best model! Val Loss: {best_loss:.4f}")
        
        # Early stopping
        if early_stopping:
            early_stopping(val_metrics["loss"])
            if early_stopping.should_stop:
                logger.info(f"  ✗ Early stopping triggered (no improvement for {early_stopping.patience} epochs)")
                break
        
        # Save checkpoint
        if epoch % train_config.get("save_every", 10) == 0 or is_best:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                best_loss, config, output_path, is_best
            )
        
        # Save history
        history_path = output_path / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
    
    # Final save
    save_checkpoint(
        model, optimizer, scheduler, epoch,
        best_loss, config, output_path, is_best=False
    )
    
    # Close TensorBoard writer
    if writer:
        writer.close()
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_path / 'checkpoints'}")
    logger.info("=" * 60)
    
    return history


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train DEM Super-Resolution Model v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Preprocessed data directory (comma-separated for multiple)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs_v2",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to use"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override device from CLI
    if args.device:
        config["device"] = args.device
    
    # Parse data directories
    data_dirs = [d.strip() for d in args.data_dir.split(",")]
    
    # Run training
    train(
        data_dirs=data_dirs,
        output_dir=args.output_dir,
        config=config,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
