"""
Spectral (Frequency Domain) Loss for DEM Super-Resolution

Preserves terrain texture and patterns by comparing frequency content.
Particularly useful for preserving small-scale details (bumps, roughness)
that are important for vehicle suspension stress calculation.

Key Features:
- FFT-based frequency comparison
- Separate weights for low and high frequencies
- High-frequency preservation for small obstacle detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class SpectralLoss(nn.Module):
    """
    Spectral Loss using FFT.
    
    Compares frequency content of predicted and target DEMs.
    Can weight low and high frequencies differently.
    
    Args:
        weight_low: Weight for low frequencies (overall shape) (default: 1.0)
        weight_high: Weight for high frequencies (fine detail) (default: 0.5)
        cutoff: Frequency cutoff ratio (0-0.5, default: 0.25)
        reduction: Loss reduction method (default: "mean")
    
    Benefits for DEM SR:
    - Preserves terrain texture
    - Reduces over-smoothing
    - Captures periodic patterns (ridges, valleys)
    - Preserves small-scale bumps (critical for vehicle stress)
    """
    def __init__(
        self,
        weight_low: float = 1.0,
        weight_high: float = 0.5,
        cutoff: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.cutoff = cutoff
        self.reduction = reduction
        
        # Cache for frequency mask
        self._mask_cache = {}
    
    def _create_frequency_mask(
        self,
        h: int,
        w: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create low-pass and high-pass frequency masks."""
        cache_key = (h, w, str(device))
        
        if cache_key not in self._mask_cache:
            # Create coordinate grid centered at DC component
            cy, cx = h // 2, w // 2
            y = torch.arange(h, device=device).float() - cy
            x = torch.arange(w, device=device).float() - cx
            yy, xx = torch.meshgrid(y, x, indexing='ij')
            
            # Normalized distance from center
            dist = torch.sqrt(yy**2 + xx**2) / min(h, w)
            
            # Low-pass mask (smooth transition)
            mask_low = torch.sigmoid(10 * (self.cutoff - dist))
            mask_high = 1 - mask_low
            
            # Add batch and channel dimensions
            mask_low = mask_low.unsqueeze(0).unsqueeze(0)
            mask_high = mask_high.unsqueeze(0).unsqueeze(0)
            
            self._mask_cache[cache_key] = (mask_low, mask_high)
        
        return self._mask_cache[cache_key]
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute spectral loss.
        
        Args:
            pred: Predicted DEM [B, 1, H, W]
            target: Ground truth DEM [B, 1, H, W]
            mask: Optional validity mask (not used for spectral)
            
        Returns:
            Spectral loss value
        """
        # Apply 2D FFT
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Shift DC component to center
        pred_fft = torch.fft.fftshift(pred_fft)
        target_fft = torch.fft.fftshift(target_fft)
        
        # Get magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        
        # Create frequency masks
        h, w = pred.shape[-2:]
        mask_low, mask_high = self._create_frequency_mask(h, w, pred.device)
        
        # Compute losses for low and high frequencies
        loss_low = F.l1_loss(
            pred_mag * mask_low,
            target_mag * mask_low,
            reduction=self.reduction,
        )
        loss_high = F.l1_loss(
            pred_mag * mask_high,
            target_mag * mask_high,
            reduction=self.reduction,
        )
        
        # Weighted combination
        total_loss = self.weight_low * loss_low + self.weight_high * loss_high
        
        return total_loss


class PhaseSpectralLoss(nn.Module):
    """
    Spectral Loss including phase information.
    
    Considers both magnitude and phase of the frequency content.
    Phase is important for spatial structure.
    
    Args:
        weight_magnitude: Weight for magnitude loss (default: 1.0)
        weight_phase: Weight for phase loss (default: 0.1)
    """
    def __init__(
        self,
        weight_magnitude: float = 1.0,
        weight_phase: float = 0.1,
        reduction: str = "mean",
    ):
        super().__init__()
        
        self.weight_magnitude = weight_magnitude
        self.weight_phase = weight_phase
        self.reduction = reduction
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute spectral loss with phase."""
        # Apply 2D FFT
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)
        
        # Magnitude loss
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        loss_mag = F.l1_loss(pred_mag, target_mag, reduction=self.reduction)
        
        # Phase loss (wrapped angle difference)
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        phase_diff = torch.abs(torch.atan2(
            torch.sin(pred_phase - target_phase),
            torch.cos(pred_phase - target_phase)
        ))
        loss_phase = phase_diff.mean() if self.reduction == "mean" else phase_diff.sum()
        
        return self.weight_magnitude * loss_mag + self.weight_phase * loss_phase


# Quick test
if __name__ == "__main__":
    print("Testing Spectral Loss...")
    print("=" * 50)
    
    # Test data
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    
    # Test SpectralLoss
    spectral_loss = SpectralLoss(weight_low=1.0, weight_high=0.5)
    loss = spectral_loss(pred, target)
    print(f"SpectralLoss: {loss.item():.4f}")
    
    # Test PhaseSpectralLoss
    phase_spectral_loss = PhaseSpectralLoss()
    loss = phase_spectral_loss(pred, target)
    print(f"PhaseSpectralLoss: {loss.item():.4f}")
    
    print("=" * 50)
    print("Spectral loss tests passed!")
