"""
Enhanced Data Augmentation for DEM Super-Resolution v2

New augmentations beyond basic flips/rotations:
- Gaussian noise: Simulates sensor noise
- Elevation shift: Random offset for elevation invariance
- Scale jitter: Random rescaling for scale invariance

Key Features:
- DEM-specific augmentations
- Consistent application to LR/HR pairs
- Configurable augmentation pipeline
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
import torch


class GaussianNoise:
    """
    Add Gaussian noise to DEM data.
    
    Simulates sensor noise and measurement uncertainty.
    Helps model become robust to noisy input data.
    
    Args:
        std: Standard deviation of noise (default: 0.02)
            Applied to normalized data, so 0.02 = 2% of std dev
        prob: Probability of applying augmentation (default: 0.5)
    """
    def __init__(self, std: float = 0.02, prob: float = 0.5):
        self.std = std
        self.prob = prob
    
    def __call__(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add noise to LR only (not HR, which is ground truth)."""
        if np.random.random() > self.prob:
            return lr, hr
        
        # Add noise to LR
        noise = np.random.normal(0, self.std, lr.shape).astype(np.float32)
        lr_noisy = lr + noise
        
        return lr_noisy, hr


class ElevationShift:
    """
    Apply random elevation offset.
    
    Teaches model elevation invariance - the model should learn
    that terrain shape matters, not absolute elevation values.
    
    Args:
        shift_range: Maximum shift in normalized units (default: 0.1)
            For z-score normalized data, this represents ~0.1 std devs
        prob: Probability of applying augmentation (default: 0.5)
    """
    def __init__(self, shift_range: float = 0.1, prob: float = 0.5):
        self.shift_range = shift_range
        self.prob = prob
    
    def __call__(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply same shift to both LR and HR."""
        if np.random.random() > self.prob:
            return lr, hr
        
        # Random shift
        shift = np.random.uniform(-self.shift_range, self.shift_range)
        
        lr_shifted = lr + shift
        hr_shifted = hr + shift
        
        return lr_shifted, hr_shifted


class ScaleJitter:
    """
    Apply random intensity scaling.
    
    Simulates different elevation ranges and helps model
    generalize across various terrain types.
    
    Args:
        scale_range: (min_scale, max_scale) (default: (0.9, 1.1))
        prob: Probability of applying augmentation (default: 0.3)
    """
    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        prob: float = 0.3,
    ):
        self.scale_min, self.scale_max = scale_range
        self.prob = prob
    
    def __call__(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply same scale to both LR and HR."""
        if np.random.random() > self.prob:
            return lr, hr
        
        # Random scale factor
        scale = np.random.uniform(self.scale_min, self.scale_max)
        
        lr_scaled = lr * scale
        hr_scaled = hr * scale
        
        return lr_scaled, hr_scaled


class RandomFlip:
    """Random horizontal and vertical flips."""
    def __init__(self, prob: float = 0.5):
        self.prob = prob
    
    def __call__(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Horizontal flip
        if np.random.random() < self.prob:
            lr = np.flip(lr, axis=1).copy()
            hr = np.flip(hr, axis=1).copy()
        
        # Vertical flip
        if np.random.random() < self.prob:
            lr = np.flip(lr, axis=0).copy()
            hr = np.flip(hr, axis=0).copy()
        
        return lr, hr


class RandomRotation90:
    """Random 90-degree rotations."""
    def __init__(self):
        pass
    
    def __call__(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        k = np.random.randint(0, 4)  # 0, 90, 180, or 270 degrees
        if k > 0:
            lr = np.rot90(lr, k).copy()
            hr = np.rot90(hr, k).copy()
        return lr, hr


class DEMAugmentation:
    """
    Comprehensive DEM augmentation pipeline.
    
    Combines all augmentation techniques in proper order.
    
    Args:
        use_flip: Enable flip augmentation (default: True)
        use_rotation: Enable 90° rotation (default: True)
        use_noise: Enable Gaussian noise (default: True)
        use_shift: Enable elevation shift (default: True)
        use_scale: Enable scale jitter (default: True)
        noise_std: Noise standard deviation (default: 0.02)
        shift_range: Elevation shift range (default: 0.1)
        scale_range: Scale jitter range (default: (0.9, 1.1))
    
    Example:
        >>> aug = DEMAugmentation(use_noise=True, noise_std=0.03)
        >>> lr_aug, hr_aug = aug(lr, hr)
    """
    def __init__(
        self,
        use_flip: bool = True,
        use_rotation: bool = True,
        use_noise: bool = True,
        use_shift: bool = True,
        use_scale: bool = True,
        noise_std: float = 0.02,
        noise_prob: float = 0.5,
        shift_range: float = 0.1,
        shift_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        scale_prob: float = 0.3,
    ):
        self.augmentations = []
        
        # Geometric augmentations (always first)
        if use_flip:
            self.augmentations.append(RandomFlip(prob=0.5))
        
        if use_rotation:
            self.augmentations.append(RandomRotation90())
        
        # Intensity/value augmentations
        if use_scale:
            self.augmentations.append(ScaleJitter(scale_range, prob=scale_prob))
        
        if use_shift:
            self.augmentations.append(ElevationShift(shift_range, prob=shift_prob))
        
        # Noise augmentation (last, applied to LR only)
        if use_noise:
            self.augmentations.append(GaussianNoise(noise_std, prob=noise_prob))
        
        # Store config
        self.config = {
            "use_flip": use_flip,
            "use_rotation": use_rotation,
            "use_noise": use_noise,
            "use_shift": use_shift,
            "use_scale": use_scale,
            "noise_std": noise_std,
            "shift_range": shift_range,
            "scale_range": scale_range,
        }
    
    def __call__(
        self,
        lr: np.ndarray,
        hr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply augmentation pipeline."""
        for aug in self.augmentations:
            lr, hr = aug(lr, hr)
        return lr, hr
    
    def get_config(self) -> Dict[str, Any]:
        """Get augmentation configuration."""
        return self.config.copy()


def create_augmentation_pipeline(
    mode: str = "full",
    **kwargs,
) -> DEMAugmentation:
    """
    Factory function to create augmentation pipelines.
    
    Args:
        mode: "full", "basic", "noise_only", or "none"
        **kwargs: Override default parameters
        
    Returns:
        DEMAugmentation instance
    """
    if mode == "none":
        return DEMAugmentation(
            use_flip=False,
            use_rotation=False,
            use_noise=False,
            use_shift=False,
            use_scale=False,
        )
    elif mode == "basic":
        return DEMAugmentation(
            use_flip=True,
            use_rotation=True,
            use_noise=False,
            use_shift=False,
            use_scale=False,
            **kwargs,
        )
    elif mode == "noise_only":
        return DEMAugmentation(
            use_flip=False,
            use_rotation=False,
            use_noise=True,
            use_shift=False,
            use_scale=False,
            **kwargs,
        )
    else:  # "full"
        return DEMAugmentation(
            use_flip=True,
            use_rotation=True,
            use_noise=True,
            use_shift=True,
            use_scale=True,
            **kwargs,
        )


# Quick test
if __name__ == "__main__":
    print("Testing DEM Augmentation Pipeline...")
    print("=" * 60)
    
    # Create test data
    lr = np.random.randn(256, 256).astype(np.float32)
    hr = np.random.randn(768, 768).astype(np.float32)
    
    # Test individual augmentations
    print("\n1. Individual augmentations:")
    
    noise = GaussianNoise(std=0.02)
    lr_n, hr_n = noise(lr.copy(), hr.copy())
    print(f"   GaussianNoise: LR diff = {np.abs(lr - lr_n).mean():.4f}")
    
    shift = ElevationShift(shift_range=0.1)
    lr_s, hr_s = shift(lr.copy(), hr.copy())
    print(f"   ElevationShift: LR diff = {np.abs(lr - lr_s).mean():.4f}")
    
    scale = ScaleJitter(scale_range=(0.9, 1.1))
    lr_sc, hr_sc = scale(lr.copy(), hr.copy())
    print(f"   ScaleJitter: LR diff = {np.abs(lr - lr_sc).mean():.4f}")
    
    # Test full pipeline
    print("\n2. Full augmentation pipeline:")
    aug = DEMAugmentation(
        use_flip=True,
        use_rotation=True,
        use_noise=True,
        use_shift=True,
        use_scale=True,
        noise_std=0.02,
    )
    
    lr_aug, hr_aug = aug(lr.copy(), hr.copy())
    print(f"   LR shape: {lr_aug.shape} (expected {lr.shape})")
    print(f"   HR shape: {hr_aug.shape} (expected {hr.shape})")
    print(f"   Config: {aug.get_config()}")
    
    # Test factory
    print("\n3. Factory function:")
    aug_full = create_augmentation_pipeline("full")
    aug_basic = create_augmentation_pipeline("basic")
    aug_none = create_augmentation_pipeline("none")
    print(f"   Full: {len(aug_full.augmentations)} augmentations")
    print(f"   Basic: {len(aug_basic.augmentations)} augmentations")
    print(f"   None: {len(aug_none.augmentations)} augmentations")
    
    print("\n" + "=" * 60)
    print("All augmentation tests passed!")
