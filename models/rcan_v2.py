"""
RCAN-DEM v2: Improved Residual Channel Attention Network for DEM Super-Resolution

Architecture for 30m → 10m (3× upscaling) DEM enhancement.

v2 Key Features:
- Spatial dropout for regularization (reduces overfitting)
- Optional multi-scale convolutions (captures features at different scales)
- Optional spatial attention (learns where to focus)
- All v1 features preserved

Reference: Image Super-Resolution Using Very Deep Residual Channel Attention Networks
https://arxiv.org/abs/1807.02758

v2 Modifications:
- Dropout for better generalization
- Spatial attention option
- Multi-scale feature option
- Configurable per-group settings
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .layers_v2 import (
    ResidualGroup_v2,
    Upsampler,
    MultiScaleConv,
    SpatialAttention,
    SpatialDropout2d,
)


class RCAN_DEM_v2(nn.Module):
    """
    RCAN v2 for DEM Super-Resolution.
    
    Improved architecture with:
    - Dropout regularization
    - Optional spatial attention
    - Optional multi-scale feature extraction
    
    Args:
        scale: Upscaling factor (default: 3 for 30m→10m)
        n_resgroups: Number of residual groups (default: 10)
        n_resblocks: Number of RCABs per group (default: 20)
        n_feats: Number of feature channels (default: 64)
        reduction: Channel attention reduction ratio (default: 16)
        res_scale: Residual scaling factor (default: 0.1)
        n_channels: Number of input/output channels (default: 1 for DEM)
        dropout: Dropout rate (default: 0.1)
        use_spatial_attention: Use spatial attention in RCABs (default: False)
        use_multiscale: Use multi-scale conv before groups (default: False)
    
    Total parameters (default config with dropout): ~15.6M
    
    MPS Notes:
        - No BatchNorm (unstable with small batches)
        - No AMP (gradient issues on MPS)
        - Use float32 precision
    """
    def __init__(
        self,
        scale: int = 3,
        n_resgroups: int = 10,
        n_resblocks: int = 20,
        n_feats: int = 64,
        reduction: int = 16,
        res_scale: float = 0.1,
        n_channels: int = 1,
        dropout: float = 0.1,
        use_spatial_attention: bool = False,
        use_multiscale: bool = False,
    ):
        super().__init__()
        
        self.scale = scale
        self.n_feats = n_feats
        self.dropout_rate = dropout
        self.use_spatial_attention = use_spatial_attention
        self.use_multiscale = use_multiscale
        
        kernel_size = 3
        
        # Store config for checkpoint
        self.config = {
            "scale": scale,
            "n_resgroups": n_resgroups,
            "n_resblocks": n_resblocks,
            "n_feats": n_feats,
            "reduction": reduction,
            "res_scale": res_scale,
            "n_channels": n_channels,
            "dropout": dropout,
            "use_spatial_attention": use_spatial_attention,
            "use_multiscale": use_multiscale,
        }
        
        # Head: expand from 1 channel to n_feats
        self.head = nn.Conv2d(n_channels, n_feats, kernel_size, padding=kernel_size // 2)
        
        # Optional: Multi-scale feature extraction after head
        if use_multiscale:
            self.multiscale = MultiScaleConv(n_feats, dilations=[1, 2, 4])
        else:
            self.multiscale = None
        
        # Body: residual groups with channel attention
        body = [
            ResidualGroup_v2(
                n_feat=n_feats,
                kernel_size=kernel_size,
                reduction=reduction,
                n_resblocks=n_resblocks,
                dropout=dropout,
                res_scale=res_scale,
                use_spatial_attention=use_spatial_attention,
            )
            for _ in range(n_resgroups)
        ]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2))
        self.body = nn.Sequential(*body)
        
        # Optional: Final spatial attention before upsampling
        if use_spatial_attention:
            self.final_attention = SpatialAttention(kernel_size=7)
        else:
            self.final_attention = None
        
        # Tail: upsampler + final conv to 1 channel
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, n_channels, kernel_size, padding=kernel_size // 2),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input LR tensor [B, 1, H, W]
            
        Returns:
            SR tensor [B, 1, H*scale, W*scale]
        """
        # Head
        x = self.head(x)
        
        # Optional multi-scale feature extraction
        if self.multiscale is not None:
            x = x + self.multiscale(x)
        
        # Body with global residual
        res = self.body(x)
        
        # Optional final attention
        if self.final_attention is not None:
            res = self.final_attention(res)
        
        res = res + x
        
        # Tail (upsampling)
        x = self.tail(res)
        
        return x
    
    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration."""
        return self.config.copy()
    
    def __repr__(self) -> str:
        return (
            f"RCAN_DEM_v2(\n"
            f"  scale={self.scale},\n"
            f"  n_feats={self.n_feats},\n"
            f"  dropout={self.dropout_rate},\n"
            f"  spatial_attention={self.use_spatial_attention},\n"
            f"  multiscale={self.use_multiscale},\n"
            f"  params={self.get_num_params():,}\n"
            f")"
        )


class RCAN_DEM_v2_Light(nn.Module):
    """
    Lightweight RCAN v2 for faster training/inference.
    
    Uses fewer residual groups and blocks:
    - 5 groups × 10 blocks (vs 10 × 20)
    - 32 features (vs 64)
    
    Good for:
    - Initial experiments
    - Limited GPU memory
    - Faster iteration
    
    Total parameters: ~2M
    """
    def __init__(
        self,
        scale: int = 3,
        n_resgroups: int = 5,
        n_resblocks: int = 10,
        n_feats: int = 32,
        reduction: int = 8,
        res_scale: float = 0.1,
        n_channels: int = 1,
        dropout: float = 0.1,
        use_spatial_attention: bool = False,
        use_multiscale: bool = False,
    ):
        super().__init__()
        
        self.scale = scale
        self.n_feats = n_feats
        self.dropout_rate = dropout
        
        kernel_size = 3
        
        # Store config
        self.config = {
            "scale": scale,
            "n_resgroups": n_resgroups,
            "n_resblocks": n_resblocks,
            "n_feats": n_feats,
            "reduction": reduction,
            "res_scale": res_scale,
            "n_channels": n_channels,
            "dropout": dropout,
            "use_spatial_attention": use_spatial_attention,
            "use_multiscale": use_multiscale,
        }
        
        # Head
        self.head = nn.Conv2d(n_channels, n_feats, kernel_size, padding=kernel_size // 2)
        
        # Optional multi-scale
        if use_multiscale:
            self.multiscale = MultiScaleConv(n_feats, dilations=[1, 2], reduction=2)
        else:
            self.multiscale = None
        
        # Body
        body = [
            ResidualGroup_v2(
                n_feat=n_feats,
                kernel_size=kernel_size,
                reduction=reduction,
                n_resblocks=n_resblocks,
                dropout=dropout,
                res_scale=res_scale,
                use_spatial_attention=use_spatial_attention,
            )
            for _ in range(n_resgroups)
        ]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=kernel_size // 2))
        self.body = nn.Sequential(*body)
        
        # Tail
        self.tail = nn.Sequential(
            Upsampler(scale, n_feats),
            nn.Conv2d(n_feats, n_channels, kernel_size, padding=kernel_size // 2),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        
        if self.multiscale is not None:
            x = x + self.multiscale(x)
        
        res = self.body(x)
        res = res + x
        x = self.tail(res)
        return x
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self) -> Dict[str, Any]:
        return self.config.copy()


def create_model_v2(
    model_type: str = "rcan_v2",
    scale: int = 3,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create RCAN-DEM v2 models.
    
    Args:
        model_type: "rcan_v2" (full) or "rcan_v2_light" (lightweight)
        scale: Upscaling factor
        **kwargs: Additional model arguments
            - dropout: Dropout rate (default: 0.1)
            - use_spatial_attention: Use spatial attention (default: False)
            - use_multiscale: Use multi-scale conv (default: False)
        
    Returns:
        Model instance
    """
    models = {
        "rcan_v2": RCAN_DEM_v2,
        "rcan_v2_light": RCAN_DEM_v2_Light,
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](scale=scale, **kwargs)


def load_v1_weights(
    model_v2: nn.Module,
    v1_checkpoint_path: str,
    strict: bool = False,
) -> nn.Module:
    """
    Load v1 model weights into v2 model.
    
    Provides backward compatibility with v1 checkpoints.
    New v2 components (dropout, attention) will be randomly initialized.
    
    Args:
        model_v2: RCAN_DEM_v2 instance
        v1_checkpoint_path: Path to v1 checkpoint
        strict: Require exact key match (default: False)
        
    Returns:
        Model with v1 weights loaded (where compatible)
    """
    checkpoint = torch.load(v1_checkpoint_path, map_location="cpu")
    v1_state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # Get v2 state dict
    v2_state_dict = model_v2.state_dict()
    
    # Find compatible keys
    loaded_keys = []
    for k, v in v1_state_dict.items():
        if k in v2_state_dict and v2_state_dict[k].shape == v.shape:
            v2_state_dict[k] = v
            loaded_keys.append(k)
    
    # Load updated state dict
    model_v2.load_state_dict(v2_state_dict, strict=strict)
    
    print(f"Loaded {len(loaded_keys)}/{len(v1_state_dict)} v1 weights")
    
    return model_v2


# Quick test
if __name__ == "__main__":
    print("Testing RCAN-DEM v2...")
    print("=" * 60)
    
    # Test RCAN_DEM_v2 (full)
    model = RCAN_DEM_v2(
        scale=3,
        dropout=0.1,
        use_spatial_attention=False,
        use_multiscale=False,
    )
    print(model)
    
    # Test forward pass
    x = torch.randn(1, 1, 256, 256)
    model.eval()
    with torch.no_grad():
        y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected: [1, 1, 768, 768]")
    assert y.shape == (1, 1, 768, 768), "Shape mismatch!"
    
    # Test with spatial attention
    print("\n" + "-" * 40)
    model_attn = RCAN_DEM_v2(
        scale=3,
        dropout=0.1,
        use_spatial_attention=True,
        use_multiscale=True,
    )
    print(f"With attention + multiscale: {model_attn.get_num_params():,} params")
    
    model_attn.eval()
    with torch.no_grad():
        y_attn = model_attn(x)
    print(f"Output shape (attention): {y_attn.shape}")
    assert y_attn.shape == (1, 1, 768, 768), "Shape mismatch!"
    
    # Test lightweight version
    print("\n" + "-" * 40)
    model_light = RCAN_DEM_v2_Light(scale=3, dropout=0.1)
    print(f"Light model params: {model_light.get_num_params():,}")
    
    model_light.eval()
    with torch.no_grad():
        y_light = model_light(x)
    print(f"Output shape (light): {y_light.shape}")
    assert y_light.shape == (1, 1, 768, 768), "Shape mismatch!"
    
    # Test factory function
    print("\n" + "-" * 40)
    model_factory = create_model_v2("rcan_v2", scale=3, dropout=0.15)
    print(f"Factory model params: {model_factory.get_num_params():,}")
    print(f"Factory model config: {model_factory.get_config()}")
    
    print("\n" + "=" * 60)
    print("All RCAN-DEM v2 tests passed!")
