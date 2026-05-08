"""
CTFusion Stage: CNN + Transformer fusion with temporal pooling.
Reproduces Figure 3 from Wang et al. 2024.
"""

import torch
import torch.nn as nn

from .cnn_submodule import CNNSubmodule
from .transformer_submodule import TransformerSubmodule


class CTFusion(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_heads: int,
        kernel_size: int = 3,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        use_geo_alpe: bool = False,
        max_len: int = 36,
    ):
        super().__init__()
        self.cnn_branch = CNNSubmodule(in_channels, out_channels, kernel_size)
        self.trans_branch = TransformerSubmodule(
            in_channels, n_heads, dim_feedforward, dropout,
            use_geo_alpe=use_geo_alpe, max_len=max_len,
        )
        # Project transformer output to match CNN output channels
        self.trans_proj = nn.Conv1d(in_channels, out_channels, 1)
        
        # Fusion: concat -> 1x1 conv
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, 1),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.MaxPool1d(2)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C_in)
            mask: (B, T) 1=valid
            coords: (B, 2) optional for Geo-ALPE
        Returns:
            x_out: (B, T//2, C_out)
            mask_out: (B, T//2)
        """
        # CNN branch
        cnn_out = self.cnn_branch(x)  # (B, T, C_out)
        
        # Transformer branch
        trans_out = self.trans_branch(x, mask, coords)  # (B, T, C_in)
        trans_out = self.trans_proj(trans_out.transpose(1, 2)).transpose(1, 2)  # (B, T, C_out)
        
        # Concatenate and fuse
        fused = torch.cat([cnn_out, trans_out], dim=-1)  # (B, T, 2*C_out)
        fused = self.fusion(fused.transpose(1, 2)).transpose(1, 2)  # (B, T, C_out)
        
        # Temporal pooling
        x_out = self.pool(fused.transpose(1, 2)).transpose(1, 2)  # (B, T//2, C_out)
        
        # Mask pooling: if ANY pixel in window is valid, window is valid
        mask_out = self.pool(mask.unsqueeze(1)).squeeze(1)  # (B, T//2)
        mask_out = (mask_out > 0).float()
        
        return x_out, mask_out