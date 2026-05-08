"""
Transformer Submodule from MCTNet paper (Figure 4).
Uses PyTorch native MultiheadAttention with proper key_padding_mask handling.
"""

import torch
import torch.nn as nn

from .alpe import GeoALPE, SinusoidalPE


class TransformerSubmodule(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int = 64,
        dropout: float = 0.1,
        use_geo_alpe: bool = False,
        max_len: int = 36,
    ):
        super().__init__()
        self.use_geo_alpe = use_geo_alpe
        self.d_model = d_model
        
        if use_geo_alpe:
            self.pos_encoder = GeoALPE(d_model, max_len=max_len)
        else:
            self.pos_encoder = SinusoidalPE(d_model, max_len)
        
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: (B, T) with 1=valid, 0=missing
            coords: (B, 2) required if use_geo_alpe=True
        Returns:
            (B, T, d_model)
        """
        # Positional encoding
        if self.use_geo_alpe:
            pos = self.pos_encoder(x, mask, coords)
        else:
            pos = self.pos_encoder(x)
        
        x_with_pos = x + pos
        
        # Self-attention: key_padding_mask True means IGNORE
        key_padding_mask = ~mask.bool() if mask is not None else None
        attn_out, _ = self.self_attn(
            x_with_pos, x_with_pos, x_with_pos,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x