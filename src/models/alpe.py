"""
Attention-based Learnable Positional Encoding (ALPE) and Geo-ALPE.
Corrects the original TF implementation which omitted the mask→Conv1D→ECA pipeline.
"""

import math
import torch
import torch.nn as nn

from .eca import ECA1D


class SinusoidalPE(nn.Module):
    """Standard absolute positional encoding from Vaswani et al. 2017."""
    def __init__(self, d_model: int, max_len: int = 36):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return self.pe[:T].unsqueeze(0).expand(x.size(0), T, -1)


class ALPE(nn.Module):
    """
    Attention-based Learnable Positional Encoding (Wang et al. 2024, Eq. 3).
    ALPE(t) = ECA( Conv1D( PE(t) * mask ) )
    """
    def __init__(self, d_model: int, max_len: int = 36):
        super().__init__()
        self.pe = SinusoidalPE(d_model, max_len)
        # Conv1D along time dimension: operates on (B, D, T)
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size=3,
            padding=1, bias=False
        )
        self.eca = ECA1D(d_model)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model) — used for shape only
            mask: (B, T) with 1=valid, 0=missing
        Returns:
            (B, T, d_model)
        """
        pos = self.pe(x)  # (B, T, d_model)
        # Zero out missing positions
        pos = pos * mask.unsqueeze(-1).float()
        
        # Conv1D expects (B, D, T)
        pos = pos.transpose(1, 2)  # (B, D, T)
        pos = self.conv(pos)
        pos = self.relu(pos)
        pos = self.eca(pos)
        pos = pos.transpose(1, 2)  # (B, T, D)
        return pos


class GeoALPE(nn.Module):
    """
    Geographic ALPE (Part 3 improvement).
    Injects lat/lon into the ALPE pathway to make positional encoding spatially adaptive.
    """
    def __init__(self, d_model: int, max_len: int = 36, d_geo: int = 2):
        super().__init__()
        self.alpe = ALPE(d_model, max_len)
        self.geo_embed = nn.Sequential(
            nn.Linear(d_geo, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, d_model)
        )
    
    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, coords: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: (B, T)
            coords: (B, 2) — lat, lon
        """
        base_pos = self.alpe(x, mask)              # (B, T, d_model)
        geo_bias = self.geo_embed(coords).unsqueeze(1)  # (B, 1, d_model)
        return base_pos + geo_bias