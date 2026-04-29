"""
CNN Submodule from MCTNet paper (Figure 6).
Extracts local spectral patterns along the time dimension.
"""

import torch
import torch.nn as nn


class CNNSubmodule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C_in)
        Returns:
            (B, T, C_out)
        """
        # PyTorch Conv1d expects (B, C, T)
        x_t = x.transpose(1, 2)  # (B, C_in, T)
        
        out = self.conv1(x_t)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + self.skip(x_t)
        out = self.relu(out)
        
        return out.transpose(1, 2)  # (B, T, C_out)