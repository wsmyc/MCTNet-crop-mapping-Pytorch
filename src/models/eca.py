"""
Efficient Channel Attention (ECA) module.
Adapted from Wang et al. 2020 for 1D temporal features.
"""

import math
import torch
import torch.nn as nn


class ECA1D(nn.Module):
    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        # Adaptive kernel size from ECA-Net paper [^15^]
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) where C=channels, T=time steps
        Returns:
            (B, C, T) with channel attention applied
        """
        # Global average pooling over time
        y = self.avg_pool(x)  # (B, C, 1)
        
        # Reshape for 1D conv over channels: (B, 1, C)
        y = y.squeeze(-1).unsqueeze(1)
        y = self.conv(y)      # (B, 1, C)
        y = self.sigmoid(y)
        
        # Reshape back and scale
        y = y.unsqueeze(-1)   # (B, C, 1)
        return x * y.expand_as(x)