"""
Efficient Channel Attention (ECA) module.
Fixed for robust 1D temporal feature processing.
"""

import math
import torch
import torch.nn as nn

class ECA1D(nn.Module):
    def __init__(self, channels: int, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        # 1. Dynamically calculate kernel size based on channel count
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        
        # 2. Pooling to collapse the temporal dimension
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 3. 1D Convolution over the CHANNEL dimension
        # Note: Conv1d expects (Batch, Channels, Length). 
        # For ECA, 'Length' is our number of spectral channels.
        self.conv = nn.Conv1d(
            1, 1, kernel_size=kernel_size, 
            padding=(kernel_size - 1) // 2, 
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) - (Batch, Channels, Time)
        Returns:
            (B, C, T) - Channel-weighted input
        """
        # x: [Batch, Channels, Time] e.g., [32, 14, 36]
        
        # 1. Temporal Global Average Pooling
        y = self.avg_pool(x)  # Shape: [B, C, 1]
        
        # 2. Prep for Conv1D over channels
        # We need to swap C and 1 to make it [B, 1, C] so Conv1D sees 'C' as the sequence length
        y = y.transpose(-1, -2) # Shape: [B, 1, C]
        
        # 3. Apply the 1D convolution and sigmoid
        y = self.conv(y)        # Shape: [B, 1, C]
        y = self.sigmoid(y)     # Shape: [B, 1, C]
        
        # 4. Final weight application
        # Transpose back to [B, C, 1]
        y = y.transpose(-1, -2) # Shape: [B, C, 1]
        
        # Multiply x [B, C, T] by y [B, C, 1]. 
        # PyTorch automatically 'broadcasts' the 1 into the T dimension.
        # This is MUCH safer than expand_as() or view().
        return x * y