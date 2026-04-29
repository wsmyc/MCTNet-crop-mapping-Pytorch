"""
Learned Attention Pooling (Part 3 improvement).
Replaces Global Max Pooling with a parametric query that attends over time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.scale = d_model ** -0.5
        nn.init.xavier_normal_(self.query)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: (B, T) 1=valid
        Returns:
            (B, d_model)
        """
        B, T, D = x.shape
        q = self.query.expand(B, 1, D)  # (B, 1, D)
        
        # Attention scores
        scores = torch.matmul(q, x.transpose(1, 2)) * self.scale  # (B, 1, T)
        
        # Masking: set missing positions to -inf
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)  # (B, 1, T)
            scores = scores.masked_fill(mask_expanded == 0, float("-inf"))
        
        attn = F.softmax(scores, dim=-1)  # (B, 1, T)
        # Handle all-masked rows (shouldn't happen but safety first)
        attn = torch.nan_to_num(attn, nan=0.0)
        
        pooled = torch.matmul(attn, x).squeeze(1)  # (B, D)
        return pooled