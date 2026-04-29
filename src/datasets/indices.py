"""
Vegetation index computation for Part 3 improvement.
All formulas validated against Sentinel-2 spectral band definitions.
"""

import torch
import torch.nn.functional as F


def compute_vegetation_indices(s2: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute 4 vegetation indices from Sentinel-2 bands.
    
    Args:
        s2: Tensor of shape (..., 10) with bands [B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12]
        eps: Small constant for numerical stability
    
    Returns:
        Tensor of shape (..., 4) with [NDVI, IRECI, MTCI, S2REP]
    """
    # Extract bands
    b4 = s2[..., 2:3]   # Red
    b5 = s2[..., 3:4]   # Red Edge 1
    b6 = s2[..., 4:5]   # Red Edge 2
    b7 = s2[..., 5:6]   # Red Edge 3
    b8 = s2[..., 6:7]   # NIR
    
    # NDVI: (NIR - Red) / (NIR + Red)
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    
    # IRECI: (B7 - B4) / (B5 / B6)  [^4^]
    ireci = (b7 - b4) / (b5 / (b6 + eps) + eps)
    
    # MTCI: (B6 - B5) / (B5 - B4)  [^7^]
    mtci = (b6 - b5) / (b5 - b4 + eps)
    
    # S2REP: 705 + 35 * ((((B7 + B4)/2) - B5) / (B6 - B5))  [^7^]
    s2rep = 705.0 + 35.0 * ((((b7 + b4) / 2.0) - b5) / (b6 - b5 + eps))
    
    # Clamp to reasonable ranges to handle extreme outliers from bad pixels
    ndvi = torch.clamp(ndvi, -1.0, 1.0)
    ireci = torch.clamp(ireci, -10.0, 10.0)
    mtci = torch.clamp(mtci, -10.0, 10.0)
    s2rep = torch.clamp(s2rep, 680.0, 780.0)
    
    return torch.cat([ndvi, ireci, mtci, s2rep], dim=-1)