import torch
import torch.nn as nn
from config import REGION_CONFIG

class RegionClassifier(nn.Module):
    def __init__(self, embed_dim: int, use_covariates: bool = False, n_cov: int = 0):
        super().__init__()
        
        feat_dim = int(embed_dim)
        if use_covariates:
            feat_dim += int(n_cov)
            
        num_ar_classes = int(len(REGION_CONFIG["arkansas"]["classes"]))
        num_ca_classes = int(len(REGION_CONFIG["california"]["classes"]))
        
        self.head_ar = nn.Linear(feat_dim, num_ar_classes)
        self.head_ca = nn.Linear(feat_dim, num_ca_classes)
        
    def forward(self, x, region_ids, covariates=None):
        if covariates is not None:
            x = torch.cat([x, covariates], dim=-1)
            
        logits = torch.zeros(x.size(0), 6, device=x.device)
        
        mask_ar = (region_ids == 0)
        mask_ca = (region_ids == 1)
        
        if mask_ar.any():
            logits[mask_ar, :5] = self.head_ar(x[mask_ar])
        if mask_ca.any():
            logits[mask_ca, :6] = self.head_ca(x[mask_ca])
            
        return logits