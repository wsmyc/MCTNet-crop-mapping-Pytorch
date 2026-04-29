"""
Region-specific classifier heads with covariate fusion.
Handles the dual-output requirement for Arkansas (5 classes) and California (6 classes).
"""

import torch
import torch.nn as nn


class CovariateFusion(nn.Module):
    def __init__(self, n_cov: int, hidden: int = 16, feat_dim: int = 80):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_cov, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, hidden),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Linear(feat_dim + hidden, feat_dim)
        self.bn = nn.BatchNorm1d(feat_dim)
    
    def forward(self, features: torch.Tensor, covariates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, feat_dim)
            covariates: (B, n_cov)
        """
        cov_emb = self.mlp(covariates)  # (B, hidden)
        fused = torch.cat([features, cov_emb], dim=-1)  # (B, feat_dim + hidden)
        out = self.project(fused)  # (B, feat_dim)
        return self.bn(out)


class RegionClassifier(nn.Module):
    def __init__(
        self,
        feat_dim: int = 80,
        num_ar_classes: int = 5,
        num_ca_classes: int = 6,
        use_covariates: bool = False,
        n_cov: int = 0,
    ):
        super().__init__()
        self.use_covariates = use_covariates
        
        if use_covariates and n_cov > 0:
            self.cov_fusion = CovariateFusion(n_cov, hidden=16, feat_dim=feat_dim)
        else:
            self.cov_fusion = None
        
        # Region-specific heads
        self.head_ar = nn.Linear(feat_dim, num_ar_classes)
        self.head_ca = nn.Linear(feat_dim, num_ca_classes)
    
    def forward(
        self,
        features: torch.Tensor,
        region_id: torch.Tensor,
        covariates: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            features: (B, feat_dim)
            region_id: (B,) with 0=AR, 1=CA
            covariates: (B, n_cov) optional
        Returns:
            (B, max(num_ar_classes, num_ca_classes)) — padded if needed,
            or list of variable-length tensors.
        """
        if self.use_covariates and covariates is not None:
            features = self.cov_fusion(features, covariates)
        
        # Route to correct head per sample
        logits_list = []
        for i, rid in enumerate(region_id):
            if rid.item() == 0:
                logits_list.append(self.head_ar(features[i:i+1]))
            else:
                logits_list.append(self.head_ca(features[i:i+1]))
        
        # Pad to common shape for batching (max 6 classes)
        max_classes = max(self.head_ar.out_features, self.head_ca.out_features)
        batch_logits = torch.zeros(
            len(region_id), max_classes,
            device=features.device, dtype=features.dtype
        )
        
        for i, logit in enumerate(logits_list):
            batch_logits[i, :logit.size(1)] = logit.squeeze(0)
        
        return batch_logits