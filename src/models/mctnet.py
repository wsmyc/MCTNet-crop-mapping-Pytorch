"""
Complete MCTNet architecture in PyTorch.
Shared backbone + region-specific heads.
"""

import torch
import torch.nn as nn

from config import MODEL_CONFIG, REGION_CONFIG
from .fusion import CTFusion
from .attention_pooling import AttentionPooling
from .classifier import RegionClassifier


class MCTNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        use_indices: bool = False,
        use_geo_alpe: bool = False,
        use_attention_pooling: bool = False,
        use_covariates: bool = False,
        n_cov: int = 0,
    ):
        super().__init__()
        self.use_indices = use_indices
        self.use_geo_alpe = use_geo_alpe
        self.use_attention_pooling = use_attention_pooling
        
        # Input channels: 10 bands + 4 optional indices
        actual_in = in_channels + (4 if use_indices else 0)
        
        stages_cfg = MODEL_CONFIG["stage_channels"]
        n_stages = MODEL_CONFIG["n_stages"]
        n_heads = MODEL_CONFIG["n_heads"]
        kernel_size = MODEL_CONFIG["kernel_size"]
        dim_ff = MODEL_CONFIG["dim_feedforward"]
        dropout = MODEL_CONFIG["dropout"]
        
        # Build stages
        self.stages = nn.ModuleList()
        current_in = actual_in
        
        for i, out_ch in enumerate(stages_cfg[:n_stages]):
            # Only first stage uses Geo-ALPE per paper design
            use_geo = use_geo_alpe and (i == 0)
            stage = CTFusion(
                in_channels=current_in,
                out_channels=out_ch,
                nhead=n_heads,
                kernel_size=kernel_size,
                dim_feedforward=dim_ff,
                dropout=dropout,
                use_geo_alpe=use_geo,
                max_len=36,
            )
            self.stages.append(stage)
            current_in = out_ch  # After pooling, channels = out_ch
        
        # Temporal pooling
        if use_attention_pooling:
            self.pool = AttentionPooling(stages_cfg[n_stages - 1])
        else:
            self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Classifier
        self.classifier = RegionClassifier(
            feat_dim=stages_cfg[n_stages - 1],
            num_ar_classes=REGION_CONFIG["arkansas"]["num_classes"],
            num_ca_classes=REGION_CONFIG["california"]["num_classes"],
            use_covariates=use_covariates,
            n_cov=n_cov,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, batch: dict) -> dict:
        """
        Args:
            batch: dict with keys:
                - x: (B, 36, C) spectral data
                - mask: (B, 36) 1=valid
                - region_id: (B,) 0=AR, 1=CA
                - coords: (B, 2) lat/lon
                - covariates: (B, n_cov) optional
        Returns:
            dict with 'logits' and 'features'
        """
        x = batch["x"]
        mask = batch["mask"]
        coords = batch.get("coords")
        region_id = batch["region_id"]
        covariates = batch.get("covariates")
        
        # Pass through stages
        for stage in self.stages:
            x, mask = stage(x, mask, coords)
        
        # Temporal pooling
        if self.use_attention_pooling:
            features = self.pool(x, mask)  # (B, D)
        else:
            # AdaptiveMaxPool1d on (B, D, T)
            features = self.pool(x.transpose(1, 2)).squeeze(-1)  # (B, D)
        
        # Classification
        logits = self.classifier(features, region_id, covariates)
        
        return {
            "logits": logits,
            "features": features,
        }