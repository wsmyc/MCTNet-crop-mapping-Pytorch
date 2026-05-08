import torch
import torch.nn as nn
from config import MODEL_CONFIG
from .fusion import CTFusion
from .attention_pooling import AttentionPooling
from .classifier import RegionClassifier

class MCTNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        n_stages: int = 3,
        n_heads: int = 2, 
        dropout: float = 0.1,
        use_indices: bool = False,
        use_geo_alpe: bool = False,
        use_attention_pooling: bool = False,
        use_covariates: bool = False,
        n_cov: int = 0,
    ):
        super().__init__()
        self.use_attention_pooling = use_attention_pooling
        
        actual_in = int(in_channels + (4 if use_indices else 0))
        self.input_norm = nn.BatchNorm1d(actual_in)
        
        self.stages = nn.ModuleList()
        current_in = actual_in
        stages_cfg = MODEL_CONFIG["stage_channels"]
        
        for i in range(int(n_stages)):
            out_ch = int(stages_cfg[i])
            self.stages.append(CTFusion(
                in_channels=current_in,
                out_channels=out_ch,
                n_heads=n_heads,
                use_geo_alpe=(use_geo_alpe and i == 0),
                max_len=36 // (2**i)
            ))
            current_in = out_ch

        final_dim = int(stages_cfg[-1])
        if use_attention_pooling:
            self.pool = AttentionPooling(final_dim)
        
        self.classifier = RegionClassifier(final_dim, use_covariates, n_cov)

    def forward(self, batch: dict) -> dict:
        x = batch["x"] 
        mask = batch["mask"]
        
        x = self.input_norm(x.transpose(1, 2)).transpose(1, 2)
        
        for stage in self.stages:
            x, mask = stage(x, mask, batch.get("coords"))
        
        features = self.pool(x, mask) if self.use_attention_pooling else torch.max(x, dim=1)[0]
        logits = self.classifier(features, batch["region_id"], batch.get("covariates"))
        
        return {"logits": logits, "features": features}