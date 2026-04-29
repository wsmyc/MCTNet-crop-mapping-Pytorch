#!/usr/bin/env python3
"""
Part 2: Covariate Ablation Study.
Trains fusion models with frozen backbones to isolate covariate contribution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader

from config import TRAIN_CONFIG, REGION_CONFIG, COVARIATE_GROUPS
from src.datasets.unified_dataset import UnifiedCropDataset, collate_fn
from src.models.mctnet import MCTNet
from src.training.trainer import Trainer


CONFIGS = {
    "S2_only": None,
    "S2_Climate": "climate",
    "S2_Topo": "topo",
    "S2_Soil": "soil",
    "S2_AllCov": "all",
}


def run_experiment(state, config_name, device):
    cov_group = CONFIGS[config_name]
    use_cov = cov_group is not None
    
    # Load datasets with covariates
    train_ds = UnifiedCropDataset(
        state, "train", use_indices=False, use_covariates=use_cov,
        covariate_group=cov_group or "all"
    )
    val_ds = UnifiedCropDataset(
        state, "val", use_indices=False, use_covariates=use_cov,
        covariate_group=cov_group or "all", covariate_scaler=train_ds.covariate_scaler
    )
    
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    # Build model
    n_cov = train_ds.covariate_dim if use_cov else 0
    model = MCTNet(
        in_channels=10, use_indices=False, use_geo_alpe=False,
        use_attention_pooling=False, use_covariates=use_cov, n_cov=n_cov,
    )
    
    # If not baseline, load pretrained backbone and freeze
    if config_name != "S2_only":
        baseline_path = f"checkpoints/mctnet_baseline_{state}_best.pt"
        state_dict = torch.load(baseline_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        # Freeze backbone
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    
    trainer = Trainer(
        model, device, lr=TRAIN_CONFIG["lr"], epochs=TRAIN_CONFIG["epochs"],
        early_stop_patience=TRAIN_CONFIG["early_stop_patience"],
        checkpoint_dir="checkpoints",
    )
    
    history, best_path = trainer.fit(train_loader, val_loader, model_name=f"ablation_{config_name}_{state}")
    return {
        "config": config_name,
        "state": state,
        "best_oa": max(history["val_oa"]),
        "best_kappa": max(history["val_kappa"]),
        "best_f1": max(history["val_f1"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=["arkansas", "california"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    results = []
    
    for cfg_name in CONFIGS.keys():
        print(f"\n{'='*60}")
        print(f"Running {cfg_name} on {args.state}")
        print(f"{'='*60}")
        res = run_experiment(args.state, cfg_name, device)
        results.append(res)
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string())
    df.to_csv(f"reports/ablation_results_{args.state}.csv", index=False)


if __name__ == "__main__":
    main()