#!/usr/bin/env python3
"""
Part 1: Baseline MCTNet reproduction.
Trains separate models per state to match paper protocol.
"""

import sys
import os
from pathlib import Path


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(Path(__file__).parent.parent))


import argparse
import torch
from torch.utils.data import DataLoader

from config import TRAIN_CONFIG, REGION_CONFIG
from src.datasets.unified_dataset import UnifiedCropDataset, collate_fn
from src.models.mctnet import MCTNet
from src.training.trainer import Trainer
from src.evaluation.metrics import evaluate_by_region

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=["arkansas", "california"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    torch.manual_seed(TRAIN_CONFIG["seed"])
    
    # Datasets
    train_ds = UnifiedCropDataset(args.state, "train", use_indices=False, use_covariates=False)
    val_ds = UnifiedCropDataset(args.state, "val", use_indices=False, use_covariates=False)
    
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = MCTNet(
        in_channels=10,
        use_indices=False,
        use_geo_alpe=False,
        use_attention_pooling=False,
        use_covariates=False,
    )
    
    device = torch.device(args.device)
    trainer = Trainer(
        model, device,
        lr=TRAIN_CONFIG["lr"],
        epochs=TRAIN_CONFIG["epochs"],
        early_stop_patience=TRAIN_CONFIG["early_stop_patience"],
        checkpoint_dir="checkpoints",
    )
    
    history, best_path = trainer.fit(train_loader, val_loader, model_name=f"mctnet_baseline_{args.state}")
    print(f"\nBest model saved to: {best_path}")
    
    # Final evaluation
    model.load_state_dict(torch.load(best_path))
    model.eval()


if __name__ == "__main__":
    main()