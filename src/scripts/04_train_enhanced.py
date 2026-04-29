#!/usr/bin/env python3
"""
Part 3: Enhanced MCTNet with all proposed improvements.
- Vegetation Index Feature Enrichment (10 -> 14 channels)
- Geo-ALPE (lat/lon injection into ALPE)
- Attention Pooling (replaces Global Max Pooling)
- All covariates combined
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from torch.utils.data import DataLoader

from config import TRAIN_CONFIG
from src.datasets.unified_dataset import UnifiedCropDataset, collate_fn
from src.models.mctnet import MCTNet
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualizer import plot_confusion_matrix, plot_training_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=["arkansas", "california"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    torch.manual_seed(TRAIN_CONFIG["seed"])
    device = torch.device(args.device)
    
    # Datasets with ALL improvements
    train_ds = UnifiedCropDataset(
        args.state, "train",
        use_indices=True,       # Part 3: 4 vegetation indices
        use_covariates=True,    # Part 2: all covariates
        covariate_group="all",
    )
    val_ds = UnifiedCropDataset(
        args.state, "val",
        use_indices=True,
        use_covariates=True,
        covariate_group="all",
        covariate_scaler=train_ds.covariate_scaler,
    )
    test_ds = UnifiedCropDataset(
        args.state, "test",
        use_indices=True,
        use_covariates=True,
        covariate_group="all",
        covariate_scaler=train_ds.covariate_scaler,
    )
    
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    # Enhanced model
    model = MCTNet(
        in_channels=10,
        use_indices=True,           # +4 indices
        use_geo_alpe=True,          # Geo-ALPE
        use_attention_pooling=True, # Attention pooling
        use_covariates=True,
        n_cov=train_ds.covariate_dim,
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    trainer = Trainer(
        model, device,
        lr=TRAIN_CONFIG["lr"],
        epochs=TRAIN_CONFIG["epochs"],
        early_stop_patience=TRAIN_CONFIG["early_stop_patience"],
        checkpoint_dir="checkpoints",
    )
    
    # Train
    history, best_path = trainer.fit(train_loader, val_loader, model_name=f"mctnet_enhanced_{args.state}")
    plot_training_history(history, f"reports/figures/training_enhanced_{args.state}.png")
    
    # Test evaluation
    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            outputs = model(batch)
            logits = outputs["logits"]
            for i, rid in enumerate(batch["region_id"]):
                n_cls = 5 if rid.item() == 0 else 6
                pred = logits[i, :n_cls].argmax().cpu().item()
                all_preds.append(pred)
                all_labels.append(batch["y"][i].cpu().item())
    
    metrics = compute_metrics(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"TEST RESULTS — Enhanced MCTNet ({args.state})")
    print(f"{'='*60}")
    print(f"OA:    {metrics['OA']:.4f}")
    print(f"Kappa: {metrics['Kappa']:.4f}")
    print(f"F1:    {metrics['F1_macro']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()