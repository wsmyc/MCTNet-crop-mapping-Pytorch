#!/usr/bin/env python3
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

script_path = os.path.abspath(__file__)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_path)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import config
from src.datasets.unified_dataset import UnifiedCropDataset, collate_fn
from src.models.mctnet import MCTNet
from src.training.trainer import Trainer
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualizer import plot_training_history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", choices=["arkansas", "california"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    TRAIN_CONFIG = config.TRAIN_CONFIG
    MODEL_CONFIG = config.MODEL_CONFIG
    torch.manual_seed(TRAIN_CONFIG["seed"])
    device = torch.device(args.device)
    
    train_ds = UnifiedCropDataset(
        args.state, "train",
        use_indices=MODEL_CONFIG.get("use_indices", True), 
        use_covariates=MODEL_CONFIG.get("use_covariates", True)
    )

    val_ds = UnifiedCropDataset(
        args.state, "val",
        use_indices=MODEL_CONFIG.get("use_indices", True),
        use_covariates=MODEL_CONFIG.get("use_covariates", True),
        covariate_scaler=train_ds.covariate_scaler,
        spectral_scaler=train_ds.spectral_scaler
    )
    
    test_ds = UnifiedCropDataset(
        args.state, "test",
        use_indices=MODEL_CONFIG.get("use_indices", True),
        use_covariates=MODEL_CONFIG.get("use_covariates", True),
        covariate_scaler=train_ds.covariate_scaler,
        spectral_scaler=train_ds.spectral_scaler
    )
    
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=TRAIN_CONFIG["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    model = MCTNet(
        in_channels=10,
        n_stages=MODEL_CONFIG.get("n_stages", 3),
        n_heads=2,
        use_indices=MODEL_CONFIG.get("use_indices", True),           
        use_geo_alpe=MODEL_CONFIG.get("use_geo_alpe", True),         
        use_attention_pooling=MODEL_CONFIG.get("use_attention_pooling", True), 
        use_covariates=MODEL_CONFIG.get("use_covariates", True),
        n_cov=train_ds.covariate_dim,
    ).to(device)
    
    trainer = Trainer(
        model, device,
        lr=TRAIN_CONFIG["lr"],
        epochs=TRAIN_CONFIG["epochs"],
        early_stop_patience=TRAIN_CONFIG["early_stop_patience"],
        checkpoint_dir="checkpoints"
    )
    
    history, best_path = trainer.fit(train_loader, val_loader, model_name=f"mctnet_enhanced_{args.state}")
    
    os.makedirs("reports/figures", exist_ok=True)
    plot_training_history(history, f"reports/figures/training_enhanced_{args.state}.png")
    
    model.load_state_dict(torch.load(best_path))
    model.eval()
    
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            out = model(batch)["logits"]
            for i, rid in enumerate(batch["region_id"]):
                n_cls = 5 if rid.item() == 0 else 6
                preds.append(out[i, :n_cls].argmax().cpu().item())
                labels.append(batch["y"][i].cpu().item())
    
    metrics = compute_metrics(labels, preds)
    print(f"\nTEST OA: {metrics['OA']:.4f} | Kappa: {metrics['Kappa']:.4f}")

if __name__ == "__main__":
    main()