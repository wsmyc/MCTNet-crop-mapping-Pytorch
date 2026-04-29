#!/usr/bin/env python3
"""
Unified evaluation script for any trained MCTNet variant.
Generates confusion matrices and per-class metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import torch
from torch.utils.data import DataLoader

from config import REGION_CONFIG
from src.datasets.unified_dataset import UnifiedCropDataset, collate_fn
from src.models.mctnet import MCTNet
from src.evaluation.metrics import evaluate_by_region
from src.evaluation.visualizer import plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state", choices=["arkansas", "california"], required=True)
    parser.add_argument("--mode", choices=["baseline", "ablation", "enhanced"], required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Dataset config based on mode
    use_indices = args.mode == "enhanced"
    use_cov = args.mode in ["ablation", "enhanced"]
    use_geo = args.mode == "enhanced"
    use_attn_pool = args.mode == "enhanced"
    
    test_ds = UnifiedCropDataset(
        args.state, "test",
        use_indices=use_indices,
        use_covariates=use_cov,
        covariate_group="all",
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Build matching model
    n_cov = test_ds.covariate_dim if use_cov else 0
    model = MCTNet(
        in_channels=10,
        use_indices=use_indices,
        use_geo_alpe=use_geo,
        use_attention_pooling=use_attn_pool,
        use_covariates=use_cov,
        n_cov=n_cov,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.to(device)
    model.eval()
    
    # Inference
    all_preds, all_labels, all_regions = [], [], []
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
                all_regions.append(rid.cpu().item())
    
    # Metrics
    results = evaluate_by_region(
        all_preds, all_labels, all_regions,
        REGION_CONFIG["arkansas"]["classes"],
        REGION_CONFIG["california"]["classes"],
    )
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS — {args.mode.upper()} on {args.state.upper()}")
    print(f"{'='*60}")
    for region, metrics in results.items():
        print(f"\n{region.upper()}:")
        print(f"  OA:    {metrics['OA']:.4f}")
        print(f"  Kappa: {metrics['Kappa']:.4f}")
        print(f"  F1:    {metrics['F1_macro']:.4f}")
        
        # Confusion matrix
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            classes = REGION_CONFIG[region]["classes"]
            plot_confusion_matrix(
                cm, classes,
                save_path=f"reports/figures/cm_{args.mode}_{args.state}_{region}.png",
                title=f"{args.mode.upper()} — {region.upper()}"
            )
            print(f"  Saved CM to reports/figures/cm_{args.mode}_{args.state}_{region}.png")


if __name__ == "__main__":
    main()