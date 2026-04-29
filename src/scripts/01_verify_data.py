#!/usr/bin/env python3
"""
Data verification script.
Validates alignment, generates NDVI curves, and checks covariate coverage.
"""
import os
import sys
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from config import REGION_CONFIG, BAND_INDICES
from src.evaluation.visualizer import plot_ndvi_profiles


def verify_state(state):
    print(f"\n{'='*60}")
    print(f"Verifying {state.upper()}")
    print(f"{'='*60}")
    
    cfg = REGION_CONFIG[state]
    X = np.load(cfg["files"]["X"])
    y = np.load(cfg["files"]["y"]).astype(int) # Ensure integers for dict matching
    mask = np.load(cfg["files"]["mask"])
    
    # DEBUG: Check if the config matches the data
    unique_ids = np.unique(y)
    print(f"IDs in data: {unique_ids}")
    print(f"IDs in config: {list(cfg['classes'].keys())}")
    
    print(f"X shape: {X.shape}, y shape: {y.shape}, mask shape: {mask.shape}")
    
    # Pass the classes explicitly
    plot_ndvi_profiles(
        X, y, mask, BAND_INDICES, cfg["classes"],
        save_path=f"reports/figures/ndvi_{state}.png"
    )
    print(f"Saved NDVI plot to reports/figures/ndvi_{state}.png")
    # ... rest of function
    
    # Check covariates
    cov_path = cfg["files"]["covariates"]
    if cov_path.exists():
        import pandas as pd
        df = pd.read_csv(cov_path)
        print(f"Covariates: {len(df)} rows, {len(df.columns)} cols")
        print(f"NaN rate: {df.isna().mean().mean()*100:.2f}%")
    else:
        print(f"WARNING: Covariate file not found: {cov_path}")


if __name__ == "__main__":
    for state in ["arkansas", "california"]:
        verify_state(state)
    print("\n✅ Data verification complete.")