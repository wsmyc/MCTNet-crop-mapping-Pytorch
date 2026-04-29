"""
Unified PyTorch Dataset for Arkansas and California crop mapping.
Handles alignment, missing data, covariate merging, and sparse coordinates.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from config import REGION_CONFIG, COVARIATE_GROUPS
from .indices import compute_vegetation_indices


class UnifiedCropDataset(Dataset):
    def __init__(
        self,
        state: str,                      # 'arkansas' or 'california'
        split: str,                      # 'train', 'val', 'test'
        use_indices: bool = False,       # Part 3: add vegetation indices
        use_covariates: bool = False,    # Part 2: add environmental data
        covariate_group: str = "all",    # 'climate', 'topo', 'soil', 'all'
        covariate_scaler: StandardScaler = None,  # Fit on train, reuse for val/test
    ):
        assert state in REGION_CONFIG
        assert split in ["train", "val", "test"]
        
        self.state = state
        self.split = split
        self.use_indices = use_indices
        self.use_covariates = use_covariates
        self.covariate_group = covariate_group
        self.region_id = 0 if state == "arkansas" else 1
        
        cfg = REGION_CONFIG[state]
        
        # ---------------------------------------------------------------------
        # Load core satellite data
        # ---------------------------------------------------------------------
        self.X = np.load(cfg["files"]["X"]).astype(np.float32)      # (N, 36, 10)
        self.y = np.load(cfg["files"]["y"]).astype(np.int64)
        self.mask = np.load(cfg["files"]["mask"]).astype(np.float32) # (N, 36), 1=valid

        # --- NEW MAPPING LOGIC START ---
        # Map CDL IDs (1, 2, 3, 5, 176) to range [0, 4]
        # We sort the keys to ensure the mapping is consistent
        sorted_ids = sorted(cfg["classes"].keys())
        self.class_map = {raw_id: i for i, raw_id in enumerate(sorted_ids)}
        
        # Use a vectorized mapping for speed
        # If an ID isn't in the config, it defaults to 0 (or you could handle errors)
        mapping_func = np.vectorize(lambda x: self.class_map.get(x, 0)) 
        self.y = mapping_func(self.y).astype(np.int64)
        
        print(f"[{state}] Re-mapped classes to range [0, {len(self.class_map)-1}]")
        # --- NEW MAPPING LOGIC END ---
        
        # Load sample manifest for splits and coordinates
        samples = pd.read_csv(cfg["files"]["samples"])
        
        # Ensure alignment by sample_id if available
        if "sample_id" in samples.columns:
            samples = samples.reset_index(drop=True)
        
        # Filter by split
        split_mask = samples["split"].str.lower() == split
        self.samples = samples[split_mask].reset_index(drop=True)
        
        idx = self.samples.index.values
        self.X = self.X[idx]
        self.y = self.y[idx]
        self.mask = self.mask[idx]
        
        # ---------------------------------------------------------------------
        # Handle sparse lat/lon (Part 3: Geo-ALPE prerequisite)
        # ---------------------------------------------------------------------
        if "lat" in self.samples.columns and "lon" in self.samples.columns:
            coords = self.samples[["lat", "lon"]].values.astype(np.float32)
            # Fill NaN with state-wise mean (computed from available values)
            valid_mask = ~np.isnan(coords).any(axis=1)
            if valid_mask.sum() > 0:
                mean_coords = coords[valid_mask].mean(axis=0)
                coords[~valid_mask] = mean_coords
            else:
                coords[:] = 0.0
            self.coords = coords
        else:
            # Fallback: zeros if coordinates not in CSV
            self.coords = np.zeros((len(self), 2), dtype=np.float32)
        
        # ---------------------------------------------------------------------
        # Load and align covariates (Part 2)
        # ---------------------------------------------------------------------
        if use_covariates:
            self._load_covariates(cfg, covariate_scaler)
        else:
            self.covariates = None
        
        print(f"[{state}/{split}] Loaded {len(self)} samples. "
              f"Indices={use_indices}, Covariates={use_covariates}")
    
    def _load_covariates(self, cfg, scaler):
        cov_path = cfg["files"]["covariates"]
        if not cov_path.exists():
            raise FileNotFoundError(f"Covariate file not found: {cov_path}")
        
        df_cov = pd.read_csv(cov_path)
        
        # Drop rows with any NaN in covariate columns
        all_cov_cols = sum(COVARIATE_GROUPS.values(), [])
        df_cov = df_cov.dropna(subset=[c for c in all_cov_cols if c in df_cov.columns])
        
        # Align with sample_ids
        if "sample_id" in self.samples.columns and "sample_id" in df_cov.columns:
            df_cov = df_cov.set_index("sample_id")
            sample_ids = self.samples["sample_id"].values
            # Reindex and fill missing with mean
            df_cov = df_cov.reindex(sample_ids)
            df_cov = df_cov.fillna(df_cov.mean())
            df_cov = df_cov.reset_index()
        else:
            # Assume row-aligned if no sample_id
            if len(df_cov) != len(self.samples):
                raise ValueError("Covariate row count mismatch and no sample_id to align")
        
        # Select group
        if self.covariate_group == "all":
            cols = sum(COVARIATE_GROUPS.values(), [])
        else:
            cols = COVARIATE_GROUPS[self.covariate_group]
        
        cols = [c for c in cols if c in df_cov.columns]
        X_cov = df_cov[cols].values.astype(np.float32)
        
        # Standardize
        if scaler is None:
            scaler = StandardScaler()
            X_cov = scaler.fit_transform(X_cov)
            self.covariate_scaler = scaler
        else:
            X_cov = scaler.transform(X_cov)
        
        self.covariates = X_cov
        self.covariate_dim = X_cov.shape[1]
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])           # (36, 10)
        mask = torch.from_numpy(self.mask[idx])     # (36,)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        coords = torch.from_numpy(self.coords[idx]) # (2,)
        region_id = torch.tensor(self.region_id, dtype=torch.long)
        
        # Part 3: Append vegetation indices on-the-fly
        if self.use_indices:
            indices = compute_vegetation_indices(x)  # (36, 4)
            x = torch.cat([x, indices], dim=-1)      # (36, 14)
        
        item = {
            "x": x,
            "mask": mask,
            "y": y,
            "region_id": region_id,
            "coords": coords,
        }
        
        if self.use_covariates:
            item["covariates"] = torch.from_numpy(self.covariates[idx])
        
        return item


def collate_fn(batch):
    """Simple collate that preserves dictionary structure."""
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}