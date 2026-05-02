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
import os
import sys

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_PATH, "..", "..", "data", "processed")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


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
        data_dir = Path(cfg["files"]["samples"]).parent

        # ---------------------------------------------------------------------
        # Load pre-split arrays produced by data_split.py
        # These files already contain only the samples for this split,
        # so no secondary indexing or filtering is needed.
        # ---------------------------------------------------------------------
        X_path    = data_dir / f"{state}_X_{split}.npy"
        y_path    = data_dir / f"{state}_y_{split}.npy"
        mask_path = data_dir / f"{state}_mask_{split}.npy"
        ##This one is working


        if not X_path.exists():
            raise FileNotFoundError(
                f"Pre-split file not found: {X_path}\n"
                f"Re-run data_split.py to generate _X_train/val/test.npy files."
            )

        self.X    = np.load(X_path).astype(np.float32)       # (N_split, 36, 10)
        self.y    = np.load(y_path).astype(np.int64)          # (N_split,)
        self.mask = np.load(mask_path).astype(np.float32)     # (N_split, 36)
        ##respects the split and is aligned with the samples CSV, so no further filtering needed

        # ---------------------------------------------------------------------
        # Map raw CDL IDs to contiguous class indices [0, num_classes-1]
        # ---------------------------------------------------------------------
        sorted_ids = sorted(cfg["classes"].keys())
        self.class_map = {raw_id: i for i, raw_id in enumerate(sorted_ids)}
        mapping_func = np.vectorize(lambda x: self.class_map.get(x, 0))
        self.y = mapping_func(self.y).astype(np.int64)
        print(f"[{state}] Re-mapped classes to range [0, {len(self.class_map)-1}]")
        #so for arkansas we have 5 classes 

        # ---------------------------------------------------------------------
        # Load sample manifest (for coordinates only — split already applied)
        # ---------------------------------------------------------------------
        samples_path = cfg["files"]["samples"]
        samples = pd.read_csv(samples_path)
        samples["split"] = samples["split"].str.lower()

        # Filter to this split to get the matching coordinates
        split_samples = samples[samples["split"] == split].reset_index(drop=True)

        if len(split_samples) != len(self.X):
            raise ValueError(
                f"[{state}/{split}] Mismatch: {len(self.X)} samples in .npy "
                f"but {len(split_samples)} rows in samples CSV. "
                f"Re-run data_split.py to regenerate all files together."
            )

        # ---------------------------------------------------------------------
        # Coordinates (for Geo-ALPE in Part 3)
        # ---------------------------------------------------------------------
        if "lat" in split_samples.columns and "lon" in split_samples.columns:
            coords = split_samples[["lat", "lon"]].values.astype(np.float32)
            valid_mask = ~np.isnan(coords).any(axis=1)
            if valid_mask.sum() > 0:
                mean_coords = coords[valid_mask].mean(axis=0)
                coords[~valid_mask] = mean_coords
            else:
                coords[:] = 0.0
            self.coords = coords
        else:
            self.coords = np.zeros((len(self.y), 2), dtype=np.float32)

        # ---------------------------------------------------------------------
        # Load and align covariates (Part 2)
        # ---------------------------------------------------------------------
        if use_covariates:
            self._load_covariates(state, data_dir, split_samples, covariate_scaler)
        else:
            self.covariates = None

        print(f"[{state}/{split}] Loaded {len(self)} samples. "
              f"Indices={use_indices}, Covariates={use_covariates}")

    def _load_covariates(self, state, data_dir, split_samples, scaler):
        """
        Load pre-split covariate arrays if available, otherwise fall back
        to the full CSV filtered by sample_id.
        """
        cov_npy = data_dir / f"{state}_cov_{self.split}.npy"

        if cov_npy.exists():
            # Fast path: pre-split covariate array from data_split.py
            X_cov = np.load(cov_npy).astype(np.float32)
        else:
            # Fallback: load from CSV and align by sample_id
            cov_path = Path(REGION_CONFIG[state]["files"]["covariates"])
            if not cov_path.exists():
                raise FileNotFoundError(f"Covariate file not found: {cov_path}")

            df_cov = pd.read_csv(cov_path)
            all_cov_cols = sum(COVARIATE_GROUPS.values(), [])
            df_cov = df_cov.dropna(subset=[c for c in all_cov_cols if c in df_cov.columns])

            if "sample_id" in split_samples.columns and "sample_id" in df_cov.columns:
                df_cov = df_cov.set_index("sample_id")
                sample_ids = split_samples["sample_id"].values
                df_cov = df_cov.reindex(sample_ids).fillna(df_cov.mean()).reset_index()
            else:
                if len(df_cov) != len(split_samples):
                    raise ValueError("Covariate row count mismatch and no sample_id to align")

            if self.covariate_group == "all":
                cols = sum(COVARIATE_GROUPS.values(), [])
            else:
                cols = COVARIATE_GROUPS[self.covariate_group]
            cols = [c for c in cols if c in df_cov.columns]
            X_cov = df_cov[cols].values.astype(np.float32)

        # Standardize: fit only on train, transform val/test
        if scaler is None:
            if self.split != "train":
                raise ValueError(
                    "covariate_scaler must be provided for val/test splits. "
                    "Fit a StandardScaler on the train dataset first and pass it here."
                )
            scaler = StandardScaler()
            X_cov = scaler.fit_transform(X_cov)
            self.covariate_scaler = scaler
        else:
            X_cov = scaler.transform(X_cov)
            self.covariate_scaler = scaler

        self.covariates = X_cov
        self.covariate_dim = X_cov.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x      = torch.from_numpy(self.X[idx])           # (36, 10)
        mask   = torch.from_numpy(self.mask[idx])        # (36,)
        y      = torch.tensor(self.y[idx], dtype=torch.long)
        coords = torch.from_numpy(self.coords[idx])      # (2,)
        region_id = torch.tensor(self.region_id, dtype=torch.long)

        if self.use_indices:
            indices = compute_vegetation_indices(x)      # (36, 4)
            x = torch.cat([x, indices], dim=-1)          # (36, 14)

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