import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import os
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import REGION_CONFIG, COVARIATE_GROUPS
from .indices import compute_vegetation_indices

class UnifiedCropDataset(Dataset):
    def __init__(
        self,
        state: str,
        split: str,
        use_indices: bool = False,
        use_covariates: bool = False,
        covariate_group: str = "all",
        covariate_scaler: StandardScaler = None,
        spectral_scaler: StandardScaler = None,
    ):
        assert state in REGION_CONFIG
        self.state = state
        self.split = split
        self.use_indices = use_indices
        self.use_covariates = use_covariates
        
        # Initialize attributes to prevent AttributeErrors
        self.covariates = None
        self.covariate_dim = 0
        self.covariate_scaler = covariate_scaler
        self.spectral_scaler = spectral_scaler
        self.region_id = 0 if state == "arkansas" else 1

        cfg = REGION_CONFIG[state]
        data_dir = Path(cfg["files"]["samples"]).parent

        # Load raw spectral data
        self.X = np.load(data_dir / f"{state}_X_{split}.npy").astype(np.float32)
        self.y = np.load(data_dir / f"{state}_y_{split}.npy").astype(np.int64)
        self.mask = np.load(data_dir / f"{state}_mask_{split}.npy").astype(np.float32)

        # Re-map classes
        sorted_ids = sorted(cfg["classes"].keys())
        self.class_map = {raw_id: i for i, raw_id in enumerate(sorted_ids)}
        mapping_func = np.vectorize(lambda x: self.class_map.get(x, 0))
        self.y = mapping_func(self.y).astype(np.int64)

        # Spectral Normalization
        if self.spectral_scaler is None:
            if split == "train":
                self.spectral_scaler = StandardScaler()
                N, T, C = self.X.shape
                self.spectral_scaler.fit(self.X.reshape(-1, C))
        
        # Coordinate Normalization
        samples = pd.read_csv(cfg["files"]["samples"])
        split_samples = samples[samples["split"].str.lower() == split].reset_index(drop=True)
        if "lat" in split_samples.columns and "lon" in split_samples.columns:
            coords = split_samples[["lat", "lon"]].values.astype(np.float32)
            coords = np.nan_to_num(coords, nan=np.nanmean(coords, axis=0))
            self.coords = (coords - coords.mean(axis=0)) / (coords.std(axis=0) + 1e-6)
        else:
            self.coords = np.zeros((len(self.y), 2), dtype=np.float32)

        # Covariates
        if use_covariates:
            self._load_covariates(state, data_dir, split_samples, covariate_scaler)

    def _load_covariates(self, state, data_dir, split_samples, scaler):
        cov_path = REGION_CONFIG[state]["files"]["covariates"]
        if not os.path.exists(cov_path):
            return

        df_cov = pd.read_csv(cov_path)
        merged = pd.merge(split_samples[['sample_id']], df_cov, on='sample_id', how='left')
        
        cols = []
        for group, features in COVARIATE_GROUPS.items():
            cols.extend(features)
        
        X_cov = merged[cols].values.astype(np.float32)
        X_cov = np.nan_to_num(X_cov, nan=0.0)

        if scaler is None:
            self.covariate_scaler = StandardScaler()
            self.covariates = self.covariate_scaler.fit_transform(X_cov)
        else:
            self.covariates = scaler.transform(X_cov)
        
        self.covariate_dim = self.covariates.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_raw = self.X[idx]
        if self.spectral_scaler is not None:
            x_raw = self.spectral_scaler.transform(x_raw)
        
        x = torch.from_numpy(x_raw).float()
        mask = torch.from_numpy(self.mask[idx])
        y = torch.tensor(self.y[idx], dtype=torch.long)
        coords = torch.from_numpy(self.coords[idx])
        region_id = torch.tensor(self.region_id, dtype=torch.long)

        if self.use_indices:
            indices = compute_vegetation_indices(x)
            x = torch.cat([x, indices], dim=-1)

        item = {"x": x, "mask": mask, "y": y, "region_id": region_id, "coords": coords}
        if self.use_covariates and self.covariates is not None:
            item["covariates"] = torch.from_numpy(self.covariates[idx])
        return item

def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}