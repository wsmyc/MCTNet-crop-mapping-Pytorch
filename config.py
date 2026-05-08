"""
Central configuration for the unified MCTNet project.
Modify paths and hyperparameters here only.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA CONFIGURATION
# =============================================================================
BAND_NAMES = [
    "B2_Blue", "B3_Green", "B4_Red", "B5_RedEdge1",
    "B6_RedEdge2", "B7_RedEdge3", "B8_NIR", "B8A_RedEdge4",
    "B11_SWIR1", "B12_SWIR2"
]

BAND_INDICES = {name: i for i, name in enumerate(BAND_NAMES)}
B4, B5, B6, B7, B8 = 2, 3, 4, 5, 6  # Shortcuts for index math

# Vegetation indices added in Part 3
INDEX_NAMES = ["NDVI", "IRECI", "MTCI", "S2REP"]

# =============================================================================
# REGION & CLASS CONFIGURATION
# =============================================================================

# Arkansas classes mapped to your Unique IDs: [1, 2, 3, 5, 176]
# Standard CDL: 1=Corn, 2=Cotton, 3=Rice, 5=Soybeans, 176=Grassland/Pasture
AR_CLASSES = {
    1: "Corn",
    2: "Cotton",
    3: "Rice",
    5: "Soybeans",
    0: "Others"
}

# California classes mapped to your Unique IDs: [3, 36, 69, 75, 76, 176]
# Standard CDL: 3=Rice, 36=Almonds, 69=Grapes, 75=Almonds(alt)/Pistachios, 76=Walnuts, 176=Grassland
CA_CLASSES = {
    3: "Rice",
    36: "Almonds",
    69: "Grapes",
    75: "Pistachios",
    36 : "Alfalfa",
    0: "Others"
}

REGION_CONFIG = {
    "arkansas": {
        "classes": AR_CLASSES,
        "num_classes": 5,
        "n_samples": 10000,
        "files": {
            "X_test": DATA_PROCESSED / "arkansas_X_test.npy",
            "y_test": DATA_PROCESSED / "arkansas_y_test.npy",
            "mask_test": DATA_PROCESSED / "arkansas_mask_test.npy",
            "X_val": DATA_PROCESSED / "arkansas_X_val.npy",
            "y_val": DATA_PROCESSED / "arkansas_y_val.npy",
            "mask_val": DATA_PROCESSED / "arkansas_mask_val.npy",
            "X_train": DATA_PROCESSED / "arkansas_X_train.npy",
            "y_train": DATA_PROCESSED / "arkansas_y_train.npy",
            "mask_train": DATA_PROCESSED / "arkansas_mask_train.npy",
            "samples": DATA_PROCESSED / "arkansas_samples.csv",
            "covariates": DATA_PROCESSED / "arkansas_covariates.csv",
        }
    },
    "california": {
        "classes": CA_CLASSES,
        "num_classes": 6,
        "n_samples": 10000,
        "files": {
            "X_test": DATA_PROCESSED / "california_X_test.npy",
            "y_test": DATA_PROCESSED / "california_y_test.npy",
            "mask_test": DATA_PROCESSED / "california_mask_test.npy",
            "X_val": DATA_PROCESSED / "california_X_val.npy",
            "y_val": DATA_PROCESSED / "california_y_val.npy",
            "mask_val": DATA_PROCESSED / "california_mask_val.npy",
            "X_train": DATA_PROCESSED / "california_X_train.npy",
            "y_train": DATA_PROCESSED / "california_y_train.npy",
            "mask_train": DATA_PROCESSED / "california_mask_train.npy",
            "samples": DATA_PROCESSED / "california_samples.csv",
            "covariates": DATA_PROCESSED / "california_covariates.csv",
        }
    }
}

# =============================================================================
# COVARIATE GROUPS (Updated to match your actual CSV columns)
# =============================================================================
COVARIATE_GROUPS = {
    "climate": ["clim_precip", "clim_temp"],
    "topo": ["topo_elevation", "topo_slope"],
    "soil": ["soil_texture"],
}

# =============================================================================
# MODEL HYPERPARAMETERS (from paper Table 3)
# =============================================================================
MODEL_CONFIG = {
    "in_channels": 10,           
    "stage_channels": [20, 40, 80],
    "n_stages": 3,
    "n_heads": 2,
    "kernel_size": 3,            
    "dim_feedforward": 64,       
    "dropout": 0.1,
    "d_model_expansion": 2,      
    "use_indices": True,        
    "use_geo_alpe": True,       
    "use_attention_pooling": True,  
    "use_covariates": True,
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 200,
    "lr": 0.001,
    "optimizer": "Adam",
    "early_stop_patience": 40,
    "lr_reduce_patience": 8,
    "lr_reduce_factor": 0.5,
    "min_lr": 1e-5,
    "seed": 42,
}