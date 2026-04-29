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
    176: "Grassland/Pasture"
}

# California classes mapped to your Unique IDs: [3, 36, 69, 75, 76, 176]
# Standard CDL: 3=Rice, 36=Almonds, 69=Grapes, 75=Almonds(alt)/Pistachios, 76=Walnuts, 176=Grassland
CA_CLASSES = {
    3: "Rice",
    36: "Almonds",
    69: "Grapes",
    75: "Pistachios",
    76: "Walnuts",
    176: "Grassland"
}

REGION_CONFIG = {
    "arkansas": {
        "classes": AR_CLASSES,
        "num_classes": len(AR_CLASSES),
        "n_samples": 10000,
        "files": {
            "X": DATA_PROCESSED / "arkansas_X.npy",
            "y": DATA_PROCESSED / "arkansas_y.npy",
            "mask": DATA_PROCESSED / "arkansas_mask.npy",
            "samples": DATA_PROCESSED / "arkansas_samples.csv",
            "covariates": DATA_PROCESSED / "arkansas_covariates.csv",
        }
    },
    "california": {
        "classes": CA_CLASSES,
        "num_classes": len(CA_CLASSES),
        "n_samples": 10000,
        "files": {
            "X": DATA_PROCESSED / "california_X.npy",
            "y": DATA_PROCESSED / "california_y.npy",
            "mask": DATA_PROCESSED / "california_mask.npy",
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
    "n_heads": 5,
    "kernel_size": 3,            
    "dim_feedforward": 64,       
    "dropout": 0.1,
    "d_model_expansion": 2,      
    "use_indices": False,        
    "use_geo_alpe": False,       
    "use_attention_pooling": False,  
    "use_covariates": False,
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
TRAIN_CONFIG = {
    "batch_size": 32,
    "epochs": 200,
    "lr": 0.001,
    "optimizer": "Adam",
    "early_stop_patience": 20,
    "lr_reduce_patience": 8,
    "lr_reduce_factor": 0.5,
    "min_lr": 1e-5,
    "seed": 42,
}