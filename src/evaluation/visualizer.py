"""
Visualization utilities for confusion matrices, NDVI validation, and training curves.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_confusion_matrix(cm, class_names, save_path, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("val_oa", "Overall Accuracy"), ("val_kappa", "Kappa"), ("val_f1", "Macro F1")]
    
    for ax, (key, title) in zip(axes, metrics):
        ax.plot(history[key], marker='o')
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_ndvi_profiles(X, y, mask, band_indices, class_names, save_path):
    """
    Validate NDVI time-series curves against paper Figure 2.
    X: (N, 36, C) numpy array
    """
    red_idx = band_indices["B4_Red"]
    nir_idx = band_indices["B8_NIR"]
    
    ndvi = np.full((len(X), 36), np.nan, dtype=np.float32)
    valid = mask > 0
    
    for i in range(len(X)):
        red = X[i, :, red_idx]
        nir = X[i, :, nir_idx]
        denom = nir + red
        ndvi[i, valid[i]] = np.where(
            denom[valid[i]] > 1e-6,
            (nir[valid[i]] - red[valid[i]]) / denom[valid[i]],
            np.nan
        )
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    
   # Use class_names.items() to get the actual ID and the Name
    for cls_id, name in class_names.items():
        # idx now looks for the actual ID (e.g., 176 for Grassland)
        idx = (y == cls_id) 
        
        if idx.sum() == 0:
            print(f"DEBUG: No pixels found for {name} (ID: {cls_id})")
            continue
            
        mean_ndvi = np.nanmean(ndvi[idx], axis=0)
        
        # We use a simple counter for color so we don't run out of colors
        plt.plot(mean_ndvi, label=name, linewidth=2.5)
    
    plt.title("NDVI Time-Series Profiles (Mean per crop)")
    plt.xlabel("Time Step (10-day intervals)")
    plt.ylabel("NDVI")
    plt.ylim(-0.2, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()