<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.2.0-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  <img src="https://img.shields.io/badge/Status-Active-success?style=flat-square" alt="Status">
</p>

<h1 align="center"> MCTNet: Multi-Stage CNN-Transformer Network for Pixel-Based Crop Mapping</h1>

<p align="center">
  <b>A lightweight deep learning framework for time-series Sentinel-2 crop classification with robust handling of missing data</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

This repository contains a complete implementation of **MCTNet** (Multi-stage CNN-Transformer Network), a lightweight deep learning architecture designed for **pixel-based crop mapping** using time-series Sentinel-2 imagery. The model integrates the complementary strengths of Convolutional Neural Networks (CNNs) and Transformers to capture both local spectral patterns and long-range temporal dependencies in agricultural remote sensing data.

Unlike patch-based approaches that require extensive labeling, MCTNet operates at the **pixel level**, significantly reducing annotation costs. The model is specifically engineered to handle **missing values** in time-series data caused by cloud cover and non-observations — a common challenge in satellite imagery — through an innovative Attention-based Learnable Positional Encoding (ALPE) module.

The framework supports two distinct agricultural regions:
- **Arkansas** (5 classes: Corn, Cotton, Rice, Soybeans, Others)
- **California** (6 classes: Rice, Almonds, Grapes, Pistachios, Alfalfa, Others)

---

##  Key Features

| Feature | Description |
|---------|-------------|
|  **Hybrid Architecture** | Combines CNN local feature extraction with Transformer global temporal modeling |
|  **Missing Data Robustness** | ALPE module handles cloud gaps without interpolation |
|  **Pixel-Level Classification** | No patch labeling required — classify individual pixels |
|  **Multi-Region Support** | Pre-configured for Arkansas and California crop systems |
|  **Geospatial Covariates** | Optional integration of climate, topography, and soil data |
|  **Vegetation Indices** | Built-in computation of NDVI, IRECI, MTCI, S2REP |
|  **Comprehensive Evaluation** | OA, Kappa, F1-score, confusion matrices, and Grad-CAM visualizations |
|  **Lightweight Design** | Minimal parameters and memory footprint |

---

##  Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                        MCTNet Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│  Input: Time-Series Sentinel-2 Spectral Bands (10 bands)    │
│         + Missing Value Masks                                 │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Stage 1    │───▶│  Stage 2    │───▶│  Stage 3    │     │
│  │ CTFusion    │    │ CTFusion    │    │ CTFusion    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│         │                  │                  │               │
│    ┌────┴────┐        ┌────┴────┐        ┌────┴────┐      │
│    │ CNN     │        │ CNN     │        │ CNN     │      │
│    │ (Local) │        │ (Local) │        │ (Local) │      │
│    └────┬────┘        └────┬────┘        └────┬────┘      │
│    ┌────┴────┐        ┌────┴────┐        ┌────┴────┐      │
│    │Transformer│      │Transformer│      │Transformer│    │
│    │ (Global)│        │ (Global)│        │ (Global)│    │
│    │  +ALPE  │        │         │        │         │    │
│    └────┬────┘        └────┬────┘        └────┬────┘    │
│         └──────────────────┴──────────────────┘          │
│                           │                               │
│                    ┌─────────────┐                        │
│                    │   Fusion    │                        │
│                    │   Module    │                        │
│                    └──────┬──────┘                        │
│                           │                               │
│                    ┌─────────────┐                        │
│                    │  Classifier │                        │
│                    │    (MLP)    │                        │
│                    └──────┬──────┘                        │
│                           ▼                               │
│                    Crop Classification                     │
└─────────────────────────────────────────────────────────────┘
```

### Module Descriptions

| Module | File | Purpose |
|--------|------|---------|
| **CTFusion** | `src/models/fusion.py` | Fuses CNN and Transformer features at each stage |
| **CNN Submodule** | `src/models/cnn_submodule.py` | 1D convolutions for local spectral-temporal features |
| **Transformer Submodule** | `src/models/transformer_submodule.py` | Self-attention for global temporal dependencies |
| **ALPE** | `src/models/alpe.py` | Attention-based Learnable Positional Encoding for missing data |
| **ECA** | `src/models/eca.py` | Efficient Channel Attention for feature recalibration |
| **Classifier** | `src/models/classifier.py` | Multi-layer perceptron for final prediction |

---

## Repository Structure

```
mctnet-crop-mapping/
│
├── 📂 config.py                    # Central configuration (paths, hyperparameters)
├── 📂 requirements.txt             # Python dependencies
│
├── 📂 data/
│   └── processed/                  # Preprocessed datasets
│       ├── arkansas_X_train.npy    # Training features (Arkansas)
│       ├── arkansas_y_train.npy    # Training labels (Arkansas)
│       ├── arkansas_X_val.npy      # Validation features
│       ├── arkansas_X_test.npy     # Test features
│       ├── arkansas_covariates.csv # Geospatial covariates
│       └── california_*            # California equivalents
│
├── 📂 src/
│   ├── 📂 datasets/
│   │   ├── unified_dataset.py      # PyTorch Dataset wrapper
│   │   ├── data_split.py           # Train/val/test splitting
│   │   └── indices.py              # Vegetation index computation
│   │
│   ├── 📂 models/
│   │   ├── mctnet.py              # Main MCTNet model
│   │   ├── fusion.py              # CNN-Transformer fusion
│   │   ├── cnn_submodule.py       # CNN feature extractor
│   │   ├── transformer_submodule.py # Transformer encoder
│   │   ├── alpe.py                # Attention-based Positional Encoding
│   │   ├── eca.py                 # Efficient Channel Attention
│   │   ├── classifier.py          # MLP classifier head
│   │   └── attention_pooling.py   # Attention pooling mechanism
│   │
│   ├── 📂 training/
│   │   └── trainer.py             # Training loop with early stopping
│   │
│   ├── 📂 evaluation/
│   │   ├── metrics.py             # Accuracy, Kappa, F1 computation
│   │   └── visualizer.py          # Confusion matrices & Grad-CAM
│   │
│   ├── 📂 utils/
│   │   └── helpers.py             # Utility functions
│   │
│   └── 📂 scripts/
│       ├── 01_verify_data.py      # Data integrity check
│       ├── 02_train_baseline.py   # Baseline MCTNet training
│       ├── 03_train_ablation.py   # Ablation study experiments
│       ├── 04_train_enhanced.py   # Enhanced model (indices + covariates)
│       └── 05_evaluate.py         # Full evaluation pipeline
│
└── 📂 reports/
    └── figures/                    # Generated visualizations
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mctnet-crop-mapping.git
cd mctnet-crop-mapping

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.2.0 | Deep learning framework |
| NumPy | 1.26.4 | Numerical computing |
| Pandas | 2.2.0 | Data manipulation |
| Scikit-learn | 1.4.0 | Metrics and preprocessing |
| Matplotlib | 3.8.3 | Visualization |
| Seaborn | 0.13.2 | Statistical plotting |
| PyYAML | 6.0.1 | Configuration files |
| tqdm | 4.66.2 | Progress bars |

---

## Usage

### 1. Verify Data Integrity

```bash
python src/scripts/01_verify_data.py
```

Checks that all preprocessed `.npy` and `.csv` files are present and correctly shaped.

### 2. Train Baseline Model

```bash
python src/scripts/02_train_baseline.py --region arkansas --epochs 200
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--region` | `arkansas` | Study region (`arkansas` or `california`) |
| `--epochs` | `200` | Maximum training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `0.001` | Learning rate |

### 3. Train Enhanced Model (with Vegetation Indices & Covariates)

```bash
python src/scripts/04_train_enhanced.py --region california --use_indices --use_covariates
```

**Enhancement flags:**
- `--use_indices`: Enable NDVI, IRECI, MTCI, S2REP computation
- `--use_covariates`: Integrate climate, topography, and soil data
- `--use_attention_pooling`: Use attention-based pooling instead of global average

### 4. Run Ablation Studies

```bash
python src/scripts/03_train_ablation.py --region arkansas --ablate alpe
```

**Ablation options:** `alpe`, `cnn`, `transformer`, `eca`

### 5. Evaluate Model

```bash
python src/scripts/05_evaluate.py --region arkansas --checkpoint src/checkpoints/mctnet_baseline_arkansas_best.pt
```

Generates:
- Confusion matrix heatmap
- Per-class precision/recall/F1 report
- Overall Accuracy (OA), Kappa coefficient, Macro-F1
- Grad-CAM feature visualization

---

## ⚙️ Configuration

All hyperparameters and paths are centralized in `config.py`:

```python
# Model architecture
MODEL_CONFIG = {
    "in_channels": 10,              # Sentinel-2 spectral bands
    "stage_channels": [20, 40, 80], # Channels per stage
    "n_stages": 3,                  # Number of CTFusion stages
    "n_heads": 5,                   # Transformer attention heads
    "kernel_size": 3,               # CNN kernel size
    "dim_feedforward": 64,        # Transformer FFN dimension
    "dropout": 0.1,
    "d_model_expansion": 2,
    "use_indices": False,           # Vegetation indices
    "use_geo_alpe": True,          # Geographic ALPE
    "use_attention_pooling": False,
    "use_covariates": False,
}

# Training
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
```

---

## Results

### Performance Metrics

| Region | OA | Kappa | Macro-F1 | Parameters |
|--------|-----|-------|----------|------------|
| **Arkansas** | **0.968** | **0.951** | **0.933** | ~50K |
| **California** | **0.852** | **0.806** | **0.829** | ~50K |

### Key Findings

- **Lightweight**: MCTNet achieves superior performance with the **fewest parameters** compared to 8 advanced baseline models
- **ALPE Impact**: The ALPE module improves Kappa by **4.2%** in Arkansas by effectively handling missing temporal observations
- **Complementary Features**: Grad-CAM visualization confirms CNN and Transformer sub-modules extract complementary local and global patterns
- **Missing Data Robustness**: Model maintains accuracy even with 30%+ missing values in time-series

### Ablation Study

| Configuration | Arkansas Kappa | California Kappa |
|---------------|----------------|------------------|
| Full MCTNet | **0.951** | **0.806** |
| w/o ALPE | 0.913 | 0.772 |
| w/o CNN | 0.938 | 0.791 |
| w/o Transformer | 0.929 | 0.783 |

---

## Advanced Features

### Vegetation Indices

The framework computes standard vegetation indices from Sentinel-2 bands:

| Index | Formula | Bands Used |
|-------|---------|------------|
| **NDVI** | (NIR - Red) / (NIR + Red) | B8, B4 |
| **IRECI** | (NIR - RedEdge1) / (RedEdge1 / RedEdge2) | B8, B5, B6 |
| **MTCI** | (RedEdge2 - RedEdge1) / (RedEdge1 - Red) | B6, B5, B4 |
| **S2REP** | 705 + 35 × ((NIR + Red) / 2 - RedEdge1) / (RedEdge2 - RedEdge1) | B8, B4, B5, B6 |

### Geospatial Covariates

Optional environmental variables per sample:

| Group | Variables | Description |
|-------|-----------|-------------|
| **Climate** | `clim_precip`, `clim_temp` | Precipitation and temperature |
| **Topography** | `topo_elevation`, `topo_slope` | Elevation and slope |
| **Soil** | `soil_texture` | Soil texture class |

---

## Citation

If you use this code or the MCTNet architecture in your research, please cite:

```bibtex
@article{wang2024mctnet,
  title={A lightweight CNN-Transformer network for pixel-based crop mapping using time-series Sentinel-2 imagery},
  author={Wang, Yumiao and Feng, Luwei and Sun, Weiwei and Wang, Lihua and Yang, Gang and Chen, Binjie},
  journal={Computers and Electronics in Agriculture},
  volume={225},
  pages={109370},
  year={2024},
  publisher={Elsevier},
  doi={10.1016/j.compag.2024.109370}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Acknowledgements

- Original MCTNet paper by Wang et al. (2024)
- Sentinel-2 data provided by ESA Copernicus Programme
- USDA Cropland Data Layer (CDL) for ground truth labels

---

## 📧 Contact

For questions or collaboration inquiries, please open an issue or contact the repository maintainer.

