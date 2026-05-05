import pandas as pd
import numpy as np
import os
import re

# ============================================================================
# REPRODUCIBILITY: Fixed random seeds
# ============================================================================
np.random.seed(42)
RNG = np.random.default_rng(42)

# ============================================================================
# 1. Setup Paths
# ============================================================================
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_PATH, "..", "..", "data", "processed")

# Constants
N_SAMPLES = 10000
TIME_STEPS = 36
BANDS = 10
COV_COLS = ['clim_precip', 'clim_temp', 'soil_texture', 'topo_elevation', 'topo_slope']

DATA_FILES = {
    'arkansas': os.path.join(BASE_PATH, 'arkansas.csv'),
    'california': os.path.join(BASE_PATH, 'california.csv')
}

STATE_BOUNDS = {
    "arkansas":   {"lat_min": 33.0,  "lat_max": 36.5,  "lon_min": -94.62, "lon_max": -89.65},
    "california": {"lat_min": 32.53, "lat_max": 42.0,  "lon_min": -124.44,"lon_max": -114.13},
}


# ============================================================================
# 2. Robust Band Column Detection
# ============================================================================


def get_band_columns(df, time_steps, bands):
    """
    Detects band columns with naming pattern: d{t}_B{b}
    e.g., d0_B2, d0_B3, ..., d0_B8A, ..., d35_B12
    """
    # Match pattern like d0_B2, d12_B8A, d35_B11
    band_cols = [c for c in df.columns if re.search(r'^d\d+_B\d+A?$', c)]
    
    if len(band_cols) != time_steps * bands:
        raise ValueError(
            f"❌ Band detection failed. Need {time_steps * bands} columns, "
            f"found {len(band_cols)} matching pattern 'd{{t}}_B{{b}}'. "
            f"Columns: {list(df.columns)}"
        )
    
    # Sort to ensure correct order: d0_B2, d0_B3, ..., d0_B12, d1_B2, ...
    # This guarantees reshape(-1, TIME_STEPS, BANDS) works correctly
    def sort_key(col):
        match = re.match(r'^d(\d+)_B(\d+A?)$', col)
        if match:
            t, b = int(match.group(1)), match.group(2)
            # Define band order to ensure consistent channel ordering
            band_order = {'2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '8A':7, '11':8, '12':9}
            return (t, band_order.get(b, 99))
        return (999, 999)
    
    band_cols = sorted(band_cols, key=sort_key)
    return band_cols
# ============================================================================
# 3. Spatial Blocking (Grid-based)
# ============================================================================
def get_spatial_block_order(indices, lats, lons, n_bins=3):
    """
    Sorts indices by spatial grid blocks. This ensures train/val/test
    slices come from geographically distinct regions instead of being
    randomly interleaved or fake-sorted by latitude only.
    """
    if len(indices) == 0:
        return indices

    lat_vals = lats[indices]
    lon_vals = lons[indices]

    lat_min, lat_max = lat_vals.min(), lat_vals.max()
    lon_min, lon_max = lon_vals.min(), lon_vals.max()

    # Prevent degenerate bins when all points share a coordinate
    eps = 1e-6
    if lat_max - lat_min < eps:
        lat_max += eps
    if lon_max - lon_min < eps:
        lon_max += eps

    lat_bins = np.linspace(lat_min, lat_max, n_bins + 1)
    lon_bins = np.linspace(lon_min, lon_max, n_bins + 1)

    lat_idx = np.digitize(lat_vals, lat_bins) - 1
    lon_idx = np.digitize(lon_vals, lon_bins) - 1
    lat_idx = np.clip(lat_idx, 0, n_bins - 1)
    lon_idx = np.clip(lon_idx, 0, n_bins - 1)

    block_id = lat_idx * n_bins + lon_idx

    # Sort by block_id (spatial region), then lat, then lon
    order = np.lexsort((lon_vals, lat_vals, block_id))
    return indices[order]


# ============================================================================
# 4. Main Processing
# ============================================================================
def process_state(state_name, file_path):
    print(f"\n--- Processing {state_name.upper()} ---")

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if len(df) > N_SAMPLES:
        df = df.sample(n=N_SAMPLES, random_state=42).reset_index(drop=True)

    if 'cropland' not in df.columns:
        print(f"❌ Error: Column 'cropland' not found in {state_name}.")
        return

    y = df['cropland'].values

    # --- Coordinates: real or reproducible synthetic ---
    if 'lat' in df.columns and 'lon' in df.columns:
        lats = df['lat'].values.astype(np.float32)
        lons = df['lon'].values.astype(np.float32)
    else:
        print(f"⚠️ Coordinates missing in {state_name}, generating synthetic coords...")
        bounds = STATE_BOUNDS.get(state_name.lower(), {})
        if bounds:
            lats = RNG.uniform(bounds["lat_min"], bounds["lat_max"], size=len(df))
            lons = RNG.uniform(bounds["lon_min"], bounds["lon_max"], size=len(df))
        else:
            lats = np.zeros(len(df), dtype=np.float32)
            lons = np.zeros(len(df), dtype=np.float32)

    # --- Robust band selection ---
    try:
        band_cols = get_band_columns(df, TIME_STEPS, BANDS)
        print(f"  Detected {len(band_cols)} band columns.")
    except ValueError as e:
        print(e)
        return

    X_raw = df[band_cols].values.astype(np.float32)
    X_reshaped = X_raw.reshape(-1, TIME_STEPS, BANDS)
    mask = np.ones((len(df), TIME_STEPS), dtype=np.float32)

    # --- Split Logic with Spatial Blocking ---
    indices = np.arange(len(df))
    target_classes = [1, 2, 3, 5, 36, 204, 69, 75]
    mapped_y = np.array([val if val in target_classes else -1 for val in y])
    unique_groups = np.unique(mapped_y)

    train_idx, val_idx, test_idx = [], [], []

    for group in unique_groups:
        group_indices = indices[mapped_y == group]

        # SPATIAL BLOCKING: sort by 3×3 geographic grid
        group_indices = get_spatial_block_order(group_indices, lats, lons, n_bins=3)

        # 240 / 60 / rest split (unchanged policy)
        if len(group_indices) >= 300:
            c_train = group_indices[:240]
            c_val = group_indices[240:300]
            c_test = group_indices[300:]
        else:
            t_split = int(0.8 * len(group_indices))
            c_train = group_indices[:t_split]
            c_val = group_indices[t_split:]
            c_test = np.array([], dtype=int)

        train_idx.extend(c_train)
        val_idx.extend(c_val)
        test_idx.extend(c_test)

    train_idx = np.array(train_idx, dtype=int)
    val_idx   = np.array(val_idx, dtype=int)
    test_idx  = np.array(test_idx, dtype=int)

    # --- NORMALIZATION: per-band, fit on TRAIN only ---
    if len(train_idx) > 0:
        X_train = X_reshaped[train_idx]                 # shape: (N_train, T, BANDS)
        band_means = X_train.mean(axis=(0, 1))          # shape: (BANDS,)
        band_stds  = X_train.std(axis=(0, 1))           # shape: (BANDS,)
        band_stds  = np.where(band_stds < 1e-8, 1.0, band_stds)

        # Apply to all splits (broadcasts over samples & time)
        X_reshaped = (X_reshaped - band_means) / band_stds
        
        # Save normalization stats for inference
        np.save(os.path.join(OUTPUT_DIR, f"{state_name}_band_means.npy"), band_means)
        np.save(os.path.join(OUTPUT_DIR, f"{state_name}_band_stds.npy"),  band_stds)
        print(f"  Normalized {BANDS} bands using train-set statistics.")
    # --------------------------------------------------

    # Build split column
    split_column = np.empty(len(df), dtype=object)
    split_column[train_idx] = 'train'
    split_column[val_idx]   = 'val'
    split_column[test_idx]  = 'test'

    # Covariates
    available_covs = [c for c in COV_COLS if c in df.columns]
    if 'sample_id' in df.columns:
        df_covs = df[['sample_id'] + available_covs].copy()
    else:
        df_covs = df[available_covs].copy()
        df_covs.insert(0, 'sample_id', range(len(df)))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- SAVE PRE-SPLIT ARRAYS ---
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_X_train.npy"), X_reshaped[train_idx])
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_X_val.npy"),   X_reshaped[val_idx])
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_X_test.npy"),  X_reshaped[test_idx])

    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_y_train.npy"), y[train_idx])
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_y_val.npy"),   y[val_idx])
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_y_test.npy"),  y[test_idx])

    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_mask_train.npy"), mask[train_idx])
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_mask_val.npy"),   mask[val_idx])
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_mask_test.npy"),  mask[test_idx])
    # ----------------------------

    # Full arrays (EDA only — do NOT use for training)
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_X.npy"),    X_reshaped)
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_y.npy"),    y)
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_mask.npy"), mask)

    # Metadata CSVs
    samples_df = pd.DataFrame({
        'sample_id': df_covs['sample_id'].values,
        'lat': lats,
        'lon': lons,
        'split': split_column
    })
    samples_df.to_csv(os.path.join(OUTPUT_DIR, f"{state_name}_samples.csv"), index=False)
    df_covs.to_csv(os.path.join(OUTPUT_DIR, f"{state_name}_covariates.csv"), index=False)

    # Split covariates
    if len(available_covs) > 0:
        cov_vals = df_covs[available_covs].values.astype(np.float32)
        np.save(os.path.join(OUTPUT_DIR, f"{state_name}_cov_train.npy"), cov_vals[train_idx])
        np.save(os.path.join(OUTPUT_DIR, f"{state_name}_cov_val.npy"),   cov_vals[val_idx])
        np.save(os.path.join(OUTPUT_DIR, f"{state_name}_cov_test.npy"),  cov_vals[test_idx])

    print(f"  Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")
    print(f"✅ Successfully saved {state_name} files to {OUTPUT_DIR}")


if __name__ == "__main__":
    for state, path in DATA_FILES.items():
        process_state(state, path)
    print("\n✅ Data processing complete.")