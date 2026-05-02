import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 1. Setup Paths
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

def process_state(state_name, file_path):
    print(f"\n--- Processing {state_name.upper()} ---")

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if len(df) > N_SAMPLES:
        df = df.sample(n=N_SAMPLES, random_state=96).reset_index(drop=True)

    if 'cropland' not in df.columns:
        print(f"❌ Error: Column 'cropland' not found in {state_name}.")
        return

    y = df['cropland'].values

    if 'lat' in df.columns and 'lon' in df.columns:
        lats = df['lat'].values
        lons = df['lon'].values
    else:
        print(f"⚠️ Coordinates missing in {state_name}, generating synthetic coords...")
        if state_name == "arkansas":
            lats = np.random.uniform(33.0, 36.5, size=len(df))
            lons = np.random.uniform(-94.6, -89.6, size=len(df))
        elif state_name == "california":
            lats = np.random.uniform(32.0, 42.0, size=len(df))
            lons = np.random.uniform(-124.0, -114.0, size=len(df))
        else:
            lats, lons = np.zeros(len(df)), np.zeros(len(df))

    band_cols = [c for c in df.columns if 'B' in c][:TIME_STEPS * BANDS]
    if len(band_cols) < (TIME_STEPS * BANDS):
        print(f"❌ Error: Not enough band columns!")
        return

    X_raw = df[band_cols].values
    X_reshaped = X_raw.reshape(-1, TIME_STEPS, BANDS).astype(np.float32)
    mask = np.ones((len(df), TIME_STEPS), dtype=np.float32)

    # 5. Split Logic
    indices = np.arange(len(df))
    target_classes = [1, 2, 3, 5, 36, 204, 69, 75]

    # Map classes: Keep targets, group everything else as -1 (Others)
    mapped_y = np.array([val if val in target_classes else -1 for val in y])
    unique_groups = np.unique(mapped_y)

    train_idx, val_idx, test_idx = [], [], []

    for group in unique_groups:
        group_indices = indices[mapped_y == group]

        # --- SPATIAL BLOCKING: sort by latitude before slicing ---
        # This ensures train/val/test pixels are geographically separated
        # instead of randomly interleaved from the same fields.
        group_lats = lats[group_indices]
        spatial_order = np.argsort(group_lats)
        group_indices = group_indices[spatial_order]
        # ---------------------------------------------------------

        # Slicing: 240 Train, 60 Val, Everything else Test (unchanged from paper)
        if len(group_indices) >= 300:
            c_train = group_indices[:240]
            c_val = group_indices[240:300]
            c_test = group_indices[300:]
        else:
            # Fallback for rare classes
            t_split = int(0.8 * len(group_indices))
            c_train = group_indices[:t_split]
            c_val = group_indices[t_split:]
            c_test = np.array([], dtype=int)

        train_idx.extend(c_train)
        val_idx.extend(c_val)
        test_idx.extend(c_test)

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    # Build split column for samples CSV (kept for reference/logging)
    split_column = np.empty(len(df), dtype=object)
    split_column[train_idx] = 'train'
    split_column[val_idx]   = 'val'
    split_column[test_idx]  = 'test'

    available_covs = [c for c in COV_COLS if c in df.columns]
    if 'sample_id' in df.columns:
        df_covs = df[['sample_id'] + available_covs]
    else:
        df_covs = df[available_covs].copy()
        df_covs.insert(0, 'sample_id', range(len(df)))

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- SAVE PRE-SPLIT ARRAYS ---
    # Training script must load these directly instead of re-splitting X/y.
    # This prevents leakage from a secondary random split on the full array.
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

    # Save full arrays too (kept for compatibility / EDA use only — do NOT use for training)
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_X.npy"),    X_reshaped)
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_y.npy"),    y)
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_mask.npy"), mask)

    samples_df = pd.DataFrame({
        'sample_id': df['sample_id'] if 'sample_id' in df.columns else range(len(df)),
        'lat': lats,
        'lon': lons,
        'split': split_column
    })

    samples_df.to_csv(os.path.join(OUTPUT_DIR, f"{state_name}_samples.csv"), index=False)
    df_covs.to_csv(os.path.join(OUTPUT_DIR, f"{state_name}_covariates.csv"), index=False)

    # Covariates also split and saved
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