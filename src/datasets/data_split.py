import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# 1. Setup Paths
# BASE_PATH is where this script lives
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
# OUTPUT_DIR is relative to the project root (adjust if needed)
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
    
    # Clean column names (removes spaces/newlines)
    df.columns = df.columns.str.strip()
    
    # 1. Sample 10k pixels
    if len(df) > N_SAMPLES:
        df = df.sample(n=N_SAMPLES, random_state=96).reset_index(drop=True)
    
    # 2. Extract Labels (using your updated column name 'cropland')
    if 'cropland' not in df.columns:
        print(f"❌ Error: Column 'cropland' not found in {state_name}. Available: {df.columns.tolist()[:5]}...")
        return
    
    y = df['cropland'].values  
    
    # Handle Coordinates
    # Use real ones if they exist, otherwise use your "Bluffing" logic
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
    
    # 3. Reshape Spectral Data (X)
    band_cols = [c for c in df.columns if 'B' in c][:TIME_STEPS * BANDS]
    if len(band_cols) < (TIME_STEPS * BANDS):
        print(f"❌ Error: Not enough band columns! Found {len(band_cols)}, need {TIME_STEPS * BANDS}")
        return

    X_raw = df[band_cols].values
    X_reshaped = X_raw.reshape(-1, TIME_STEPS, BANDS).astype(np.float32)
    
    # 4. Create Mask
    mask = np.ones((len(df), TIME_STEPS), dtype=np.float32)
    
    # 5. Split Logic
    indices = np.arange(len(df))
    train_idx, temp_idx = train_test_split(indices, test_size=0.18, stratify=y, random_state=96)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.8, stratify=y[temp_idx], random_state=96)
    
    split_column = np.empty(len(df), dtype=object)
    split_column[train_idx] = 'train'
    split_column[val_idx] = 'val'
    split_column[test_idx] = 'test'
    
    # 6. Separate Covariates
    available_covs = [c for c in COV_COLS if c in df.columns]
    # We include 'sample_id' only if it exists in the raw CSV
    if 'sample_id' in df.columns:
        df_covs = df[['sample_id'] + available_covs]
    else:
        df_covs = df[available_covs].copy()
        df_covs.insert(0, 'sample_id', range(len(df)))
    
    # 7. Save Files
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_X.npy"), X_reshaped)
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_y.npy"), y)
    np.save(os.path.join(OUTPUT_DIR, f"{state_name}_mask.npy"), mask)
    
    # Save manifest
    samples_df = pd.DataFrame({
        'sample_id': df['sample_id'] if 'sample_id' in df.columns else range(len(df)),
        'lat': lats,
        'lon': lons,
        'split': split_column
    })
    
    samples_df.to_csv(os.path.join(OUTPUT_DIR, f"{state_name}_samples.csv"), index=False)
    df_covs.to_csv(os.path.join(OUTPUT_DIR, f"{state_name}_covariates.csv"), index=False)
    
    print(f"✅ Successfully saved {state_name} files to {OUTPUT_DIR}")

# --- Execution ---
if __name__ == "__main__":
    for state, path in DATA_FILES.items():
        process_state(state, path)
    print("\n✅ Data processing complete.")