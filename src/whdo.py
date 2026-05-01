import numpy as np

# Load your splits
X_train = np.load('data/processed/X_train.npy')
X_val = np.load('data/processed/X_val.npy')
X_test = np.load('data/processed/X_test.npy')

# Check for exact duplicates
train_hashes = set([hash(x.tobytes()) for x in X_train])
val_hashes = set([hash(x.tobytes()) for x in X_val])
test_hashes = set([hash(x.tobytes()) for x in X_test])

print(f"Train-Val overlap: {len(train_hashes & val_hashes)}")
print(f"Train-Test overlap: {len(train_hashes & test_hashes)}")
print(f"Val-Test overlap: {len(val_hashes & test_hashes)}")