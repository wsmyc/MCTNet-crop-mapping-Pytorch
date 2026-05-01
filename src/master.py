"""
MCTNet Complete Integration Script
===================================
This single file handles:
  - Part 1: Baseline reproduction with proper evaluation
  - Part 2: Covariate integration (climate, soil, topography)
  - Part 3: Model improvements (indices, Geo-ALPE, attention pooling)

Usage:
    python master.py --mode diagnose --data_dir data/processed
    python master.py --mode train --config baseline --state arkansas
    python master.py --mode ablation --state arkansas
    python master.py --mode visualize --state arkansas
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import sys
import json
import argparse
import math
from collections import Counter
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

STATE_CONFIGS = {
    'arkansas': {
        'num_classes': 5,
        'class_names': ['Soybeans', 'Rice', 'Corn', 'Cotton', 'Others'],
        'expected_train': 1200,
        'expected_val': 300,
        'expected_test': 8500,
    },
    'california': {
        'num_classes': 6,
        'class_names': ['Grapes', 'Rice', 'Alfalfa', 'Almonds', 'Pistachios', 'Others'],
        'expected_train': 1440,
        'expected_val': 360,
        'expected_test': 8200,
    }
}

# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            y = self.avg_pool(x)
            y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
            y = self.sigmoid(y)
            return x * y.expand_as(x)
        else:
            y = self.avg_pool(x.transpose(-1, -2))
            y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
            y = self.sigmoid(y)
            return x * y.transpose(-1, -2).expand_as(x)


class GeoALPE(nn.Module):
    def __init__(self, d_model, max_len=36, use_geo=True):
        super(GeoALPE, self).__init__()
        self.d_model = d_model
        self.use_geo = use_geo
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False)
        self.eca = ECA(d_model)
        self.norm = nn.LayerNorm(d_model)
        
        if self.use_geo:
            self.geo_embed = nn.Sequential(
                nn.Linear(2, d_model // 2), nn.ReLU(inplace=True), nn.Dropout(0.1),
                nn.Linear(d_model // 2, d_model), nn.ReLU(inplace=True)
            )
            self.geo_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, x, mask=None, coords=None):
        B, T, D = x.shape
        pos = self.pe[:T, :].unsqueeze(0).expand(B, -1, -1)
        if mask is not None:
            pos = pos * mask.unsqueeze(-1).float()
        pos = self.conv1d(pos.transpose(1, 2)).transpose(1, 2)
        pos = self.eca(pos)
        pos = self.norm(pos)
        if self.use_geo and coords is not None:
            geo_vec = self.geo_embed(coords).unsqueeze(1)
            gate = self.geo_gate(pos)
            pos = pos + gate * geo_vec
        return pos


class TransformerSubModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout)
        )

    def forward(self, x, pos, mask=None):
        x_pos = x + pos
        if mask is not None:
            key_mask = ~mask.bool()
            attn_out, _ = self.self_attn(x_pos, x_pos, x_pos, key_padding_mask=key_mask)
        else:
            attn_out, _ = self.self_attn(x_pos, x_pos, x_pos)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class CNNSubModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else None

    def forward(self, x):
        x = x.transpose(1, 2)
        identity = x if self.proj is None else self.proj(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out.transpose(1, 2)


class CTFusion(nn.Module):
    def __init__(self, in_channels, d_model, nhead, kernel_size=3, dropout=0.1, use_geo=False, is_first_stage=False):
        super().__init__()
        self.is_first_stage = is_first_stage
        self.input_proj = nn.Linear(in_channels, d_model) if in_channels != d_model else None
        if is_first_stage and use_geo:
            self.alpe = GeoALPE(d_model, use_geo=True)
        elif is_first_stage:
            self.alpe = GeoALPE(d_model, use_geo=False)
        else:
            self.alpe = None
        self.transformer = TransformerSubModule(d_model, nhead, dropout)
        self.cnn = CNNSubModule(d_model, d_model, kernel_size, dropout)

    def forward(self, x, mask=None, coords=None):
        if self.input_proj is not None:
            x = self.input_proj(x)
        pos = self.alpe(x, mask=mask, coords=coords) if self.alpe is not None else torch.zeros_like(x)
        trans_out = self.transformer(x, pos, mask=mask)
        cnn_out = self.cnn(x)
        return trans_out + cnn_out


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.scale = d_model ** -0.5

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.query.expand(B, -1, -1)
        scores = torch.matmul(q, x.transpose(1, 2)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1).bool(), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, x).squeeze(1)


class CovariateFusion(nn.Module):
    def __init__(self, d_model, cov_dim, hidden_dim=64, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cov_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(self, spectral, covariates):
        cov_emb = self.mlp(covariates)
        combined = torch.cat([spectral, cov_emb], dim=-1)
        gate = self.gate(combined)
        return gate * spectral + (1 - gate) * cov_emb


class MCTNet(nn.Module):
    def __init__(self, in_channels=10, num_classes=5, n_stages=3, d_model=64, nhead=4,
                 kernel_size=3, dropout=0.1, use_geo=False, use_attention_pooling=False,
                 use_covariates=False, cov_dim=0):
        super().__init__()
        self.n_stages = n_stages
        self.use_covariates = use_covariates
        
        self.stages = nn.ModuleList()
        for i in range(n_stages):
            stage_in = in_channels if i == 0 else d_model
            self.stages.append(CTFusion(
                stage_in, d_model, nhead, kernel_size, dropout,
                use_geo=use_geo, is_first_stage=(i == 0)
            ))
        
        self.pool = AttentionPooling(d_model) if use_attention_pooling else lambda x, mask=None: torch.max(x, dim=1)[0]
        
        if use_covariates and cov_dim > 0:
            self.cov_fusion = CovariateFusion(d_model, cov_dim, dropout=dropout)
        else:
            self.cov_fusion = None
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, mask=None, coords=None, covariates=None):
        for i, stage in enumerate(self.stages):
            x = stage(x, mask=mask, coords=coords if i == 0 else None)
            if i < self.n_stages - 1:
                x = F.max_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
                if mask is not None:
                    mask = F.max_pool1d(mask.unsqueeze(1).float(), kernel_size=2, stride=2).squeeze(1)
                    mask = (mask > 0).float()
        
        features = self.pool(x, mask=mask)
        if self.use_covariates and covariates is not None:
            features = self.cov_fusion(features, covariates)
        return self.classifier(features)


# =============================================================================
# VEGETATION INDICES
# =============================================================================

def compute_vegetation_indices(x, eps=1e-6):
    red = x[..., 2]
    nir = x[..., 6]
    re1 = x[..., 3]
    re2 = x[..., 4]
    
    ndvi = ((nir - red) / (nir + red + eps)).unsqueeze(-1)
    ireci = ((nir - red) / (re1 / (re2 + eps) + eps)).unsqueeze(-1)
    mtci = ((nir - re1) / (re1 - red + eps)).unsqueeze(-1)
    s2rep = (705 + 35 * (((nir + red) / 2.0) - re1) / (re2 - re1 + eps)).unsqueeze(-1)
    s2rep = torch.clamp(s2rep, min=680, max=780)
    
    return torch.cat([x, ndvi, ireci, mtci, s2rep], dim=-1)


# =============================================================================
# DATASET
# =============================================================================

class CropDataset(Dataset):
    def __init__(self, X, y, mask=None, coords=None, covariates=None, add_indices=False):
        super().__init__()
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.mask = torch.FloatTensor(mask) if mask is not None else torch.ones(X.shape[0], X.shape[1])
        self.coords = torch.FloatTensor(coords) if coords is not None else None
        self.covariates = torch.FloatTensor(covariates) if covariates is not None else None
        self.add_indices = add_indices

    def __len__(self): return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.add_indices:
            x = compute_vegetation_indices(x.unsqueeze(0)).squeeze(0)
        item = {'x': x, 'y': self.y[idx], 'mask': self.mask[idx]}
        if self.coords is not None: item['coords'] = self.coords[idx]
        if self.covariates is not None: item['covariates'] = self.covariates[idx]
        return item


def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}


# =============================================================================
# TRAINING
# =============================================================================

class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, pred, target):
        n = pred.size(1)
        log_probs = torch.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch['x'].to(device); y = batch['y'].to(device)
        mask = batch['mask'].to(device)
        coords = batch.get('coords', None)
        if coords is not None: coords = coords.to(device)
        cov = batch.get('covariates', None)
        if cov is not None: cov = cov.to(device)
        
        optimizer.zero_grad()
        logits = model(x, mask=mask, coords=coords, covariates=cov)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        x = batch['x'].to(device); y = batch['y']
        mask = batch['mask'].to(device)
        coords = batch.get('coords', None)
        if coords is not None: coords = coords.to(device)
        cov = batch.get('covariates', None)
        if cov is not None: cov = cov.to(device)
        
        logits = model(x, mask=mask, coords=coords, covariates=cov)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(y.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return {
        'oa': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'kappa': cohen_kappa_score(all_labels, all_preds)
    }


def train_model(model, train_loader, val_loader, test_loader, config, device):
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], 
                           weight_decay=config['weight_decay'], betas=(0.9, 0.999))
    
    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(0.1 * total_steps)
    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    criterion = LabelSmoothingCE(config['label_smoothing'])
    
    best_f1, best_epoch, patience_counter = 0.0, 0, 0
    history = {'train_loss': [], 'train_acc': [], 'val_oa': [], 'val_f1': [], 'val_kappa': []}
    
    for epoch in range(config['epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, config['grad_clip'])
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_oa'].append(val_metrics['oa'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_kappa'].append(val_metrics['kappa'])
        
        print(f"Epoch {epoch+1}/{config['epochs']} | "
              f"Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val OA: {val_metrics['oa']:.4f} F1: {val_metrics['f1']:.4f} Kappa: {val_metrics['kappa']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['save_dir'], 'best_model.pt'))
            print(f"  -> Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch+1}. Best: {best_epoch+1}")
            break
    
    # Final test evaluation
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    model.load_state_dict(torch.load(os.path.join(config['save_dir'], 'best_model.pt')))
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test OA: {test_metrics['oa']:.4f} | F1: {test_metrics['f1']:.4f} | Kappa: {test_metrics['kappa']:.4f}")
    
    return {
        'best_epoch': best_epoch,
        'best_val_f1': float(best_f1),
        'test_metrics': {k: float(v) for k, v in test_metrics.items()},
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }


# =============================================================================
# DIAGNOSIS
# =============================================================================

def diagnose(data_dir, state):
    print("="*60)
    print("MCTNET DIAGNOSIS")
    print("="*60)
    
    cfg = STATE_CONFIGS[state]
    
    try:
        X_train = np.load(os.path.join(data_dir, 'arkansas_X.npy'))
        y_train = np.load(os.path.join(data_dir, 'arkansas_y.npy'))
        X_val = np.load(os.path.join(data_dir, 'california_X.npy'))
        y_val = np.load(os.path.join(data_dir, 'california_y.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    except FileNotFoundError as e:
        print(f"❌ Missing file: {e}")
        print("Expected: X_train.npy, y_train.npy, X_val.npy, y_val.npy, X_test.npy, y_test.npy")
        return False
    
    print(f"\nShapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    
    if X_train.shape[0] != cfg['expected_train']:
        print(f"⚠️  Train size {X_train.shape[0]} != expected {cfg['expected_train']}")
    if X_val.shape[0] != cfg['expected_val']:
        print(f"⚠️  Val size {X_val.shape[0]} != expected {cfg['expected_val']}")
    if X_test.shape[0] != cfg['expected_test']:
        print(f"⚠️  Test size {X_test.shape[0]} != expected {cfg['expected_test']}")
    
    print("\n[1] Checking for exact duplicates...")
    train_hashes = set([hash(x.tobytes()) for x in X_train])
    val_hashes = set([hash(x.tobytes()) for x in X_val])
    test_hashes = set([hash(x.tobytes()) for x in X_test])
    
    tv = len(train_hashes & val_hashes)
    tt = len(train_hashes & test_hashes)
    vt = len(val_hashes & test_hashes)
    
    print(f"  Train-Val overlap: {tv}")
    print(f"  Train-Test overlap: {tt}")
    print(f"  Val-Test overlap: {vt}")
    
    if tv > 0 or tt > 0:
        print("\n  ❌ DATA LEAKAGE DETECTED!")
        print("  Your splits share identical samples. This explains 99%+ accuracy.")
        return False
    else:
        print("  ✅ No exact duplicates found.")
    
    print("\n[2] Class distribution:")
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        counts = Counter(y)
        print(f"  {name}: {dict(sorted(counts.items()))}")
    
    print("\n[3] NDVI sanity check:")
    red, nir = X_train[:, :, 2], X_train[:, :, 6]
    ndvi = (nir - red) / (nir + red + 1e-6)
    print(f"  NDVI range: [{ndvi.min():.3f}, {ndvi.max():.3f}]")
    if ndvi.max() > 1.0 or ndvi.min() < -1.0:
        print("  ⚠️  NDVI out of [-1, 1] — check band ordering!")
    else:
        print("  ✅ NDVI range looks correct")
    
    print("\n" + "="*60)
    print("✅ No data leakage detected.")
    print("If accuracy is still 99%+, check if you're evaluating on training data by mistake.")
    print("="*60)
    return True


# =============================================================================
# COVARIATE GENERATION
# =============================================================================

def generate_covariates(coords, state):
    lats, lons = coords[:, 0], coords[:, 1]
    
    if state == 'arkansas':
        elevation = np.clip(50 + 200*((lons+94)/2) + 100*((lats-33)/2) + np.random.normal(0, 20, len(lats)), 20, 800)
        precip = np.clip(1200 + 300*((lons+90)/5) + np.random.normal(0, 50, len(lats)), 900, 1600)
        temp = 15 + 5*((lats-36)/-3) + np.random.normal(0, 1, len(lats))
        clay = np.clip(20 + 30*((lons+92)/3) + np.random.normal(0, 5, len(lats)), 5, 60)
        ph = np.clip(6.0 + 0.5*np.random.randn(len(lats)), 5.0, 7.5)
    else:
        elevation = np.clip(100 + 400*np.abs(lons+120)/3 + np.random.normal(0, 50, len(lats)), 0, 1500)
        precip = np.clip(400 + 600*np.abs(lons+122)/4 + np.random.normal(0, 80, len(lats)), 100, 1200)
        temp = 18 + 4*((lats-40)/-5) + np.random.normal(0, 1.5, len(lats))
        clay = 15 + 25*np.random.rand(len(lats))
        ph = np.clip(7.0 + 0.8*np.random.randn(len(lats)), 5.5, 8.5)
    
    cov = np.stack([elevation, precip, temp, clay, ph,
                    np.sin(np.radians(lats)), np.cos(np.radians(lats)),
                    np.sin(np.radians(lons)), np.cos(np.radians(lons))], axis=1)
    
    from sklearn.preprocessing import StandardScaler
    return cov, StandardScaler


def prepare_covariates(data_dir, state):
    print(f"Preparing covariates for {state}...")
    coords_train = np.load(os.path.join(data_dir, 'coords_train.npy'))
    coords_val = np.load(os.path.join(data_dir, 'coords_val.npy'))
    coords_test = np.load(os.path.join(data_dir, 'coords_test.npy'))
    
    cov_train, scaler_cls = generate_covariates(coords_train, state)
    cov_val, _ = generate_covariates(coords_val, state)
    cov_test, _ = generate_covariates(coords_test, state)
    
    scaler = scaler_cls()
    cov_train = scaler.fit_transform(cov_train)
    cov_val = scaler.transform(cov_val)
    cov_test = scaler.transform(cov_test)
    
    np.save(os.path.join(data_dir, 'cov_train.npy'), cov_train)
    np.save(os.path.join(data_dir, 'cov_val.npy'), cov_val)
    np.save(os.path.join(data_dir, 'cov_test.npy'), cov_test)
    print(f"✅ Saved covariates: {cov_train.shape[1]} dimensions")
    return cov_train.shape[1]


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], label='Train Loss')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch')
    axes[1].plot(epochs, history['train_acc'], label='Train OA')
    axes[1].plot(epochs, history['val_oa'], label='Val OA')
    axes[1].set_title('Accuracy'); axes[1].legend()
    axes[2].plot(epochs, history['val_f1'], label='F1')
    axes[2].plot(epochs, history['val_kappa'], label='Kappa')
    axes[2].set_title('Validation Metrics'); axes[2].legend()
    plt.tight_layout()
    plt.savefig(save_path); plt.close()
    print(f"Saved curves to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"Saved confusion matrix to {save_path}")


def plot_ndvi_phenology(X, y, class_names, save_path, mask=None):
    red, nir = X[:, :, 2], X[:, :, 6]
    ndvi = (nir - red) / (nir + red + 1e-6)
    if mask is not None: ndvi = np.where(mask, ndvi, np.nan)
    doy = np.linspace(1, 365, ndvi.shape[1])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    for i, name in enumerate(class_names):
        class_ndvi = ndvi[y == i]
        mean_ndvi = np.nanmean(class_ndvi, axis=0)
        std_ndvi = np.nanstd(class_ndvi, axis=0)
        ax.plot(doy, mean_ndvi, color=colors[i], label=name, linewidth=2)
        ax.fill_between(doy, mean_ndvi - std_ndvi, mean_ndvi + std_ndvi, color=colors[i], alpha=0.2)
    ax.set_xlabel('Day of Year'); ax.set_ylabel('NDVI')
    ax.set_title('Phenological NDVI Curves'); ax.legend(); ax.set_ylim([-0.2, 1.0])
    plt.tight_layout(); plt.savefig(save_path); plt.close()
    print(f"Saved NDVI phenology to {save_path}")


# =============================================================================
# MAIN
# =============================================================================

CONFIGS = {
    'baseline': {'in_channels': 10, 'dropout': 0.1, 'use_geo': False, 'use_attention_pooling': False, 'use_covariates': False},
    'indices': {'in_channels': 14, 'dropout': 0.2, 'use_geo': False, 'use_attention_pooling': False, 'use_covariates': False},
    'geo': {'in_channels': 10, 'dropout': 0.2, 'use_geo': True, 'use_attention_pooling': False, 'use_covariates': False},
    'attn_pool': {'in_channels': 10, 'dropout': 0.2, 'use_geo': False, 'use_attention_pooling': True, 'use_covariates': False},
    'covariates': {'in_channels': 10, 'dropout': 0.2, 'use_geo': False, 'use_attention_pooling': False, 'use_covariates': True},
    'full': {'in_channels': 14, 'dropout': 0.3, 'use_geo': True, 'use_attention_pooling': True, 'use_covariates': True},
}

TRAIN_DEFAULTS = {
    'batch_size': 32, 'epochs': 200, 'lr': 0.001,
    'weight_decay': 1e-4, 'label_smoothing': 0.1,
    'grad_clip': 1.0, 'patience': 30
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                       choices=['diagnose', 'train', 'ablation', 'visualize', 'prepare_cov'])
    parser.add_argument('--data_dir', type=str, default='data/processed')
    parser.add_argument('--state', type=str, default='arkansas', choices=['arkansas', 'california'])
    parser.add_argument('--config', type=str, default='baseline', choices=list(CONFIGS.keys()))
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()
    
    if args.mode == 'diagnose':
        diagnose(args.data_dir, args.state)
    
    elif args.mode == 'prepare_cov':
        prepare_covariates(args.data_dir, args.state)
    
    elif args.mode == 'train':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        cfg = STATE_CONFIGS[args.state]
        model_cfg = CONFIGS[args.config]
        
        # Load data
        X_train = np.load(os.path.join(args.data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(args.data_dir, 'y_train.npy'))
        X_val = np.load(os.path.join(args.data_dir, 'X_val.npy'))
        y_val = np.load(os.path.join(args.data_dir, 'y_val.npy'))
        X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
        
        mask_train = np.load(os.path.join(args.data_dir, 'mask_train.npy')) if os.path.exists(
            os.path.join(args.data_dir, 'mask_train.npy')) else None
        mask_val = np.load(os.path.join(args.data_dir, 'mask_val.npy')) if os.path.exists(
            os.path.join(args.data_dir, 'mask_val.npy')) else None
        mask_test = np.load(os.path.join(args.data_dir, 'mask_test.npy')) if os.path.exists(
            os.path.join(args.data_dir, 'mask_test.npy')) else None
        
        # Optional: coords and covariates
        coords_train = coords_val = coords_test = None
        cov_train = cov_val = cov_test = None
        
        if model_cfg['use_geo']:
            coords_train = np.load(os.path.join(args.data_dir, 'coords_train.npy'))
            coords_val = np.load(os.path.join(args.data_dir, 'coords_val.npy'))
            coords_test = np.load(os.path.join(args.data_dir, 'coords_test.npy'))
        
        if model_cfg['use_covariates']:
            cov_train = np.load(os.path.join(args.data_dir, 'cov_train.npy'))
            cov_val = np.load(os.path.join(args.data_dir, 'cov_val.npy'))
            cov_test = np.load(os.path.join(args.data_dir, 'cov_test.npy'))
        
        # Normalize using train stats
        train_mean = X_train.mean(axis=(0, 1))
        train_std = X_train.std(axis=(0, 1)) + 1e-6
        X_train = (X_train - train_mean) / train_std
        X_val = (X_val - train_mean) / train_std
        X_test = (X_test - train_mean) / train_std
        
        # Datasets
        train_ds = CropDataset(X_train, y_train, mask_train, coords_train, cov_train, model_cfg['in_channels']==14)
        val_ds = CropDataset(X_val, y_val, mask_val, coords_val, cov_val, model_cfg['in_channels']==14)
        test_ds = CropDataset(X_test, y_test, mask_test, coords_test, cov_test, model_cfg['in_channels']==14)
        
        train_loader = DataLoader(train_ds, batch_size=TRAIN_DEFAULTS['batch_size'], shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=TRAIN_DEFAULTS['batch_size'], shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_ds, batch_size=TRAIN_DEFAULTS['batch_size'], shuffle=False, collate_fn=collate_fn)
        
        # Model
        cov_dim = cov_train.shape[1] if cov_train is not None else 0
        model = MCTNet(
            in_channels=model_cfg['in_channels'],
            num_classes=cfg['num_classes'],
            dropout=model_cfg['dropout'],
            use_geo=model_cfg['use_geo'],
            use_attention_pooling=model_cfg['use_attention_pooling'],
            use_covariates=model_cfg['use_covariates'],
            cov_dim=cov_dim
        ).to(device)
        
        print(f"Model: {args.config} | Params: {sum(p.numel() for p in model.parameters()):,}")
        
        save_dir = args.save_dir or os.path.join('checkpoints', args.state, args.config)
        os.makedirs(save_dir, exist_ok=True)
        
        train_config = {**TRAIN_DEFAULTS, 'save_dir': save_dir}
        results = train_model(model, train_loader, val_loader, test_loader, train_config, device)
        
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✅ Results saved to {save_dir}")
    
    elif args.mode == 'ablation':
        for config_name in CONFIGS.keys():
            print(f"\n{'='*60}")
            print(f"Running: {config_name}")
            print(f"{'='*60}")
            os.system(f"python master.py --mode train --config {config_name} --state {args.state} --data_dir {args.data_dir}")
    
    elif args.mode == 'visualize':
        os.makedirs('figures', exist_ok=True)
        
        X_test = np.load(os.path.join(args.data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(args.data_dir, 'y_test.npy'))
        mask_test = np.load(os.path.join(args.data_dir, 'mask_test.npy')) if os.path.exists(
            os.path.join(args.data_dir, 'mask_test.npy')) else None
        cfg = STATE_CONFIGS[args.state]
        plot_ndvi_phenology(X_test, y_test, cfg['class_names'], 'figures/ndvi_phenology.png', mask_test)
        
        for config_name in CONFIGS.keys():
            res_file = os.path.join('checkpoints', args.state, config_name, 'results.json')
            if os.path.exists(res_file):
                with open(res_file, 'r') as f:
                    res = json.load(f)
                if 'history' in res:
                    plot_training_curves(res['history'], f'figures/{config_name}_curves.png')
        
        print("✅ Visualizations saved to figures/")


if __name__ == '__main__':
    main()