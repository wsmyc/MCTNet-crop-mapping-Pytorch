"""
Training engine with metric tracking, early stopping, and LR scheduling.
Optimized for Multi-Task heads and dynamic class counts.
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import numpy as np

# Import config to get live class counts
from config import REGION_CONFIG

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 0.001,
        epochs: int = 200,
        early_stop_patience: int = 20,
        lr_reduce_patience: int = 8,
        lr_reduce_factor: float = 0.5,
        min_lr: float = 1e-5,
        checkpoint_dir: str = "./checkpoints",
    ):
        self.model = model.to(device)
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=lr_reduce_factor,
            patience=lr_reduce_patience, min_lr=min_lr
        )
        
        self.early_stop_patience = early_stop_patience
        self.best_metric = -np.inf
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_oa": [], "val_kappa": [], "val_f1": []}
    
    def _compute_metrics(self, y_true, y_pred):
        oa = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return oa, kappa, f1
    
    def _run_epoch(self, dataloader, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()
        
        all_preds, all_labels = [], []
        total_loss = 0.0
        
        context = torch.enable_grad if is_train else torch.no_grad
        
        with context():
            for batch in tqdm(dataloader, desc="Train" if is_train else "Val", leave=False):
                # Move everything in the batch dictionary to the GPU
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                
                if is_train:
                    self.optimizer.zero_grad()
                
                # Model returns a dict: {"logits": ..., "features": ...}
                outputs = self.model(batch)
                logits = outputs["logits"] # Shape: (Batch, Max_Possible_Classes)
                
                # --- DYNAMIC SLICING ---
                # Identify state based on region_id of the first sample in batch
                # 0 = Arkansas, 1 = California
                state_key = "arkansas" if batch["region_id"][0].item() == 0 else "california"
                n_cls = REGION_CONFIG[state_key]["num_classes"]
                
                # Slice logits to match the specific state's head
                logits = logits[:, :n_cls]
                
                loss = self.criterion(logits, batch["y"])
                
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                
                # Get Predictions
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["y"].cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        oa, kappa, f1 = self._compute_metrics(all_labels, all_preds)
        return avg_loss, oa, kappa, f1
    
    def fit(self, train_loader, val_loader, model_name: str = "mctnet"):
        best_path = None
        
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            
            train_loss, train_oa, _, _ = self._run_epoch(train_loader, is_train=True)
            val_loss, val_oa, val_kappa, val_f1 = self._run_epoch(val_loader, is_train=False)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_oa"].append(val_oa)
            self.history["val_kappa"].append(val_kappa)
            self.history["val_f1"].append(val_f1)
            
            print(f"Epoch {epoch:03d}/{self.epochs} | "
                  f"Loss: {train_loss:.4f} | Val OA: {val_oa:.4f} | "
                  f"Kappa: {val_kappa:.4f} | F1: {val_f1:.4f} | "
                  f"Time: {time.time()-start:.1f}s")
            
            # Update LR based on Validation Accuracy
            self.scheduler.step(val_oa)
            
            # Check for improvement
            if val_oa > self.best_metric:
                self.best_metric = val_oa
                self.patience_counter = 0
                best_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pt")
                torch.save(self.model.state_dict(), best_path)
                print(f"  --> New Best Model Saved! (OA: {val_oa:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"\n[Early Stopping] No improvement for {self.early_stop_patience} epochs.")
                    break
        
        return self.history, best_path