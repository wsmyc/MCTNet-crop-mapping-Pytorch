"""
Training engine with metric tracking, early stopping, and LR scheduling.
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
            patience=lr_reduce_patience, min_lr=min_lr, verbose=True
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
            for batch in tqdm(dataloader, desc="Train" if is_train else "Val"):
                # Move to device
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                
                if is_train:
                    self.optimizer.zero_grad()
                
                outputs = self.model(batch)
                logits = outputs["logits"]
                
                # Slice logits to correct class count per sample
                # region_id: 0=AR (5 classes), 1=CA (6 classes)
                losses = []
                for i, rid in enumerate(batch["region_id"]):
                    n_cls = 5 if rid.item() == 0 else 6
                    logit_slice = logits[i, :n_cls].unsqueeze(0)
                    label = batch["y"][i].unsqueeze(0)
                    losses.append(self.criterion(logit_slice, label))
                
                loss = torch.stack(losses).mean()
                
                if is_train:
                    loss.backward()
                    self.optimizer.step()
                
                total_loss += loss.item()
                
                # Predictions
                for i, rid in enumerate(batch["region_id"]):
                    n_cls = 5 if rid.item() == 0 else 6
                    pred = logits[i, :n_cls].argmax().cpu().item()
                    all_preds.append(pred)
                    all_labels.append(batch["y"][i].cpu().item())
        
        avg_loss = total_loss / len(dataloader)
        oa, kappa, f1 = self._compute_metrics(all_labels, all_preds)
        return avg_loss, oa, kappa, f1
    
    def fit(self, train_loader, val_loader, model_name: str = "mctnet"):
        best_path = None
        
        for epoch in range(1, self.epochs + 1):
            start = time.time()
            train_loss, _, _, _ = self._run_epoch(train_loader, is_train=True)
            val_loss, val_oa, val_kappa, val_f1 = self._run_epoch(val_loader, is_train=False)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_oa"].append(val_oa)
            self.history["val_kappa"].append(val_kappa)
            self.history["val_f1"].append(val_f1)
            
            print(f"Epoch {epoch}/{self.epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val OA: {val_oa:.4f} | Kappa: {val_kappa:.4f} | F1: {val_f1:.4f} | "
                  f"Time: {time.time()-start:.1f}s")
            
            # LR scheduling on OA
            self.scheduler.step(val_oa)
            
            # Early stopping on OA
            if val_oa > self.best_metric:
                self.best_metric = val_oa
                self.patience_counter = 0
                best_path = os.path.join(self.checkpoint_dir, f"{model_name}_best.pt")
                torch.save(self.model.state_dict(), best_path)
                print(f"  -> Saved best model (OA: {val_oa:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break
        
        return self.history, best_path