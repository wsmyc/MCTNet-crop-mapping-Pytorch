import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix
from config import REGION_CONFIG

def compute_metrics(y_true, y_pred, class_names=None):
    oa = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    result = {
        "OA": float(oa),
        "Kappa": float(kappa),
        "F1_macro": float(f1),
    }
    
    if class_names is not None:
        # labels=range(len(class_names)) ensures the matrix includes the 'Others' class
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        result["confusion_matrix"] = cm
    
    return result

def evaluate_by_region(model, dataloader, device, state_name=None):
    """
    Standard evaluation loop. 
    Handles multi-task head slicing and produces metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move data to GPU
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)
            
            # Forward pass
            outputs = model(batch)
            logits = outputs["logits"]
            
            # Identify the state for this batch to slice the head correctly
            # If state_name is provided (baseline), use that. Otherwise, check region_id.
            if state_name is None:
                current_state = "arkansas" if batch["region_id"][0].item() == 0 else "california"
            else:
                current_state = state_name
            
            n_cls = REGION_CONFIG[current_state]["num_classes"]
            
            # Prediction
            preds = logits[:, :n_cls].argmax(dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["y"].cpu().numpy())
    
    # Get class list for confusion matrix
    if state_name:
        class_list = list(REGION_CONFIG[state_name]["classes"].values())
        # Append "Others" if the count matches
        if len(class_list) < REGION_CONFIG[state_name]["num_classes"]:
            class_list.append("Others")
    else:
        class_list = None

    metrics = compute_metrics(all_labels, all_preds, class_list)
    
    # Print results nicely
    print(f"\nResults for {state_name.upper() if state_name else 'Dataset'}:")
    print(f"  OA:    {metrics['OA']:.4f}")
    print(f"  Kappa: {metrics['Kappa']:.4f}")
    print(f"  F1:    {metrics['F1_macro']:.4f}")
    
    return metrics