"""
Evaluation metrics: OA, Kappa, Macro F1.
Matches paper equations (4)-(6) exactly.
"""

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix


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
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
        result["confusion_matrix"] = cm
    
    return result


def evaluate_by_region(all_preds, all_labels, all_region_ids, ar_classes, ca_classes):
    """
    Compute metrics separately for Arkansas and California.
    """
    results = {}
    
    for region_name, rid_val, classes in [
        ("arkansas", 0, ar_classes),
        ("california", 1, ca_classes),
    ]:
        mask = np.array(all_region_ids) == rid_val
        if mask.sum() == 0:
            continue
        
        y_true = np.array(all_labels)[mask]
        y_pred = np.array(all_preds)[mask]
        
        results[region_name] = compute_metrics(y_true, y_pred, classes)
    
    return results