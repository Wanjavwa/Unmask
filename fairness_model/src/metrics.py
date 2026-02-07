"""
Metrics for binary classification: accuracy, precision, recall, F1, confusion matrix, FPR, FNR.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_metrics(y_true, y_pred, zero_division=0):
    """
    Compute accuracy, precision, recall, F1, confusion matrix, FPR, FNR.
    y_true, y_pred: list or array of integer labels (0=real, 1=fake).
    Returns dict with all metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "confusion_matrix": np.zeros((2, 2), dtype=np.int64),
            "false_positive_rate": 0.0, "false_negative_rate": 0.0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        }
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # Ensure 2x2: rows true, cols pred [[TN, FP], [FN, TP]]
    if cm.shape != (2, 2):
        cm_full = np.zeros((2, 2), dtype=np.int64)
        for i in range(min(2, cm.shape[0])):
            for j in range(min(2, cm.shape[1])):
                cm_full[i, j] = cm[i, j]
        cm = cm_full
    tn, fp, fn, tp = cm.ravel()
    # Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=zero_division, average="binary", pos_label=1)),
        "recall": float(recall_score(y_true, y_pred, zero_division=zero_division, average="binary", pos_label=1)),
        "f1": float(f1_score(y_true, y_pred, zero_division=zero_division, average="binary", pos_label=1)),
        "confusion_matrix": cm,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def format_metrics(metrics):
    """Format metrics dict into a readable string for logging."""
    lines = [
        f"Accuracy:           {metrics['accuracy']:.4f}",
        f"Precision:          {metrics['precision']:.4f}",
        f"Recall:             {metrics['recall']:.4f}",
        f"F1-score:           {metrics['f1']:.4f}",
        f"False positive rate: {metrics['false_positive_rate']:.4f}",
        f"False negative rate: {metrics['false_negative_rate']:.4f}",
        "Confusion matrix (rows=true, cols=pred) [real, fake]:",
        f"  {metrics['confusion_matrix'].tolist()}",
    ]
    return "\n".join(lines)
