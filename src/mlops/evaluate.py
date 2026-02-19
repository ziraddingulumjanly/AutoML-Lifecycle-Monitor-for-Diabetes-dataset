from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def pick_metric(task: str, y_true, y_pred, y_proba=None, explicit: str = "auto") -> Tuple[str, float]:
    if explicit != "auto":
        # compute supported
        if explicit == "roc_auc":
            return "roc_auc", roc_auc_score(y_true, y_proba)
        if explicit == "f1":
            return "f1", f1_score(y_true, y_pred, average="weighted")
        if explicit == "rmse":
            return "rmse", float(np.sqrt(mean_squared_error(y_true, y_pred)))
        raise ValueError(f"Unsupported explicit metric: {explicit}")

    if task == "classification":
        # prefer roc_auc if binary + probabilities available
        uniq = np.unique(y_true)
        if len(uniq) == 2 and y_proba is not None:
            return "roc_auc", roc_auc_score(y_true, y_proba)
        return "f1", f1_score(y_true, y_pred, average="weighted")

    # regression
    return "rmse", float(np.sqrt(mean_squared_error(y_true, y_pred)))

def full_report(task: str, y_true, y_pred, y_proba=None) -> Dict:
    if task == "classification":
        report = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        }
        uniq = np.unique(y_true)
        if len(uniq) == 2 and y_proba is not None:
            report["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        return report

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
