from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .constants import LABELS


def macro_f1_ids(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 12) -> float:
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes*n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.mean(np.divide(2*tp, denominator, out=np.zeros_like(tp), where=denominator > 0)))


def apply_weights(probabilities: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (probabilities * weights.reshape(1, -1)).argmax(axis=1)


def fit_class_weights(probabilities: np.ndarray, y_true: np.ndarray,
                      grid: np.ndarray | None = None, sweeps: int = 4) -> np.ndarray:
    if grid is None:
        grid = np.geomspace(0.2, 40.0, 35)
    weights = np.ones(probabilities.shape[1], dtype=np.float64)
    for _ in range(sweeps):
        for class_id in range(1, probabilities.shape[1]):
            scores = []
            for candidate in grid:
                trial = weights.copy()
                trial[class_id] = candidate
                scores.append(macro_f1_ids(y_true, apply_weights(probabilities, trial), probabilities.shape[1]))
            scores = np.asarray(scores)
            plateau = np.flatnonzero(scores >= scores.max() - 5e-4)
            weights[class_id] = grid[plateau[len(plateau)//2]]
    return weights


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(LABELS)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(LABELS)))
    per_class = {
        label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                "support": int(s[i]), "predicted": int((y_pred == i).sum())}
        for i, label in enumerate(LABELS)
    }
    return {"macro_f1": float(f.mean()), "accuracy": float((y_true == y_pred).mean()),
            "per_class": per_class, "confusion_matrix": cm.tolist()}


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

