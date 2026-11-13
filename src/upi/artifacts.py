from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import LABELS


def environment_manifest() -> dict:
    packages = {}
    try:
        import importlib.metadata as metadata
        for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "lightgbm", "xgboost", "catboost", "pyyaml"):
            try:
                packages[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                packages[name] = None
    except Exception:
        pass
    return {"python": sys.version, "platform": platform.platform(), "packages": packages}


def git_revision(root: Path) -> dict:
    def command(*args):
        result = subprocess.run(["git", *args], cwd=root, check=False, text=True,
                                capture_output=True, encoding="utf-8")
        return result.stdout.strip() if result.returncode == 0 else None
    return {"commit": command("rev-parse", "HEAD"), "branch": command("branch", "--show-current"),
            "dirty": bool(command("status", "--porcelain"))}


def validate_submission(frame: pd.DataFrame, sample: pd.DataFrame) -> None:
    if frame.columns.tolist() != ["id", "label"]:
        raise ValueError("submission columns must be exactly id,label")
    if frame.id.duplicated().any() or sample.id.duplicated().any():
        raise ValueError("duplicate submission IDs")
    if frame.id.tolist() != sample.id.tolist():
        raise ValueError("submission IDs/order differ from sample_submission.csv")
    illegal = sorted(set(frame.label) - set(LABELS))
    if illegal:
        raise ValueError(f"illegal labels: {illegal}")


def write_submission(path: Path, ids: np.ndarray, prediction_ids: np.ndarray, sample: pd.DataFrame) -> dict:
    raw = pd.DataFrame({"id": ids, "label": np.asarray(LABELS)[prediction_ids]})
    indexed = raw.set_index("id")
    ordered = pd.DataFrame({"id": sample.id, "label": sample.id.map(indexed.label)})
    if ordered.label.isna().any():
        raise ValueError("missing predictions for sample IDs")
    validate_submission(ordered, sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path), "rows": len(ordered), "bytes": path.stat().st_size,
            "sha256": digest, "predicted_counts": ordered.label.value_counts().to_dict()}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

