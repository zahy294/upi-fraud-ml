"""Hash-addressed features and portable, keyed prediction artifacts."""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import FEATURES, LABELS
from .data import require_unique
from .features import build_features


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_hashes():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def data_hashes(data_dir):
    return {name: sha256(Path(data_dir) / name) for name in
            ["train.csv", "test.csv", "accounts.csv", "merchants.csv", "sample_submission.csv"]}


def environment():
    return {"python": platform.python_version(), "packages": {
        name: importlib.metadata.version(name) for name in
        ["numpy", "pandas", "lightgbm", "pyarrow", "scikit-learn"]}}


def git_revision():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                   cwd=Path(__file__).resolve().parents[2]).strip()


def cached_features(transactions, accounts, merchants, end, cache_dir, hashes, schema=None):
    spec = {"data_hashes": hashes, "context_end_exclusive": end,
            "source_hashes": source_hashes(), "features": list(FEATURES),
            "category_schema": schema}
    key = digest(spec)
    base = Path(cache_dir) / key
    base.mkdir(parents=True, exist_ok=True)
    meta_path, path = base / "metadata.json", base / "features.parquet"
    if meta_path.exists() and path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata["spec"] == spec and sha256(path) == metadata["sha256"]:
            print(f"Using verified feature cache {key[:12]}", flush=True)
            frame = pd.read_parquet(path)
            require_unique(frame, "txn_id")
            return frame, metadata
    context = transactions if end is None else transactions.loc[transactions.timestamp < pd.Timestamp(end)]
    print(f"Building {len(context):,} rows of features; context end {end or 'all supplied rows'}", flush=True)
    frame, categories = build_features(context, accounts, merchants, category_schema=schema)
    frame.to_parquet(path, index=False)
    metadata = {"spec": spec, "cache_key": key, "path": str(path.resolve()),
        "sha256": sha256(path), "rows": len(frame), "category_schema": categories,
        "context_min_timestamp": str(frame.timestamp.min()),
        "context_max_timestamp": str(frame.timestamp.max())}
    write_json(meta_path, metadata)
    return frame, metadata


def save_probabilities(path, metadata, P, rule_codes, prediction_codes, y=None):
    frame = metadata[["txn_id", "timestamp", "sender_id", "receiver_id", "device_id"]].reset_index(drop=True).copy()
    require_unique(frame, "txn_id")
    if P.shape != (len(frame), 12):
        raise ValueError("Probability rows differ from keyed metadata")
    for c, name in enumerate(LABELS):
        frame[f"prob_{name}"] = P[:, c]
    frame["rule_label"] = np.asarray(LABELS)[rule_codes]
    for variant, codes in prediction_codes.items():
        frame[f"prediction_{variant}"] = np.asarray(LABELS)[codes]
    if y is not None:
        frame["true_label"] = np.asarray(LABELS)[y]
    frame.to_parquet(path, index=False)
    write_json(Path(path).with_suffix(".schema.json"), {
        "label_order": list(LABELS), "probability_columns": [f"prob_{c}" for c in LABELS],
        "rows": len(frame), "sha256": sha256(path), "key": "txn_id"})
    return frame
