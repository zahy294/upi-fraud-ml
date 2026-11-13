from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config["_config_path"] = str(path)
    config["_config_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config


def resolve_path(config: dict[str, Any], key: str, default: str) -> Path:
    value = config.get("paths", {}).get(key, default)
    path = Path(value)
    if not path.is_absolute():
        root = Path(config["_config_path"]).parents[1]
        path = root / path
    return path.resolve()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

