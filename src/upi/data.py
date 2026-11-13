from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import LABELS, RAW_TRANSACTION_COLUMNS


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_inputs(raw_dir: Path, sample_fraction: float | None = None, seed: int = 20261113):
    parse = ["timestamp"]
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=parse)
    test = pd.read_csv(raw_dir / "test.csv", parse_dates=parse)
    accounts = pd.read_csv(raw_dir / "accounts.csv", parse_dates=["opened_on"])
    merchants = pd.read_csv(raw_dir / "merchants.csv", parse_dates=["onboarded_on"])
    sample = pd.read_csv(raw_dir / "sample_submission.csv")
    train["split"], test["split"] = "train", "test"
    if sample_fraction and sample_fraction < 1:
        senders = pd.concat([train.sender_id, test.sender_id]).drop_duplicates()
        chosen = set(senders.sample(frac=sample_fraction, random_state=seed))
        train = train[train.sender_id.isin(chosen)].copy()
        test = test[test.sender_id.isin(chosen)].copy()
        sample = sample[sample.id.isin(set(test.txn_id))].copy()
    return train, test, accounts, merchants, sample


def audit_inputs(raw_dir: Path, output: Path | None = None) -> dict[str, Any]:
    train, test, accounts, merchants, sample = read_inputs(raw_dir)
    # read_inputs adds only this provenance column after reading the raw schema.
    expected_train = set(RAW_TRANSACTION_COLUMNS + ["label", "split"])
    expected_test = set(RAW_TRANSACTION_COLUMNS + ["split"])
    legal = set(LABELS)
    account_ids = set(accounts.account_id)
    merchant_ids = set(merchants.merchant_id)
    p2p = test.receiver_type.eq("P2P")
    result: dict[str, Any] = {
        "files": {name: {"bytes": (raw_dir / name).stat().st_size,
                         "sha256": sha256_file(raw_dir / name)}
                  for name in ["train.csv", "test.csv", "accounts.csv", "merchants.csv", "sample_submission.csv"]},
        "rows": {"train": len(train), "test": len(test), "accounts": len(accounts),
                 "merchants": len(merchants), "sample_submission": len(sample)},
        "schema": {"train_missing": sorted(expected_train - set(train.columns)),
                   "train_extra": sorted(set(train.columns) - expected_train),
                   "test_missing": sorted(expected_test - set(test.columns)),
                   "test_extra": sorted(set(test.columns) - expected_test)},
        "time": {"train_min": str(train.timestamp.min()), "train_max": str(train.timestamp.max()),
                 "test_min": str(test.timestamp.min()), "test_max": str(test.timestamp.max())},
        "duplicates": {"train_txn_id": int(train.txn_id.duplicated().sum()),
                       "test_txn_id": int(test.txn_id.duplicated().sum()),
                       "cross_split_ids": len(set(train.txn_id) & set(test.txn_id)),
                       "account_id": int(accounts.account_id.duplicated().sum()),
                       "merchant_id": int(merchants.merchant_id.duplicated().sum())},
        "labels": {str(k): int(v) for k, v in train.label.value_counts().items()},
        "illegal_labels": sorted(set(train.label) - legal),
        "joins": {"missing_train_senders": int((~train.sender_id.isin(account_ids)).sum()),
                  "missing_test_senders": int((~test.sender_id.isin(account_ids)).sum()),
                  "missing_test_p2p_receivers": int((p2p & ~test.receiver_id.isin(account_ids)).sum()),
                  "missing_test_merchant_receivers": int((~p2p & ~test.receiver_id.isin(merchant_ids)).sum())},
        "submission": {"id_set_matches": set(sample.id) == set(test.txn_id),
                       "id_order_matches": sample.id.tolist() == test.txn_id.tolist(),
                       "duplicate_ids": int(sample.id.duplicated().sum())},
        "tails": {"train_on_or_after_october": int((train.timestamp >= "2026-10-01").sum()),
                  "test_on_or_after_november": int((test.timestamp >= "2026-11-01").sum())},
    }
    failures = [
        result["schema"]["train_missing"], result["schema"]["test_missing"],
        result["illegal_labels"], any(result["duplicates"].values()),
        any(result["joins"].values()), not result["submission"]["id_set_matches"],
    ]
    result["passed"] = not any(failures)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result

