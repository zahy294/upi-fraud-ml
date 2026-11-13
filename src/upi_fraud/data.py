"""Typed loading, keyed joins and submission integrity checks."""
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import LABELS


def require_unique(frame, key):
    if frame[key].isna().any() or not frame[key].is_unique:
        raise ValueError(f"{key} must be nonmissing and unique")


def align(frame, ids, key="txn_id"):
    """Preserve caller order and reject both duplicate and missing query keys."""
    require_unique(frame, key)
    ids = pd.Index(ids, name=key)
    if not ids.is_unique or ids.hasnans:
        raise ValueError("Query IDs must be unique and nonmissing")
    if not ids.isin(frame[key]).all():
        raise ValueError("Query contains IDs missing from the feature/artifact frame")
    return frame.set_index(key).loc[ids].reset_index()


def load_data(data_dir):
    data_dir = Path(data_dir)
    frames = []
    for split in ("train", "test"):
        f = pd.read_csv(data_dir / f"{split}.csv", dtype={
            "app_version": "str", "merchant_category": "str", "qr_vpa_mismatch": "float32",
        }, parse_dates=["timestamp"])
        require_unique(f, "txn_id")
        if f.timestamp.isna().any():
            raise ValueError("Missing transaction timestamp")
        if split == "train":
            if f.label.isna().any() or not f.label.isin(LABELS).all():
                raise ValueError("Training labels must belong to the official vocabulary")
        else:
            if "label" in f:
                raise ValueError("Test data must be unlabelled")
            f["label"] = None
        f["split"] = split
        # This immutable loader ordinal reproduces stable timestamp ties in the
        # original EDA. Shuffling a loaded frame must preserve this metadata.
        f["_source_order"] = np.arange(len(f), dtype="int64") + sum(len(x) for x in frames)
        frames.append(f)
    transactions = pd.concat(frames, ignore_index=True)
    require_unique(transactions, "txn_id")
    accounts = pd.read_csv(data_dir / "accounts.csv")
    merchants = pd.read_csv(data_dir / "merchants.csv")
    require_unique(accounts, "account_id")
    require_unique(merchants, "merchant_id")
    p2p = transactions.receiver_type == "P2P"
    if not transactions.receiver_type.isin(["P2P", "MERCHANT"]).all():
        raise ValueError("Unknown receiver type")
    if not transactions.sender_id.isin(accounts.account_id).all():
        raise ValueError("Sender missing from accounts")
    if not transactions.loc[p2p, "receiver_id"].isin(accounts.account_id).all():
        raise ValueError("P2P recipient missing from accounts")
    if not transactions.loc[~p2p, "receiver_id"].isin(merchants.merchant_id).all():
        raise ValueError("Merchant recipient missing from merchants")
    if not transactions.loc[p2p, "receiver_id"].str.startswith("AC").all():
        raise ValueError("P2P ID prefix mismatch")
    if not transactions.loc[~p2p, "receiver_id"].str.startswith("MR").all():
        raise ValueError("Merchant ID prefix mismatch")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    check_submission(sample, sample)
    test_ids = transactions.loc[transactions.split == "test", "txn_id"]
    if set(sample.id) != set(test_ids):
        raise ValueError("Sample IDs differ from test IDs")
    return transactions, accounts, merchants, sample


def check_submission(submission, sample):
    if list(submission) != ["id", "label"]:
        raise ValueError("Submission columns must be exactly id,label")
    require_unique(submission, "id")
    if len(submission) != len(sample) or set(submission.id) != set(sample.id):
        raise ValueError("Submission must contain exactly the sample IDs")
    if not np.array_equal(submission.id.to_numpy(), sample.id.to_numpy()):
        raise ValueError("Submission must preserve sample order")
    if submission.label.isna().any() or not submission.label.isin(LABELS).all():
        raise ValueError("Invalid or missing submission label")
    return {"rows": len(submission), "label_counts": submission.label.value_counts().to_dict()}
