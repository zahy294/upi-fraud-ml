"""Measure predeclared motif coverage on July/August labels, without training."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.upi.motifs import build_motifs
from src.upi.fit_submit import file_hash


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, default=Path("runs/motif_audit.json"))
    a = ap.parse_args(argv)
    root = a.root.resolve()
    output = a.output if a.output.is_absolute() else root / a.output
    if output.exists():
        raise FileExistsError("choose a fresh audit output path")
    base_path = root / "data/processed/base_v0.parquet"
    columns = ["txn_id", "timestamp", "split", "sender_id", "receiver_id", "receiver_type",
               "device_id", "txn_type", "amount", "balance_before", "label"]
    base = pd.read_parquet(base_path, columns=columns)
    features = build_motifs(base)
    date = pd.to_datetime(base.timestamp)
    eligible = base.split.eq("train") & date.dt.month.isin([7, 8])
    f1_path = root / "data/processed/features_v1.parquet"
    f1 = pd.read_parquet(f1_path, columns=["txn_id", "d_snd_prevbal_q", "d_snd_nextbal_q"])
    f1 = f1.set_index("txn_id").loc[base.txn_id].reset_index()
    existing = f1[["d_snd_prevbal_q", "d_snd_nextbal_q"]].fillna(np.inf).min(axis=1) < .002
    probes = {
        "skip_drain_quarter": features.motif_quarter_skip_residual < .002,
        "skip_drain_third": features.motif_third_skip_residual < .002,
        "regular_device_burst": (features.motif_device_count >= 6) & (features.motif_device_gap_cv <= .25),
        "repeat_pair_amount": (features.motif_pair_week_count >= 3) & (features.motif_pair_amount_cv <= .01),
    }
    results = {}
    for name, mask in probes.items():
        results[name] = {}
        for month in (7, 8):
            selected = eligible & date.dt.month.eq(month) & mask
            labels = base.loc[selected, "label"]
            extra = selected & ~existing
            results[name][str(month)] = {"rows": int(selected.sum()),
                "label_counts": labels.value_counts().to_dict(),
                "fraud_label_share": float(labels.ne("legit").mean()) if len(labels) else None,
                "outside_nearest_drain_rows": int(extra.sum()),
                "outside_nearest_drain_label_counts": base.loc[extra, "label"].value_counts().to_dict()}
    payload = {"label_windows": "July and August only", "training_performed": False,
        "note": "Raw noisy-label associations, not clean-test precision or a model score. Fixed thresholds were declared before inspection.",
        "results": results, "hashes": {"base": file_hash(base_path), "v1": file_hash(f1_path),
        "motifs.py": file_hash(root / "src/upi/motifs.py"), "audit_motifs.py": file_hash(Path(__file__))}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
