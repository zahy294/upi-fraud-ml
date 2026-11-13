"""Zero-training diagnostics and candidate submissions from saved probabilities.

No training imports/calls. Outputs are experiments, with no claimed test score.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src/upi"))
from common import LABELS
from make_submission import (predict_labels, prepare_rule_context, write_submission,
                             _proba, _decision_scores, _apply_rules)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def align_ids(ids, expected):
    ids, expected = pd.Index(np.asarray(ids).astype(str)), pd.Index(np.asarray(expected).astype(str))
    if (ids.has_duplicates or expected.has_duplicates or len(ids) != len(expected)
            or not expected.isin(ids).all()):
        raise ValueError("unique, exactly matching transaction ID sets required")
    return ids.get_indexer(expected)


def load_proba(path, expected):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as saved:
            ids, p, classes = saved["ids"], saved["proba"], saved["classes"]
        if pd.isna(ids).any():
            raise ValueError("missing probability IDs")
        if classes.dtype.kind in "iu":
            if not np.array_equal(np.sort(classes), np.arange(12)):
                raise ValueError("probability classes must contain all 12 classes")
            order = np.argsort(classes)
        else:
            if len(classes) != 12 or set(classes.astype(str)) != set(LABELS):
                raise ValueError("probability class names must contain all 12 classes")
            order = [list(classes.astype(str)).index(k) for k in LABELS]
        p = _proba(p)
        if len(ids) != len(p):
            raise ValueError("probability ID/row mismatch")
        return p[align_ids(ids, expected)][:, order]
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, dtype={"id": str})
        if set(frame.columns) != {"id", *LABELS} or frame.id.isna().any():
            raise ValueError("probability CSV requires id plus 12 named class columns; hard labels cannot be blended")
        return _proba(frame.loc[:, LABELS].to_numpy())[align_ids(frame.id, expected)]
    raise ValueError("use an ID-labelled NPZ or wide probability CSV; bare arrays are unsafe")


def read_submission(path, expected):
    frame = pd.read_csv(path, dtype={"id": str})
    if (set(frame.columns) != {"id", "label"} or frame.isna().any().any()
            or not frame.label.isin(LABELS).all()):
        raise ValueError("submission requires id,label with valid 12-class labels")
    return frame.iloc[align_ids(frame.id, expected)].reset_index(drop=True)


def counts(frame):
    n = int(frame.label.ne("legit").sum())
    return {"rows": len(frame), "frauds": n, "fraud_rate": n / len(frame) if len(frame) else 0.,
            "per_class": {k: int(frame.label.eq(k).sum()) for k in LABELS}}


def compare_labels(current, baseline):
    """Both frames must already be ID-aligned; neither provides ground truth."""
    if not np.array_equal(current.id, baseline.id):
        raise ValueError("comparison IDs must be aligned")
    a, b = current.label, baseline.label
    changed = a.ne(b)
    transitions = pd.DataFrame({"current": a[changed], "baseline": b[changed]}).value_counts()
    return {"changed_labels": int(changed.sum()),
            "changed_fraud_status": int(a.ne("legit").ne(b.ne("legit")).sum()),
            "baseline_only_fraud": int((a.eq("legit") & b.ne("legit")).sum()),
            "current_only_fraud": int((a.ne("legit") & b.eq("legit")).sum()),
            "changed_fraud_type": int((changed & a.ne("legit") & b.ne("legit")).sum()),
            "transitions": [{"current": x, "baseline": y, "rows": int(n)}
                            for (x, y), n in transitions.items()]}


def exact_count_prediction(p, groups, offsets, context, k):
    """Rank calibrated best-fraud/legit log odds. Preserve direct bands.

    This is a user-selected test-count hypothesis, not supervised calibration.
    Ties are resolved by transaction ID, independent of input order.
    """
    p = _proba(p)
    groups = np.asarray(groups)
    if groups.shape != (len(p),) or len(context["band"]) != len(p):
        raise ValueError("unaligned probability/group/context rows")
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= len(p):
        raise ValueError("top-k must be an integer from zero to the test row count")
    scores = _decision_scores(np.log(np.maximum(p, 1e-9)), groups, offsets)
    forced = np.asarray(context["band"]) > 0
    if k < int(forced.sum()):
        raise ValueError("requested fraud count is below protected direct-band count")
    margin = scores[:, 1:].max(axis=1) - scores[:, 0]
    free = np.flatnonzero(~forced)
    ranked = free[np.lexsort((np.asarray(context["txn_id"])[free].astype(str), -margin[free]))]
    selected = forced.copy()
    selected[ranked[:k - int(forced.sum())]] = True
    pred = np.where(selected, scores[:, 1:].argmax(axis=1) + 1, 0)
    pred = _apply_rules(pred, context)
    if int((pred > 0).sum()) != k:
        raise ValueError("rules changed the requested fraud count")
    return pred


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile-dir", type=Path, required=True, help="e.g. runs/local_podium/ensemble")
    ap.add_argument("--baseline-csv", type=Path, help="exact friend's submission, for comparison/count matching")
    ap.add_argument("--friend-proba", type=Path, help="NPZ ids/proba/classes or CSV id plus 12 class probabilities")
    ap.add_argument("--friend-weight", type=float, default=.65)
    ap.add_argument("--fraud-log-offset", type=float, default=0., help="additional shift for all fraud classes")
    ap.add_argument("--zero-offsets", action="store_true", help="use uncalibrated blended probabilities plus event rules")
    ap.add_argument("--use-baseline-types", action="store_true",
                    help="use baseline's fraud type where both predict fraud, preserving direct bands; this is a hard-label hybrid, not a probability blend")
    count = ap.add_mutually_exclusive_group()
    count.add_argument("--top-k", type=int)
    count.add_argument("--match-baseline-count", action="store_true")
    ap.add_argument("--output", type=Path, help="omit for diagnostics only; must be a new file")
    a = ap.parse_args(argv)
    if not np.isfinite(a.friend_weight) or not 0 <= a.friend_weight <= 1 or not np.isfinite(a.fraud_log_offset):
        ap.error("weights must be in [0,1] and offsets finite")
    if a.match_baseline_count and a.baseline_csv is None:
        ap.error("--match-baseline-count requires --baseline-csv")
    if a.use_baseline_types and a.baseline_csv is None:
        ap.error("--use-baseline-types requires --baseline-csv")
    if a.output and a.output.suffix.lower() != ".csv":
        ap.error("--output must be a new .csv file")
    refit = a.profile_dir / "refit"
    current_path = refit / "submission.csv"
    current = pd.read_csv(current_path, dtype={"id": str})
    current = read_submission(current_path, current.id)
    ids = current.id.to_numpy()
    report = {"interpretation": "diagnostics only; test labels and candidate F1 are unknown",
              "current": counts(current), "inputs": {str(current_path): digest(current_path)}}
    baseline = None
    if a.baseline_csv:
        baseline = read_submission(a.baseline_csv, ids)
        report["baseline"] = counts(baseline)
        report.update(compare_labels(current, baseline))
        report["inputs"][str(a.baseline_csv)] = digest(a.baseline_csv)
    if a.output:
        sidecar = a.output.with_suffix(".json")
        if a.output.exists() or sidecar.exists():
            raise FileExistsError("candidate CSV/report exists; choose a new output name")
        meta = pd.read_parquet(refit / "meta.parquet")
        meta = meta.iloc[align_ids(meta.txn_id, ids)].reset_index(drop=True)
        f3 = pd.read_parquet(refit / "features_v3.parquet")
        f5 = pd.read_parquet(refit / "features_v5.parquet")
        context = prepare_rule_context(meta, f3, f5)
        # groups.npy follows meta.parquet's original order, not submission order.
        original_meta = pd.read_parquet(refit / "meta.parquet", columns=["txn_id"])
        groups = np.load(refit / "groups.npy", allow_pickle=False)[align_ids(original_meta.txn_id, ids)]
        own_path = refit / "test_proba.npz"
        p = load_proba(own_path, ids)
        offset_path = a.profile_dir / "validation/offsets.json"
        offsets = json.loads(offset_path.read_text(encoding="utf-8"))
        if a.zero_offsets:
            offsets = {"global": [0.] * 12, "conditional": {}, "class_order": list(LABELS)}
        if a.friend_proba:
            p = a.friend_weight * load_proba(a.friend_proba, ids) + (1 - a.friend_weight) * p
            report["inputs"][str(a.friend_proba)] = digest(a.friend_proba)
        offsets["global"] = [0.] + [v + a.fraud_log_offset for v in offsets["global"][1:]]
        k = report["baseline"]["frauds"] if a.match_baseline_count else a.top_k
        pred = (exact_count_prediction(p, groups, offsets, context, k) if k is not None
                else predict_labels(p, groups, offsets, context))
        type_changes = 0
        if a.use_baseline_types:
            baseline_y = baseline.label.map({label: i for i, label in enumerate(LABELS)}).to_numpy()
            use = (pred > 0) & (baseline_y > 0) & (context["band"] == 0)
            type_changes = int((pred[use] != baseline_y[use]).sum())
            pred[use] = baseline_y[use]
        result = write_submission(ids, pred, current[["id"]], a.output)
        report["candidate"] = result
        if baseline is not None:
            candidate_frame = pd.DataFrame({"id": ids, "label": np.asarray(LABELS)[pred]})
            report["candidate_vs_baseline"] = compare_labels(candidate_frame, baseline)
        report["settings"] = {"friend_weight": a.friend_weight if a.friend_proba else None,
                              "fraud_log_offset": a.fraud_log_offset, "top_k": k,
                              "use_baseline_types": a.use_baseline_types, "baseline_type_changes": type_changes,
                              "offsets": offsets, "direct_bands_protected": True}
        for path in (own_path, offset_path, refit / "meta.parquet", refit / "features_v3.parquet",
                     refit / "features_v5.parquet", refit / "groups.npy"):
            report["inputs"][str(path)] = digest(path)
        report["code_sha256"] = {str(path): digest(path) for path in
            (Path(__file__), Path(__file__).parent / "src/upi/make_submission.py")}
        sidecar.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in {"inputs", "settings", "code_sha256"}}
    if a.output:
        summary["output"] = str(a.output)
        summary["report"] = str(sidecar)
        summary["settings"] = {k: v for k, v in report["settings"].items() if k != "offsets"}
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
