"""Fold-scoped ordered device bands, preserving sibling-event scoring."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .common import LABELS
    from .events import Events, HOUR, DAY
    from .features_v6 import align_pack, fit_mask, validate_probabilities, entity_codes
except ImportError:
    from common import LABELS
    from events import Events, HOUR, DAY
    from features_v6 import align_pack, fit_mask, validate_probabilities, entity_codes

N_ACCOUNTS = 420000
BOT_POOL = (420001, 423000)
ORDER = list(range(1, 12))
K = len(ORDER)


def legit_block_start(nums, lo, run=40):
    """First observed run of consecutive IDs; raise if no block is supported."""
    nums = np.unique(np.asarray(nums, dtype=np.int64))
    nums = nums[nums >= lo]
    if run < 1 or len(nums) < run:
        raise ValueError("missing contiguous legitimate device block")
    starts = np.r_[0, np.flatnonzero(np.diff(nums) != 1) + 1]
    ends = np.r_[starts[1:], len(nums)]
    valid = starts[(ends - starts) >= run]
    if not len(valid):
        raise ValueError("missing contiguous legitimate device block")
    return int(nums[valid[0]])


def segment(score, prior=None, sd=None):
    score = np.asarray(score, dtype=float)
    if score.ndim != 2 or score.shape[1] != K or not len(score) or not np.isfinite(score).all():
        raise ValueError("score must be a nonempty finite (width,11) matrix")
    if prior is not None:
        prior = np.asarray(prior, dtype=float)
        if prior.shape != (K - 1,) or not np.isfinite(prior).all() or sd is None or not np.isfinite(sd) or sd <= 0:
            raise ValueError("invalid boundary prior")
    n = len(score)
    cs = np.vstack([np.zeros(K), np.cumsum(score, axis=0)])
    pos = np.arange(n + 1)
    best, back = cs[:, 0].copy(), []
    for k in range(1, K):
        penalty = -((pos - prior[k - 1]) ** 2) / (2 * sd ** 2) if prior is not None else np.zeros(n + 1)
        cand = best - cs[:, k] + penalty
        idx, maximum = np.zeros(n + 1, dtype=int), np.empty(n + 1)
        current = 0
        for j in range(n + 1):
            if cand[j] > cand[current]:
                current = j
            idx[j], maximum[j] = current, cand[current]
        best = maximum + cs[:, k]
        back.append(idx)
    cuts = [n]
    for idx in reversed(back):
        cuts.append(int(idx[cuts[-1]]))
    return cuts[::-1][:-1]


def band_scores(probs, kind, dnum, start, width, lam_mule, lam_sim):
    probs, kind, dnum = np.asarray(probs, float), np.asarray(kind), np.asarray(dnum)
    if width <= 0 or not len(probs):
        raise ValueError("absent teacher support or invalid band width")
    validate_probabilities(probs, len(dnum))
    if not np.isfinite(probs).all() or len(kind) != len(dnum) or ((dnum < start) | (dnum >= start + width)).any():
        raise ValueError("invalid device scores/support")
    Q = probs[:, 1:].copy()
    mule = Q[:, 0].copy()
    drain = Q[:, 3] + Q[:, 4]
    sim_shaped = kind == 1
    Q[:, 1] += 0.1 * mule
    Q[:, 3] = drain
    Q[:, 4] = np.where(sim_shaped, 0.1 * drain, drain)
    sim = np.where(sim_shaped, drain, 0.)
    score = np.zeros((width, K))
    np.add.at(score, dnum - start, np.log(np.maximum(Q, 0.05)))
    for col, anchor, lam in ((0, mule, lam_mule), (3, sim, lam_sim)):
        if not np.isfinite(lam):
            raise ValueError("nonfinite anchor density")
        has = np.zeros(width, bool)
        has[(dnum - start)[anchor >= 0.5]] = True
        score[~has, col] += np.log1p(-np.clip(lam, 1e-6, 1 - 1e-6))
    return score


def _density(dn, anchor, support, lo, hi):
    if hi <= lo:
        raise ValueError("zero-width anchor band; training-fold edge support is insufficient")
    inside = support & (dn >= lo) & (dn < hi)
    if not inside.any():
        raise ValueError("absent teacher support inside anchor band")
    count = len(np.unique(dn[inside & anchor]))
    return float(np.clip(count / (hi - lo), 1e-6, 1 - 1e-6)), count


def build_bands(base: pd.DataFrame, teacher_proba: np.ndarray,
                f1: pd.DataFrame, fit_ids: pd.Index) -> tuple[pd.DataFrame, dict]:
    fit = fit_mask(base, fit_ids)
    if not len(base) or not fit.any():
        raise ValueError("band fitting requires nonempty base and training fold")
    p, available = validate_probabilities(teacher_proba, len(base))
    kind = align_pack(base, f1).snd_drain_kind.to_numpy(dtype=int)
    suffix = base.device_id.astype("string").str.extract(r"^[^0-9]+([0-9]+)$", expand=False)
    if suffix.isna().any():
        raise ValueError("device IDs must contain a prefix and integer suffix")
    dn = pd.to_numeric(suffix, errors="raise").to_numpy(dtype=np.int64)
    if (dn < 0).any() or not base.split.isin(["train", "test"]).all():
        raise ValueError("invalid device IDs or split")
    train = base.split.eq("train").to_numpy()
    start = BOT_POOL[1] + 1
    end = legit_block_start(dn[train & (dn >= start)], start) - 1
    width = end - start + 1
    if width <= 0:
        raise ValueError("invalid training fraud block width")
    # Read labels exclusively from the outer fitting fold.
    y = np.full(len(base), -1, dtype=int)
    mapped = base.loc[fit, "label"].map({lab: k for k, lab in enumerate(LABELS)})
    if mapped.isna().any():
        raise ValueError("invalid fit label")
    y[fit] = mapped.to_numpy(dtype=int)
    score = np.zeros((width, K))
    labelled = fit & (dn >= start) & (dn <= end) & (y > 0)
    if not labelled.any():
        raise ValueError("no fraud-label support for training band edges")
    np.add.at(score, (dn[labelled] - start, y[labelled] - 1), 1.)
    edges_train = np.array([start] + [start + cut for cut in segment(score)] + [end + 1])
    # Empty segments are allowed by the original DP; fail for anchor segments
    # whose densities would be undefined instead of silently using NaN.
    support = fit & available
    mule = np.nan_to_num(p[:, 1], nan=0.) >= 0.5
    sim = (np.nan_to_num(p[:, 4] + p[:, 5], nan=0.) >= 0.5) & (kind == 1)
    lam_mule, count_mule = _density(dn, mule, support, edges_train[0], edges_train[1])
    lam_sim, count_sim = _density(dn, sim, support, edges_train[3], edges_train[4])
    edges_test = None
    test = ~train
    if test.any():
        start_test = int(dn[train].max()) + 1
        ext = test & (dn > BOT_POOL[1])
        if not ext.any() or (dn[ext] < start_test).any():
            raise ValueError("test device block overlaps train or has no extended devices")
        end_test = legit_block_start(dn[ext], start_test) - 1
        width_test = end_test - start_test + 1
        if width_test <= 0:
            raise ValueError("invalid test fraud block width")
        in_block = test & (dn >= start_test) & (dn <= end_test)
        if not in_block.any() or not available[in_block].all():
            raise ValueError("missing teacher probabilities inside test fraud block")
        prior = (edges_train[1:-1] - start) / width * width_test
        test_score = band_scores(p[in_block], kind[in_block], dn[in_block],
                                 start_test, width_test, lam_mule, lam_sim)
        cuts = segment(test_score, prior=prior, sd=0.02 * width_test)
        edges_test = np.array([start_test] + [start_test + c for c in cuts] + [end_test + 1])
    band = np.zeros(len(base), dtype=np.int8)
    for edges, mask in ((edges_train, train), (edges_test, test)):
        if edges is not None:
            for k in range(K):
                band[mask & (dn >= edges[k]) & (dn < edges[k + 1])] = ORDER[k]
    band[(dn >= BOT_POOL[0]) & (dn <= BOT_POOL[1])] = 2
    result = {"txn_id": base.txn_id.to_numpy(), "band": band.astype("float32"),
              "dev_ext_legit": ((dn > N_ACCOUNTS) & (band == 0)).astype("float32")}
    t = base.t.to_numpy()
    snd, rcv = entity_codes(base, "snd"), entity_codes(base, "rcv")
    for suffix, mask in [("any", band > 0)] + [(str(k), band == k) for k in ORDER]:
        result[f"snd_band_{suffix}_3h"] = Events(snd, t, mask=mask).count(snd, t, -3 * HOUR, 3 * HOUR).astype("float32")
        result[f"rcv_band_{suffix}_24h"] = Events(rcv, t, mask=mask).count(rcv, t, -DAY, DAY).astype("float32")
    diagnostic = {"train": edges_train.tolist(),
                  "test": None if edges_test is None else edges_test.tolist(),
                  "order": [LABELS[k] for k in ORDER], "bot_pool": list(BOT_POOL),
                  "lam_mule": lam_mule, "lam_sim": lam_sim,
                  "unique_mule_anchors": count_mule, "unique_sim_anchors": count_sim,
                  "fit_rows": int(fit.sum()), "teacher_fit_rows": int(support.sum()),
                  "empty_train_segments": int((np.diff(edges_train) == 0).sum())}
    return pd.DataFrame(result), diagnostic


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("tag")
    a = ap.parse_args(argv)
    processed = a.root / "data/processed"
    base = pd.read_parquet(processed / "base_v0.parquet")
    train = base.split.eq("train").to_numpy()
    p = np.full((len(base), 12), np.nan)
    p[train] = np.load(a.root / "models" / a.tag / "oof.npy")
    if (~train).any():
        p[~train] = np.load(a.root / "models" / a.tag / "test_proba.npy")
    frame, diagnostic = build_bands(base, p, pd.read_parquet(processed / "features_v1.parquet"),
                                    pd.Index(base.txn_id[train]))
    frame.to_parquet(processed / "features_v3.parquet", index=False)
    (processed / "band_edges.json").write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    print(json.dumps(diagnostic, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
