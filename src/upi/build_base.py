"""Build complete-context base and sequence ratios before any row filtering."""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from .features_v6 import align_pack, validate_ids
except ImportError:
    from features_v6 import align_pack, validate_ids


def build_tables(frame: pd.DataFrame, relational: pd.DataFrame):
    """Pure base/ratio construction; retains all input rows and row features."""
    validate_ids(frame)
    df = frame.sort_values(["t", "txn_id"], kind="stable").reset_index(drop=True).copy()
    R0 = align_pack(df, relational)
    for col in ("split", "label", "month"):
        if col in R0 and col in df and not df[col].equals(R0[col]):
            raise ValueError(f"relational metadata differs for {col}")
    for col in R0.columns:
        if col in {"txn_id", "split", "label", "month"}:
            continue
        if col in df:
            raise ValueError(f"duplicate base/relational column {col}")
        df[col] = R0[col]
    R = pd.DataFrame(index=df.index)
    R["bal/amt"] = df.balance_before / df.amount

    def seq(key, name, window=7200):
        keys = [key, "t", "txn_id"] if key != "receiver_id" else ["receiver_type", key, "t", "txn_id"]
        o = df.sort_values(keys, kind="stable")
        same_p = o[key].eq(o[key].shift())
        same_n = o[key].eq(o[key].shift(-1))
        if key == "receiver_id":
            same_p &= o.receiver_type.eq(o.receiver_type.shift())
            same_n &= o.receiver_type.eq(o.receiver_type.shift(-1))
        dtp = (o.t - o.t.shift()).where(same_p)
        dtn = (o.t.shift(-1) - o.t).where(same_n)
        okp, okn = same_p & (dtp <= window), same_n & (dtn <= window)
        for suffix, value, valid in (
            ("bal/prevbal", o.balance_before / o.balance_before.shift(), okp),
            ("bal/nextbal", o.balance_before / o.balance_before.shift(-1), okn),
            ("amt/prevamt", o.amount / o.amount.shift(), okp),
            ("amt/nextamt", o.amount / o.amount.shift(-1), okn),
            ("bal/prevamt", o.balance_before / o.amount.shift(), okp),
            ("amt/prevbal", o.amount / o.balance_before.shift(), okp),
            ("dt_prev", dtp, okp),
        ):
            R[f"{name}:{suffix}"] = value.where(valid).reindex(df.index)

    seq("sender_id", "snd")
    seq("receiver_id", "rcv")
    seq("device_id", "dev")
    # Account-to-account forwarding only: merchant IDs cannot match senders.
    out = df[["sender_id", "t", "amount", "balance_before"]].sort_values("t", kind="stable").rename(
        columns={"sender_id": "k", "amount": "o_amt", "balance_before": "o_bal", "t": "o_t"})
    q = df.loc[df.receiver_type.eq("P2P"), ["receiver_id", "t", "amount"]].rename(
        columns={"receiver_id": "k"}).sort_values("t", kind="stable")
    q["row"] = q.index
    m1 = pd.merge_asof(q, out, left_on="t", right_on="o_t", by="k",
                      direction="forward", allow_exact_matches=False, tolerance=7200).set_index("row")
    R["fwd:o_amt/amt"] = (m1.o_amt / m1.amount).reindex(df.index)
    R["fwd:o_bal/amt"] = (m1.o_bal / m1.amount).reindex(df.index)
    inn = df.loc[df.receiver_type.eq("P2P"), ["receiver_id", "t", "amount", "balance_before"]].sort_values("t", kind="stable").rename(
        columns={"receiver_id": "k", "amount": "i_amt", "balance_before": "i_bal", "t": "i_t"})
    q2 = df[["sender_id", "t", "amount", "balance_before"]].rename(
        columns={"sender_id": "k"}).sort_values("t", kind="stable")
    q2["row"] = q2.index
    m2 = pd.merge_asof(q2, inn, left_on="t", right_on="i_t", by="k",
                      direction="backward", allow_exact_matches=False, tolerance=7200).set_index("row")
    R["back:amt/i_amt"] = (m2.amount / m2.i_amt).reindex(df.index)
    R["back:bal/i_amt"] = (m2.balance_before / m2.i_amt).reindex(df.index)
    R = R.replace([np.inf, -np.inf], np.nan).astype("float32")
    # Carry IDs as a named index, preserving v1's legacy iteration over numeric columns.
    R.index = pd.Index(df.txn_id, name="txn_id")
    if not np.array_equal(R.index.to_numpy(), df.txn_id.to_numpy()):
        raise ValueError("ratio ID order differs from base")
    return df, R


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    a = ap.parse_args(argv)
    # EDA common has a different API from src/upi/common.py; resolve explicitly
    # only when executing the CLI, avoiding module-name collisions.
    spec = importlib.util.spec_from_file_location("upi_eda_common", a.root / "scripts/eda/common.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    frame, accounts, merchants = helper.load(a.root)
    frame = helper.add_row_features(frame, accounts, merchants)
    processed = a.root / "data/processed"
    df, ratios = build_tables(frame, pd.read_parquet(processed / "relational_v0.parquet"))
    processed.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed / "base_v0.parquet", index=False)
    ratios.to_parquet(processed / "ratios_v0.parquet", index=True)
    print("base_v0:", df.shape, "| ratios_v0:", ratios.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
