"""Shared loading, row-level features and plotting style for the EDA scripts."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

LABELS = [
    "legit", "mule_account", "velocity_bot", "layering_chain", "sim_swap", "account_takeover",
    "phishing_collect", "qr_swap", "refund_abuse", "fake_cashback", "loan_app_extortion",
    "merchant_collusion",
]
FRAUD = LABELS[1:]
T0 = pd.Timestamp("2026-07-01")

# validated reference palette (dataviz skill), light mode
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
SEQ_BLUES = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2,
        "legend.frameon": False, "figure.dpi": 110,
    })


def seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("seqblue", SEQ_BLUES)


def load(root: Path):
    inter = root / "data/interim"
    tr = pd.read_parquet(inter / "train.parquet")
    te = pd.read_parquet(inter / "test.parquet")
    ac = pd.read_parquet(inter / "accounts.parquet")
    me = pd.read_parquet(inter / "merchants.parquet")
    tr["split"], te["split"] = "train", "test"
    te["label"] = np.nan
    df = pd.concat([tr, te], ignore_index=True)
    return df, ac, me


def add_row_features(df: pd.DataFrame, ac: pd.DataFrame, me: pd.DataFrame) -> pd.DataFrame:
    ts = df.timestamp
    df["t"] = ((ts - T0).dt.total_seconds()).astype(np.int64)
    df["month"] = ts.dt.strftime("%Y-%m")
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    df["is_night"] = df.hour.between(0, 5).astype(np.int8)
    df["is_late_evening"] = df.hour.between(20, 23).astype(np.int8)
    a = df.amount
    a_int = np.floor(a)
    df["amt_is_integer"] = (a == a_int).astype(np.int8)
    df["amt_round_100"] = ((a % 100) == 0).astype(np.int8)
    df["amt_round_500"] = ((a % 500) == 0).astype(np.int8)
    df["amt_round_1000"] = ((a % 1000) == 0).astype(np.int8)
    gap_to_next_1000 = np.ceil(a / 1000) * 1000 - a
    df["amt_just_under_1000s"] = ((gap_to_next_1000 > 0) & (gap_to_next_1000 <= 50) & (a >= 900)).astype(np.int8)
    df["amt_ends_99"] = ((a_int % 100) == 99).astype(np.int8)
    df["amt_tiny_le10"] = (a <= 10).astype(np.int8)
    df["amt_to_balance"] = a / df.balance_before.replace(0, np.nan)
    df["app_major"] = df.app_version.str.split(".").str[0]

    acc = ac.set_index("account_id")
    df["home_city"] = df.sender_id.map(acc.home_city)
    df["kyc_level"] = df.sender_id.map(acc.kyc_level)
    df["os"] = df.sender_id.map(acc.os)
    df["bank_code"] = df.sender_id.map(acc.bank_code)
    df["away_from_home"] = (df.device_city != df.home_city).astype(np.int8)
    df["home_device"] = (df.device_id == "DV0" + df.sender_id.str[2:]).astype(np.int8)
    opened = pd.to_datetime(df.sender_id.map(acc.opened_on))
    df["acct_age_from_opened"] = (ts - opened).dt.days

    p2p = df.receiver_type == "P2P"
    df["rcv_kyc_level"] = df.receiver_id.map(acc.kyc_level).where(p2p)
    rcv_open = pd.to_datetime(df.receiver_id.map(acc.opened_on))
    df["rcv_acct_age_days"] = (ts - rcv_open).dt.days.where(p2p)
    df["rcv_home_city"] = df.receiver_id.map(acc.home_city).where(p2p)

    mer = me.set_index("merchant_id")
    df["mer_category"] = df.receiver_id.map(mer.category)
    df["mer_city"] = df.receiver_id.map(mer.city)
    onb = pd.to_datetime(df.receiver_id.map(mer.onboarded_on))
    df["mer_age_days"] = (ts - onb).dt.days.where(~p2p)
    df["mer_cat_matches_txn"] = (df.mer_category == df.merchant_category).astype(float).where(~p2p)
    df["mer_city_eq_device_city"] = (df.mer_city == df.device_city).astype(float).where(~p2p)
    return df


def md_table(df: pd.DataFrame, floatfmt: str = ".3g") -> str:
    """Small markdown table writer (avoids the optional tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    idx_name = df.index.name or ""
    out = ["| " + " | ".join([str(idx_name)] + cols) + " |",
           "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else format(v, floatfmt))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(out)
