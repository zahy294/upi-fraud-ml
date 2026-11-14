"""Read-only sibling-shape investigation. Never fits a model or changes inputs.

Run with the installed Python 3.11 executable and -I. All numerical libraries
are limited to four threads before import. Outputs go to stdout unless a NEW
UTF-8 text file under docs/ is explicitly supplied. Labels are evaluation only.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "ARROW_NUM_THREADS"):
    os.environ[_key] = "4"
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
pa.set_cpu_count(4)
pa.set_io_thread_count(4)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/upi"))
from common import LABELS

CASES = [("1_sim_ato", 4, [5]), ("2_cash_phish", 9, [6]),
         ("3a_phish_cash", 6, [9]), ("3b_phish_loan", 6, [10]),
         ("4_layer_mule", 3, [1]), ("5_mule_bot", 1, [2]),
         ("6_qr_other", 7, [8, 11])]
MONTHS = ["2026-07", "2026-08", "2026-09"]
FORBIDDEN = {"label", "raw", "clean", "app_version", "app_major", "ip_risk_score"}


def load(root=ROOT):
    base = pd.read_parquet(root / "data/processed/base_v0.parquet")
    tr = base.split.eq("train").to_numpy()
    b = base.loc[tr].reset_index(drop=True)
    columns = {1: ["is_1p1", "is_1p02", "snd_drain_kind", "snd_vict_in_cnt_48h",
                   "rcv_vict_cnt_24h", "r_snd_bal_prevbal"],
               2: ["S_flag", "S_n", "S_span", "S_pos", "Ds_n", "Ds_span", "Ds_pos",
                   "Vs_n", "Vs_span", "Vs_pos", "qr_native", "L_flag", "R_flag"],
               3: ["band"],
               4: ["bait_cnt_before_7d", "bait_prev_dt_h", "payback11_next_dt_h", "pair_tiny_cnt_7d"],
               5: ["ch_len", "ch_depth_back", "ch_depth_fwd", "ch_root_is_102", "ch_root_snd_vict_in"]}
    for version, names in columns.items():
        f = pd.read_parquet(root / f"data/processed/features_v{version}.parquet", columns=["txn_id"]+names)
        if not np.array_equal(f.txn_id, base.txn_id):
            raise ValueError(f"Feature v{version} row order mismatch")
        for name in names:
            b[name] = f[name].to_numpy()[tr]
    clean = np.load(root / "models/oof_raw/y_clean.npy")
    oof = np.load(root / "models/oof_c3/oof.npy", mmap_mode="r")
    weights = np.load(root / "models/oof_c3/weights_s5_c3.npy")
    if clean.shape != (len(b),) or oof.shape != (len(b), len(LABELS)):
        raise ValueError("OOF / clean-label row count mismatch")
    b["pred"] = (oof * weights).argmax(axis=1)
    truth = pd.DataFrame({"clean": clean,
        "raw": b.label.map(dict(zip(LABELS, range(len(LABELS))))).astype(int),
        "app3": b.app_major.eq("3").to_numpy(), "ip": b.ip_risk_score.to_numpy()})
    b = b.drop(columns=list(FORBIDDEN.intersection(b.columns)))
    b["month"] = b.month.astype(str).where(b.month.astype(str) <= "2026-09", "2026-09")
    b["event"] = detected_components(b)
    return b, truth


def detected_components(b):
    """Exactly the user's grouping; no labels/traps, full detected train graph."""
    det = np.flatnonzero(b.pred.to_numpy() > 0)
    t = b.t.to_numpy(dtype=np.int64)
    p2p = b.receiver_type.eq("P2P").to_numpy()
    snd = b.sender_id.str[2:].to_numpy(dtype=np.int64)
    rnum = b.receiver_id.str[2:].to_numpy(dtype=np.int64)
    rcv = np.where(p2p, rnum, rnum + 1_000_000)
    dev = b.device_id.str[2:].to_numpy(dtype=np.int64)
    pair = np.minimum(snd, rcv)*2_000_000 + np.maximum(snd, rcv)
    edges = []
    for key, width, eligible in [(snd, 604800, np.ones(len(b), bool)),
                                 (rcv, 86400, p2p | b.S_flag.eq(1).to_numpy()),
                                 (dev, 600, np.ones(len(b), bool)), (pair, 604800, p2p)]:
        idx = det[eligible[det]]
        order = idx[np.lexsort((t[idx], key[idx]))]
        keep = (key[order[1:]] == key[order[:-1]]) & (np.diff(t[order]) <= width)
        edges.append((order[:-1][keep], order[1:][keep]))
    left = np.concatenate([a for a, _ in edges]); right = np.concatenate([z for _, z in edges])
    graph = coo_matrix((np.ones(len(left), dtype=np.int8), (left, right)), shape=(len(b), len(b)))
    return connected_components(graph, directed=False)[1]


def groups(b):
    # Band rows participate in the user's reporting IDs, but never supply rule attributes.
    return {int(e): np.asarray(idx, dtype=np.int64)
            for e, idx in b.loc[(b.pred > 0) & (b.band == 0)].groupby("event", sort=False).groups.items()}


def show_event(b, truth, rows):
    z = b.loc[rows].sort_values(["t", "txn_id"]).copy()
    z["dt"] = z.t.diff().fillna(0).astype(int)
    z["clean"] = np.asarray(LABELS)[truth.clean.to_numpy()[z.index]]
    z["raw"] = np.asarray(LABELS)[truth.raw.to_numpy()[z.index]]
    z["prediction"] = np.asarray(LABELS)[z.pred]
    cols = ["txn_id", "timestamp", "dt", "sender_id", "receiver_id", "device_id", "home_device",
            "amount", "balance_before", "txn_type", "merchant_category", "new_payee", "device_age_days",
            "sim_age_hours", "login_to_txn_sec", "pin_retries", "qr_vpa_mismatch", "prediction", "clean", "raw"]
    return z[cols].to_string(index=False, float_format=lambda x: format(x, ".10g"))


def side_by_side(a, z, title_a, title_z):
    left = [title_a]+a.splitlines(); right = [title_z]+z.splitlines()
    width = max(map(len, left))
    return "\n".join((left[i] if i < len(left) else "").ljust(width)+"  ||  "+
                     (right[i] if i < len(right) else "") for i in range(max(len(left), len(right))))


def inspect(b, truth, repeats=2):
    g = groups(b)
    for name, pred, siblings in CASES:
        print("\n===", name, "===", flush=True)
        populations = []
        for types in [[pred], siblings]:
            keep = b.band.eq(0).to_numpy() & b.pred.eq(pred).to_numpy() & np.isin(truth.clean, types)
            print("group", [LABELS[c] for c in types], "rows", int(keep.sum()), "events", b.event[keep].nunique(), flush=True)
            ids = b.loc[keep, "event"].drop_duplicates().tolist()
            # Earliest eligible events, one example per month, no size/label-purity filtering.
            chosen = []
            for month in MONTHS:
                available = b.loc[keep & b.month.eq(month).to_numpy(), "event"].drop_duplicates().tolist()
                chosen.extend(available[:repeats])
            populations.append(chosen)
        for j in range(min(len(populations[0]), len(populations[1]))):
            a, z = populations[0][j], populations[1][j]
            print(side_by_side(show_event(b, truth, g[a]), show_event(b, truth, g[z]),
                               f"{name} main event={a}", f"{name} sibling event={z}"), flush=True)


def session_map(b, mask, key, width, anchors):
    """Complete label-free sessions, including nondetected band-zero companions."""
    selected = np.flatnonzero(np.asarray(mask) & b.band.eq(0).to_numpy())
    t = b.t.to_numpy()
    key = np.asarray(key)
    order = selected[np.lexsort((t[selected], key[selected]))]
    starts = np.r_[True, (key[order[1:]] != key[order[:-1]]) | (np.diff(t[order]) > width)]
    sid = np.cumsum(starts)
    row_session = np.full(len(b), -1, dtype=np.int64)
    row_session[order] = sid
    needed = set(row_session[anchors]) - {-1}
    out = {}
    for session, ids in pd.Series(order).groupby(sid, sort=False):
        if session in needed:
            out[int(session)] = ids.to_numpy()
    return row_session, out


def chain_map(b, anchors):
    """Connected PAY/P2P forwarding links, 0 < dt <= 2h and .9 <= child/parent <= 1.

    Reconstruct independently of predictions, band types and labels. A parent
    is the most recent receipt into the sender, matching v5's timing policy.
    """
    good = b.band.eq(0) & b.receiver_type.eq("P2P") & b.txn_type.eq("PAY") & b.amount.eq(np.floor(b.amount))
    x = b.loc[good, ["sender_id", "receiver_id", "t", "amount"]].copy()
    x["row"] = x.index
    q = x[["row", "sender_id", "t", "amount"]].rename(columns={"sender_id":"key"}).sort_values("t")
    source = x[["row", "receiver_id", "t", "amount"]].rename(
        columns={"row":"parent", "receiver_id":"key", "t":"pt", "amount":"pa"}).sort_values("pt")
    matched = pd.merge_asof(q, source, left_on="t", right_on="pt", by="key",
                            tolerance=7200, direction="backward", allow_exact_matches=False)
    ratio = matched.amount / matched.pa
    valid = matched.loc[ratio.between(.9, 1), ["row", "parent"]].astype(np.int64)
    graph = coo_matrix((np.ones(len(valid), np.int8), (valid.row, valid.parent)), shape=(len(b),len(b)))
    _, comp = connected_components(graph, directed=False)
    needed = set(comp[anchors])
    out = {}
    members = np.unique(np.r_[valid.row, valid.parent, anchors])
    for event, ids in pd.Series(members).groupby(comp[members], sort=False):
        if event in needed:
            out[int(event)] = ids.to_numpy()
    return comp, out


def physical_groups(b, pred):
    anchor = np.flatnonzero(b.band.eq(0).to_numpy() & b.pred.eq(pred).to_numpy())
    p2p = b.receiver_type.eq("P2P").to_numpy()
    integer = b.amount.eq(np.floor(b.amount)).to_numpy()
    if pred == 4:
        mask = p2p & b.txn_type.eq("PAY").to_numpy() & integer & b.new_payee.eq(1).to_numpy()
        keys = pd.factorize(b.sender_id)[0]
        row_group, out = session_map(b, mask, keys, 7200, anchor)
        family = "drain"
    elif pred in [6, 9]:
        snd = b.sender_id.str[2:].to_numpy(dtype=np.int64)
        rcv = b.receiver_id.str[2:].to_numpy(dtype=np.int64)
        key = np.minimum(snd, rcv)*2_000_000+np.maximum(snd, rcv)
        row_group, out = session_map(b, p2p, key, 604800, anchor)
        family = "pair"
    elif pred == 3:
        row_group, out = chain_map(b, anchor)
        family = "chain"
    elif pred == 1:
        victim = p2p & b.txn_type.eq("PAY").to_numpy() & integer & b.new_payee.eq(1).to_numpy() & b.amount.mod(10).eq(0).to_numpy()
        keys = pd.factorize(b.receiver_id)[0]
        row_group, out = session_map(b, victim, keys, 86400, anchor)
        # Payout-only anchors can recover their incoming session by sender role.
        all_victims = b.loc[victim & b.band.eq(0).to_numpy()].groupby("receiver_id").indices
        vb = b.loc[victim & b.band.eq(0).to_numpy()]
        vrows = vb.index.to_numpy()
        recovered = {}
        for i in anchor[row_group[anchor] < 0]:
            r = b.iloc[i]
            positions = all_victims.get(r.sender_id)
            if positions is None:
                continue
            candidates = vrows[positions]
            candidates = candidates[(b.t.to_numpy()[candidates] < r.t) & (b.t.to_numpy()[candidates] >= r.t-3*86400)]
            if not len(candidates) or r.t-b.t.iloc[candidates[-1]] > 7200:
                continue
            existing = int(row_group[candidates[-1]])
            # Recover the full same-receiver session from the raw victim pool.
            complete = vrows[row_group[vrows] == existing]
            if not len(complete):
                continue
            out[existing] = np.unique(np.r_[out.get(existing, complete), i])
            row_group[i] = existing
            recovered[existing] = True
        # Attach outgoing transfers within 2h after the last incoming row, regardless of prediction.
        sends = b.loc[b.band.eq(0) & b.receiver_type.eq("P2P") & b.txn_type.eq("PAY")].groupby("sender_id").groups
        tt = b.t.to_numpy()
        for event, ids in list(out.items()):
            v = ids[victim[ids]]
            if not len(v):
                continue
            receiver = b.receiver_id.iloc[v[-1]]
            possible = np.asarray(sends.get(receiver, []), dtype=int)
            end = tt[v].max()
            possible = possible[(tt[possible] > end) & (tt[possible] <= end+7200)]
            if len(possible):
                out[event] = np.unique(np.r_[ids, possible])
        family = "fanin"
    else:
        row_group, out = session_map(b, b.S_flag.eq(1).to_numpy(), pd.factorize(b.receiver_id)[0], 86400, anchor)
        family = "shop"
    # Every eligible anchor has a unit, even when it has no recognizable shape.
    for i in anchor[row_group[anchor] < 0]:
        unit = -int(i)-1
        row_group[i] = unit
        out[unit] = np.array([i], dtype=np.int64)
    return anchor, row_group, out, family


def exact_parameters(x):
    """Observable event parameters only; refuses label/trap columns."""
    if FORBIDDEN.intersection(x.columns):
        raise ValueError("Labels/traps supplied to parameter extraction")
    x = x.sort_values(["t", "txn_id"])
    n = len(x)
    a = x.amount.to_numpy(dtype=float); bal = x.balance_before.to_numpy(dtype=float)
    t = x.t.to_numpy(dtype=np.int64); gaps = np.diff(t)
    cents = np.rint(bal*100).astype(np.int64)
    away = x.home_device.eq(0).to_numpy()
    p = {"n": n, "span": int(t[-1]-t[0]), "senders": x.sender_id.nunique(),
         "receivers": x.receiver_id.nunique(), "devices": x.device_id.nunique(),
         "unique_amounts": len(np.unique(a)), "amount_min": float(a.min()), "amount_max": float(a.max()),
         "amount_sum": float(a.sum()), "first_amount": a[0], "last_amount": a[-1],
         "first_balance": bal[0], "last_balance": bal[-1],
         "amount_gcd_cents": int(np.gcd.reduce(np.rint(a*100).astype(np.int64))),
         "amount_sorted": bool(np.all(np.diff(a)>=0)), "amount_decreasing": bool(np.all(np.diff(a)<=0)),
         "amount_repeat": bool(len(np.unique(a))<n), "amount_integer_all": bool(np.all(a==np.floor(a))),
         "amount_500_count": int((a==500).sum()), "amount_tiny_count": int((a<=10).sum()),
         "balance_floor11_count": int((np.abs(bal-1.1*a)<=.011).sum()),
         "balance_floor102_count": int((np.abs(bal-1.02*a)<=.011).sum()),
         "balance_below_amount_count": int((bal<a).sum()),
         "nonhome_count": int(away.sum()), "nonhome_first": bool(away[0]), "nonhome_last": bool(away[-1]),
         "nonhome_only_first": bool(away[0] and away.sum()==1),
         "nonhome_only_last": bool(away[-1] and away.sum()==1),
         "nonhome_positions": ",".join(map(str,np.flatnonzero(away))),
         "type_order": ",".join(x.txn_type.astype(str)),
         "category_order": ",".join(x.merchant_category.astype(object).fillna("NA").astype(str)),
         "amount_set": ",".join(format(v,".12g") for v in np.unique(a)),
         "gap_sequence": ",".join(map(str,gaps)),
         "newpayee_order": ",".join(map(str,x.new_payee)),
         "home_order": ",".join(map(str,x.home_device)),
         "first_type": str(x.txn_type.iloc[0]), "last_type": str(x.txn_type.iloc[-1]),
         "first_category": str(x.merchant_category.iloc[0]),
         "first_home_city_matches_receiver": bool(x.home_city.iloc[0] == x.rcv_home_city.iloc[0]) if pd.notna(x.rcv_home_city.iloc[0]) else False,
         "first_sender_receiver_same_kyc": bool(str(x.kyc_level.iloc[0]) == str(x.rcv_kyc_level.iloc[0])),
         "first_device_home_city": bool(x.device_city.iloc[0] == x.home_city.iloc[0]),
         "first_device_receiver_city": bool(x.device_city.iloc[0] == x.rcv_home_city.iloc[0]) if pd.notna(x.rcv_home_city.iloc[0]) else False,
         "first_merchant_category_matches": bool(str(x.merchant_category.iloc[0])==str(x.mer_category.iloc[0])),
         "first_merchant_city_matches": bool(str(x.device_city.iloc[0])==str(x.mer_city.iloc[0])),
         "first_acct_age": int(x.acct_age_from_opened.iloc[0]),
         "first_acct_age_matches": bool(x.sender_account_age_days.iloc[0]==x.acct_age_from_opened.iloc[0]),
         "first_amt_equals_acct_age": bool(a[0]==x.acct_age_from_opened.iloc[0]),
         "first_amt_equals_receiver_age": bool(a[0]==x.rcv_acct_age_days.iloc[0]),
         "first_amt_equals_merchant_age": bool(a[0]==x.mer_age_days.iloc[0]),
         "first_sender_same_os_all": bool(x.os.nunique()==1), "first_sender_same_bank_all": bool(x.bank_code.nunique()==1),
         "first_sender_same_city_all": bool(x.home_city.nunique()==1)}
    owner = "AC" + x.device_id.astype(str).str[3:]
    owners = set(owner[away]); participants = set(x.sender_id) | set(x.receiver_id)
    p["nonhome_owner_is_participant"] = bool(owners.intersection(participants))
    p["nonhome_owner_is_first_sender"] = bool(x.sender_id.iloc[0] in owners)
    p["nonhome_owner_is_first_receiver"] = bool(x.receiver_id.iloc[0] in owners)
    p["first_bank"] = str(x.bank_code.iloc[0]); p["first_os"] = str(x.os.iloc[0])
    p["first_kyc"] = str(x.kyc_level.iloc[0]);p["first_receiver_kyc"] = str(x.rcv_kyc_level.iloc[0])
    for name, series in [("sender", x.sender_id), ("receiver",x.receiver_id)]:
        codes = series.str[2:].to_numpy(dtype=np.int64)
        p[name+"_ascending"] = bool(n>1 and np.all(np.diff(codes)>0))
        p[name+"_descending"] = bool(n>1 and np.all(np.diff(codes)<0))
        p[name+"_first_is_min"] = bool(codes[0]==codes.min())
    for col, prefix in [(a,"amount"), (bal,"balance"), (a/np.maximum(bal,.001),"amount_balance")]:
        p[prefix+"_min"] = float(np.min(col)); p[prefix+"_max"] = float(np.max(col))
        p[prefix+"_median"] = float(np.median(col))
    for modulus in [2, 4, 5, 10, 25, 100]:
        p[f"first_balance_cents_mod{modulus}"] = int(cents[0]%modulus)
        p[f"first_amount_cents_mod{modulus}"] = int(round(a[0]*100)%modulus)
    for q in [.25, .35]:
        key = str(q).replace(".","p")
        if n>1:
            residual = bal[1:]-q*bal[:-1]
            p[f"q{key}_max_resid"] = float(np.abs(residual).max())
            p[f"q{key}_rounded_all"] = bool(np.all(np.abs(bal[1:]-np.round(q*bal[:-1],2))<1e-7))
            p[f"q{key}_floor_all"] = bool(np.all(cents[1:]==np.floor(q*cents[:-1]+1e-7)))
            p[f"q{key}_ceil_all"] = bool(np.all(cents[1:]==np.ceil(q*cents[:-1]-1e-7)))
            p[f"q{key}_resid_signs"] = ",".join(map(str,np.sign(np.round(residual,6)).astype(int)))
            if n<=12:
                scale = q**np.arange(n)
                lower = (bal-.00500001)/scale; upper = (bal+.00500001)/scale
                p[f"q{key}_latent_start_possible"] = bool(lower.max()<=upper.min())
            else:
                p[f"q{key}_latent_start_possible"] = False
        else:
            p[f"q{key}_max_resid"] = np.nan
            for suffix in ["rounded_all","floor_all","ceil_all","latent_start_possible"]:
                p[f"q{key}_{suffix}"] = False
            p[f"q{key}_resid_signs"] = ""
    for q in [.55,.6,.65,.7,.75,.8,.9,.95,.98,1.02,1.1]:
        p[f"amount_round_{q:g}_count"] = int((np.abs(a-np.maximum(500,np.round(q*bal)))<1e-7).sum())
    for unit in [2,5,10,30,60,300,600,3600,86400]:
        p[f"gap_all_mod{unit}"] = bool(len(gaps) and np.all(gaps%unit==0))
    for tag, value in [("min",np.min), ("max",np.max), ("median",np.median), ("mean",np.mean)]:
        p["gap_"+tag] = float(value(gaps)) if len(gaps) else np.nan
    p["gap_gcd"] = int(np.gcd.reduce(gaps)) if len(gaps) else 0
    p["gap_increasing"] = bool(n>=3 and np.all(np.diff(gaps)>=0))
    p["gap_decreasing"] = bool(n>=3 and np.all(np.diff(gaps)<=0))
    p["equal_gap"] = bool(n>=3 and len(np.unique(gaps))==1)
    p["repeat_gap"] = bool(len(np.unique(gaps))<len(gaps))
    p["same_second"] = bool(len(gaps) and np.any(gaps==0))
    for typ in ["PAY","COLLECT","QR","REFUND"]:
        p["type_"+typ+"_count"] = int(x.txn_type.eq(typ).sum())
    # Direction-sensitive bait/payback parameters.
    tiny = (a<=10) & x.txn_type.eq("PAY").to_numpy() & x.receiver_type.eq("P2P").to_numpy()
    big = np.flatnonzero(a>=500)
    p["bait_n"] = int(tiny.sum()); p["bait_set"] = ",".join(format(v,"g") for v in np.unique(a[tiny]))
    p["bait_order"] = ",".join(format(v,"g") for v in a[tiny])
    p["bait_first_amount"] = float(a[np.flatnonzero(tiny)[0]]) if tiny.any() else np.nan
    p["bait_bal_equal"] = bool(tiny.sum()>1 and np.ptp(bal[tiny])==0)
    p["payback_n"] = len(big)
    p["bait_reverse_n"] = 0; p["bait_last_to_payback"] = np.nan; p["bait_first_to_payback"] = np.nan
    if len(big):
        k = big[-1]
        reverse = tiny & x.sender_id.eq(x.receiver_id.iloc[k]).to_numpy() & x.receiver_id.eq(x.sender_id.iloc[k]).to_numpy() & (t<t[k])
        p["bait_reverse_n"] = int(reverse.sum())
        if reverse.any():
            p["bait_last_to_payback"] = int(t[k]-t[reverse][-1])
            p["bait_first_to_payback"] = int(t[k]-t[reverse][0])
        p["payback_floor_resid"] = float(bal[k]-1.1*a[k])
        p["payback_type"] = str(x.txn_type.iloc[k])
    else:
        p["payback_floor_resid"] = np.nan; p["payback_type"] = "NONE"
    # Directed chain and fan-in relationships, retaining the full precision of source rows.
    ratios=[]; chain_gaps=[]; joins=0
    sender=x.sender_id.to_numpy(); receiver=x.receiver_id.to_numpy()
    for k in range(1,n):
        possible=np.flatnonzero((receiver[:k]==sender[k]) & (t[k]-t[:k]<=7200) & (t[k]>t[:k]))
        if len(possible):
            j=possible[-1]; ratios.append(a[k]/a[j]);chain_gaps.append(int(t[k]-t[j]));joins+=1
    p["chain_links"] = joins
    p["chain_ratio_sequence"] = ",".join(format(v,".10f") for v in ratios)
    p["chain_gap_sequence"] = ",".join(map(str,chain_gaps))
    p["chain_ratio_min"] = float(min(ratios)) if ratios else np.nan
    p["chain_ratio_max"] = float(max(ratios)) if ratios else np.nan
    p["chain_same_ratio"] = bool(len(ratios)>1 and np.ptp(ratios)<1e-7)
    if n>1:
        incoming=(receiver[:-1]==sender[-1])
        p["fanin_n"] = int(incoming.sum())
        total=a[:-1][incoming].sum()
        p["payout_over_fanin"] = float(a[-1]/total) if total else np.nan
        p["payout_floor102_resid"] = float(bal[-1]-1.02*a[-1])
        p["payout_gap"] = int(t[-1]-t[:-1][incoming][-1]) if incoming.any() else np.nan
    else:
        p["fanin_n"]=0; p["payout_over_fanin"]=np.nan; p["payout_floor102_resid"]=np.nan;p["payout_gap"]=np.nan
    return p


def parameter_tables(b):
    result={}
    for pred in sorted(set(p for _,p,_ in CASES)):
        anchors, mapping, units, family = physical_groups(b, pred)
        records=[]
        for unit, rows in units.items():
            p=exact_parameters(b.loc[rows])
            p["unit"]=unit
            records.append(p)
        params=pd.DataFrame(records).set_index("unit")
        rows=params.reindex(mapping[anchors]).reset_index(drop=True)
        rows.index=anchors
        rows["unit"]=mapping[anchors]
        result[pred]=(rows,units,family)
        print("parameters",LABELS[pred],family,"anchors",len(anchors),"units",len(units),"columns",len(params.columns),flush=True)
    return result


def inspect_physical(b, truth, repeats=2):
    cache={}
    for name,pred,siblings in CASES:
        if pred not in cache:
            cache[pred]=physical_groups(b,pred)
        anchors,mapping,units,family=cache[pred]
        print("\n===",name,"physical",family,"===",flush=True)
        picks=[]
        for labels in [[pred],siblings]:
            mask=np.isin(truth.clean.to_numpy()[anchors],labels)
            selected=anchors[mask]
            print("group",[LABELS[c] for c in labels],"rows",len(selected),
                  "reporting_events",b.event.iloc[selected].nunique(),"physical_units",len(np.unique(mapping[selected])),flush=True)
            chosen=[]
            for month in MONTHS:
                candidates=selected[b.month.to_numpy()[selected]==month]
                chosen.extend(pd.unique(mapping[candidates])[:repeats])
            picks.append(chosen)
        for a,z in zip(*picks):
            print(side_by_side(show_event(b,truth,units[a]),show_event(b,truth,units[z]),
                               f"main {family}={a}",f"sibling {family}={z}"),flush=True)


def movement(b, truth, rows, target):
    """All other labels remain in denominator; event right means all moved rows right."""
    rows=np.asarray(rows,dtype=np.int64)
    out=[]
    for month in MONTHS+["ALL"]:
        r=rows if month=="ALL" else rows[b.month.to_numpy()[rows]==month]
        item={"month":month,"rows":len(r),"events":int(b.event.iloc[r].nunique())}
        for label in ["clean","raw"]:
            y=truth[label].to_numpy()[r]
            correct=y==target
            item[label+"_right_rows"]=int(correct.sum());item[label+"_wrong_rows"]=int((~correct).sum())
            item[label+"_harmed_rows"]=int((y==b.pred.to_numpy()[r]).sum())
            tab=pd.DataFrame({"event":b.event.to_numpy()[r],"right":correct}).groupby("event").right.agg(["all","any"])
            item[label+"_right_events"]=int(tab["all"].sum())
            item[label+"_wrong_events"]=int((~tab["all"]).sum())
            item[label+"_mixed_events"]=int((tab["any"] & ~tab["all"]).sum())
        item["app3_share"]=float(truth.app3.iloc[r].mean()) if len(r) else None
        out.append(item)
    return out


def predicates(table):
    """Finite interpretable checks; never train a model or use labels here."""
    skip={"unit"}
    for c in table.columns:
        if c in skip:
            continue
        s=table[c]
        unique=s.dropna().unique()
        if len(unique)<=40 or s.dtype==object:
            for value in unique:
                mask=s.eq(value).to_numpy()
                if mask.sum()>=3:
                    yield f"{c} == {value!r}",mask
        if pd.api.types.is_numeric_dtype(s) and s.dtype!=bool and len(unique)>4:
            # Fixed recipe-scale cutoffs, not an exhaustive optimization over observed values.
            if "resid" in c:
                thresholds=[-.011,-.006,-.001,0,.001,.006,.011,.02,.1,1]
            elif "ratio" in c or "amount_balance" in c:
                thresholds=[.25,.35,.5,.55,.6,.65,.7,.75,.8,.85,.9,.92,.95,.96,.97,.98,.99,1,1.02,1.1,1.2,1.5,2]
            elif "gap" in c or c=="span":
                thresholds=[0,1,5,10,15,30,60,120,300,600,900,1200,1800,3600,7200,10800,21600,43200,86400,172800,604800]
            elif c in ["n","senders","receivers","devices","unique_amounts","fanin_n","chain_links"] or c.endswith("_count"):
                thresholds=[1,2,3,4,5,6,8,10,12,15,20,25,30,40,60]
            elif "cents_mod" in c:
                thresholds=[]
            else:
                thresholds=[10,100,500,1000,1500,2000,2500,5000,10000,20000,25000,50000,100000]
            for value in thresholds:
                for op in ["<=",">"]:
                    mask=(s<=value if op=="<=" else s>value).to_numpy()
                    if 3<=mask.sum()<len(s):
                        yield f"{c} {op} {value}",mask


def scan(b,truth,tables):
    for name,pred,siblings in CASES:
        table,units,family=tables[pred]
        rows=table.index.to_numpy()
        print("\n=== SCAN",name,family,"===",flush=True)
        for lab in [pred]+siblings:
            r=rows[truth.clean.to_numpy()[rows]==lab]
            clean_table=table.loc[r].drop_duplicates("unit")
            cols=["n","span","gap_min","gap_max","gap_gcd","bait_reverse_n","first_amount","chain_ratio_min",
                  "chain_ratio_max","payout_over_fanin","nonhome_count","q0p25_max_resid"]
            print("LABEL",LABELS[lab],"units",len(clean_table),flush=True)
            print(clean_table[cols].quantile([0,.25,.5,.75,1]).round(8).to_string(),flush=True)
            for c in ["type_order","home_order","amount_set","bait_set","q0p25_rounded_all","q0p25_latent_start_possible","chain_links"]:
                print(c,clean_table[c].value_counts().head(12).to_dict(),flush=True)
        # Discover on July only. August/September outcomes cannot change candidate selection.
        july=b.month.to_numpy()[rows]==MONTHS[0]
        pool=list(predicates(table))
        for target in siblings:
            scored=[]
            for expr,mask in pool:
                selected=rows[mask & july]
                if b.event.iloc[selected].nunique()<3:
                    continue
                correct=truth.clean.to_numpy()[selected]==target
                if correct.sum()<3:
                    continue
                precision=float(correct.mean())
                scored.append((precision,int(correct.sum()),expr,mask))
            scored.sort(key=lambda x:(x[0],x[1]),reverse=True)
            print("JULY-RANKED CHECKS target",LABELS[target],"tested",len(pool),"supported",len(scored),flush=True)
            for precision,correct,expr,mask in scored[:12]:
                print("RULE",expr,"=>",LABELS[target],flush=True)
                print(json.dumps(movement(b,truth,rows[mask],target)),flush=True)
            qualified=[]
            for precision,correct,expr,mask in scored:
                if precision<1:
                    break
                stats=movement(b,truth,rows[mask],target)
                if stats[-1]["events"]>=30 and all(s["rows"]>0 and s["clean_wrong_rows"]==0 for s in stats[:3]):
                    qualified.append((expr,stats))
            print("EXACT_30_EVENT_RULES",json.dumps(qualified),flush=True)


def targeted(b,truth,tables,root=ROOT):
    """Predeclared multi-parameter checks plus complete accounting of labels."""
    for name,pred,siblings in CASES:
        tab,units,family=tables[pred]
        rows=tab.index.to_numpy()
        print("\n=== TARGETED",name,"===",flush=True)
        for label in [pred]+siblings:
            for month in MONTHS:
                chosen=rows[(truth.clean.to_numpy()[rows]==label) & (b.month.to_numpy()[rows]==month)]
                u=tab.loc[chosen].drop_duplicates("unit")
                counts={"type":LABELS[label],"month":month,"rows":len(chosen),
                        "report_events":int(b.event.iloc[chosen].nunique()),"physical_units":len(u),
                        "unit_sizes":{str(k):int(v) for k,v in u.n.value_counts().sort_index().items()},
                        "q25_latent":int(u.q0p25_latent_start_possible.sum()),
                        "q25_round":int(u.q0p25_rounded_all.sum()),
                        "gap_gcd1":int(u.gap_gcd.eq(1).sum()),
                        "first_only_nonhome":int(u.nonhome_only_first.sum()),
                        "last_only_nonhome":int(u.nonhome_only_last.sum()),
                        "two_reverse_baits":int(u.bait_reverse_n.eq(2).sum())}
                print("SUMMARY",json.dumps(counts),flush=True)
        # Contradictions are measured, not resolved using a vote or used in rule construction.
        scoped=truth.loc[rows,["clean","raw"]].copy();scoped["unit"]=tab.unit.to_numpy()
        for col in ["clean","raw"]:
            sizes=scoped.groupby("unit")[col].nunique()
            print("MIXED_UNITS",col,int((sizes>1).sum()),"of",len(sizes),flush=True)
        recipes=[]
        if pred==4:
            base=tab.q0p25_latent_start_possible & tab.n.ge(2)
            recipes=[("quarter recurrence AND n=2",base & tab.n.eq(2),5),
                     ("quarter recurrence AND rounded-from-previous balances",base & tab.q0p25_rounded_all,5),
                     ("quarter recurrence AND all gaps divisible by 30",base & tab.gap_all_mod30,5),
                     ("quarter recurrence AND nonhome only first",base & tab.nonhome_only_first,5),
                     ("quarter recurrence AND nonhome only last",base & tab.nonhome_only_last,5)]
        elif pred==9:
            recipes=[("exactly 1 reverse bait AND payback gap <=10800 AND payback COLLECT",
                      tab.bait_reverse_n.eq(1) & tab.bait_last_to_payback.le(10800) & tab.payback_type.eq("COLLECT"),6),
                     ("exactly 1 reverse bait AND balance exceeds 1.1 floor by >.011",
                      tab.bait_reverse_n.eq(1) & tab.payback_floor_resid.gt(.011),6),
                     ("payback row with at least 2 reverse baits [known positive control]",
                      tab.bait_reverse_n.ge(2) & b.amount.loc[rows].ge(500),9)]
        elif pred==6 and siblings==[9]:
            recipes=[("exactly 1 reverse bait AND payback gap >3600",
                      tab.bait_reverse_n.eq(1) & tab.bait_last_to_payback.gt(3600),9),
                     ("payback row with at least 2 reverse baits [known tell; residual scope]",
                      tab.bait_reverse_n.ge(2) & b.amount.loc[rows].ge(500),9)]
        elif pred==6:
            recipes=[("COLLECT singleton AND amount 4999",tab.n.eq(1)&tab.first_type.eq("COLLECT")&tab.first_amount.eq(4999),10),
                     ("COLLECT singleton AND nonhome device",tab.n.eq(1)&tab.first_type.eq("COLLECT")&tab.nonhome_first,10)]
        elif pred==3:
            recipes=[("chain >=4 rows AND every hop ratio >=.965 and <=.995",
                      tab.n.ge(4)&tab.chain_ratio_min.ge(.965)&tab.chain_ratio_max.le(.995),1),
                     ("chain >=2 rows AND gap gcd >1",tab.n.ge(2)&tab.gap_gcd.gt(1),1),
                     ("chain >=2 rows AND nonhome only first",tab.n.ge(2)&tab.nonhome_only_first,1),
                     ("chain >=2 rows AND nonhome only last",tab.n.ge(2)&tab.nonhome_only_last,1)]
        elif pred==1:
            recipes=[("fanin >=10 incoming AND payout balance exactly 1.02 within .011",
                      tab.fanin_n.ge(10)&tab.payout_floor102_resid.abs().le(.011),2),
                     ("fanin >=10 incoming AND nonhome only first",tab.fanin_n.ge(10)&tab.nonhome_only_first,2),
                     ("fanin >=10 incoming AND nonhome only last",tab.fanin_n.ge(10)&tab.nonhome_only_last,2)]
        else:
            recipes=[("three-row merchant session in QR,PAY,PAY order",tab.n.eq(3)&tab.type_order.eq("QR,PAY,PAY"),8),
                     ("three-row session with nonhome first and same sender bank",tab.n.eq(3)&tab.nonhome_first&tab.first_sender_same_bank_all,8),
                     ("native QR singleton with participant-owned nonhome device",tab.n.eq(1)&tab.first_type.eq("QR")&tab.nonhome_owner_is_participant,11)]
        for expr,mask,target in recipes:
            print("TARGET_RULE",expr,"=>",LABELS[target],flush=True)
            print(json.dumps(movement(b,truth,rows[np.asarray(mask)],target)),flush=True)

    # Multi-payout arithmetic: the busiest receiver is chosen WITHOUT labels.
    table,units,_=tables[1]
    rows=table.index.to_numpy()
    records=[]
    for unit,idx in units.items():
        x=b.loc[idx].sort_values("t")
        receiver=x.receiver_id.value_counts().index[0]
        incoming=x.loc[x.receiver_id.eq(receiver)]
        outgoing=x.loc[x.sender_id.eq(receiver) & x.t.gt(incoming.t.max())]
        amounts=incoming.amount.to_numpy(); total=amounts.sum(); out_amount=outgoing.amount.to_numpy()
        params={"unit":unit,"in_n":len(incoming),"out_n":len(outgoing),
                "payout_sum_ratio":out_amount.sum()/total if len(outgoing) else np.nan,
                "out_all_102":bool(len(outgoing) and np.all(np.abs(outgoing.balance_before-1.02*outgoing.amount)<=.011)),
                "out_sum_eq_first_balance":bool(len(outgoing) and abs(out_amount.sum()-outgoing.balance_before.iloc[0])<=.011),
                "in_amounts_unique":bool(len(np.unique(amounts))==len(amounts)),
                "out_amounts_decrease":bool(len(outgoing)>1 and np.all(np.diff(out_amount)<0)),
                "out_balances_decrease":bool(len(outgoing)>1 and np.all(np.diff(outgoing.balance_before)<0)),
                "payout_balance_resid_max":float(np.abs(outgoing.balance_before-1.02*outgoing.amount).max()) if len(outgoing) else np.nan}
        for factor in [.5,.6,.65,.7,.75,.8,.85,.9,.95,.98,1.]:
            params[f"payout_sum_round_{factor}"]=bool(len(outgoing) and abs(out_amount.sum()-round(total*factor))<1e-7)
        records.append(params)
    fan=pd.DataFrame(records).set_index("unit").reindex(table.unit).reset_index(drop=True)
    fan.index=rows
    print("\n=== MULTI-PAYOUT ARITHMETIC ===",flush=True)
    for label in [1,2]:
        selected=fan.loc[truth.clean.to_numpy()[rows]==label].copy()
        selected["unit"]=table.unit.loc[selected.index]
        selected=selected.drop_duplicates("unit")
        selected=selected[selected.in_n>=3]
        print(LABELS[label],"units",len(selected),"out_n",selected.out_n.value_counts().to_dict(),flush=True)
        print(selected[["in_n","out_n","payout_sum_ratio","payout_balance_resid_max"]].quantile([0,.5,1]).to_string(),flush=True)
        for c in fan.select_dtypes(include="bool").columns:
            print(c,int(selected[c].sum()),"of",len(selected),flush=True)
    for expr,mask in predicates(fan):
        m=mask & fan.in_n.ge(3).to_numpy()
        july=m & (b.month.to_numpy()[rows]==MONTHS[0])
        r=rows[july]
        if len(r) and (truth.clean.to_numpy()[r]==2).all() and b.event.iloc[r].nunique()>=3:
            print("PAYOUT_JULY_EXACT",expr,json.dumps(movement(b,truth,rows[m],2)),flush=True)
    print("PAYOUT_RULE all outgoing balance=1.02*amount AND in_n>=3",flush=True)
    print(json.dumps(movement(b,truth,rows[fan.in_n.ge(3)&fan.out_all_102],2)),flush=True)

    # Repeated sender histories for singleton-looking phishing/loan siblings.
    anchors=np.flatnonzero((b.pred==6)&(b.band==0))
    pool=b.receiver_type.eq("P2P") & b.txn_type.eq("COLLECT")
    mapping,units=session_map(b,pool,pd.factorize(b.sender_id)[0],604800,anchors)
    rec=[]
    for unit,ids in units.items():
        p=exact_parameters(b.loc[ids]);p["unit"]=unit;rec.append(p)
    par=pd.DataFrame(rec).set_index("unit")
    mapped=anchors[mapping[anchors]>=0]
    repeated=par.reindex(mapping[mapped]);repeated.index=mapped
    print("\n=== SENDER COLLECT HISTORIES ===",flush=True)
    for target in [6,9,10]:
        x=repeated.loc[truth.clean.to_numpy()[mapped]==target]
        print(LABELS[target],"rows",len(x),"counts",x.n.value_counts().to_dict(),flush=True)
    for target in [9,10]:
        for expr,mask in [("sender COLLECT session n>=3",repeated.n.ge(3)),
                          ("sender COLLECT session n>=2 with all different receivers",repeated.n.ge(2)&repeated.n.eq(repeated.receivers)),
                          ("sender COLLECT session n>=2, repeated exact amount",repeated.n.ge(2)&repeated.amount_repeat)]:
            print("HISTORY_RULE",expr,"=>",LABELS[target],flush=True)
            print(json.dumps(movement(b,truth,mapped[np.asarray(mask)],target)),flush=True)

    # Verify account/merchant-table joins rather than trusting a derived feature blindly.
    accounts=pd.read_csv(root/"data/raw/accounts.csv").set_index("account_id")
    merchants=pd.read_csv(root/"data/raw/merchants.csv").set_index("merchant_id")
    first=np.unique(np.concatenate([np.array([b.t.loc[v].idxmin() for v in units.values()])
                                    for _,units,_ in tables.values()]))
    x=b.loc[first]
    opened=pd.to_datetime(x.sender_id.map(accounts.opened_on))
    age=(x.timestamp-opened).dt.days
    merchant=x.receiver_type.eq("MERCHANT")
    onboard=pd.to_datetime(x.receiver_id.map(merchants.onboarded_on))
    mer_age=(x.timestamp-onboard).dt.days
    print("\nREFERENCE_TABLE_CHECK",json.dumps({"first_rows":len(first),
          "sender_age_mismatches":int(age.ne(x.acct_age_from_opened).sum()),
          "merchant_first_rows":int(merchant.sum()),
          "merchant_age_mismatches":int(mer_age[merchant].ne(x.mer_age_days[merchant]).sum()),
          "first_amount_equals_sender_age":int(x.amount.eq(age).sum()),
          "first_amount_equals_merchant_age":int(x.amount[merchant].eq(mer_age[merchant]).sum())}),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=["inspect","physical","scan","targeted"], default="inspect")
    parser.add_argument("--examples-per-month", type=int, default=2)
    parser.add_argument("--output", type=Path, help="New UTF-8 text file under docs/; never overwrite")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to((args.root / "docs").resolve()):
            parser.error("Output must be a new file under docs/")
        sys.stdout = output.open("x", encoding="utf-8")
    b, truth = load(args.root)
    print("loaded", len(b), "train rows; band-zero detected", int(((b.band == 0) & (b.pred > 0)).sum()), flush=True)
    if args.mode=="inspect":
        inspect(b, truth, args.examples_per_month)
    elif args.mode=="physical":
        inspect_physical(b,truth,args.examples_per_month)
    elif args.mode=="scan":
        scan(b,truth,parameter_tables(b))
    else:
        targeted(b,truth,parameter_tables(b),args.root)


if __name__ == "__main__":
    main()
