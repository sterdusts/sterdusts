"""因子评估：RankIC（非重叠采样 t 值）、分年份、分层收益、对公开因子正交化，并登记每次试验。"""
import json
import os
import time
from datetime import datetime

import numpy as np
import polars as pl

DATA = "/home/user/data"
TRIALS = os.path.join(os.path.dirname(__file__), "research", "trials.jsonl")
HOLDOUT_START = datetime(2026, 4, 1)   # 锁定验证集：2026-04-01 之后，研究期间不碰
MIN_NAMES = 15
BASELINE = ["ret_1h", "ret_4h", "ret_24h", "ret_7d", "vol_24h", "lqv", "funding_rate", "doi_4h", "doi_24h",
            "buy_4h", "beta", "dtopls_24h"]


def make_factors(p):
    eps = 1e-6
    p = p.with_columns(pl.col("qv_24h").log().alias("lqv"))
    f = {}
    for d in (1, 2, 3, 5, 10):
        f[f"h1_imb{d}"] = (pl.col(f"up{d}") - pl.col(f"dn{d}")) / (pl.col(f"up{d}") + pl.col(f"dn{d}") + eps)
        f[f"h1_diff{d}"] = pl.col(f"up{d}") - pl.col(f"dn{d}")
    f["h1_gimb"] = (pl.col("gup") - pl.col("gdn")) / (pl.col("gup") + pl.col("gdn") + eps)
    f["h1_gdiff"] = pl.col("gup") - pl.col("gdn")
    for w in ("1h", "4h", "24h"):
        f[f"h2_net{w}"] = pl.col(f"liqL_{w}") - pl.col(f"liqS_{w}")
    n4 = pl.col("liq_avg30") * 48 + eps
    f["h2_net4h_rel"] = (pl.col("liqL_4h") - pl.col("liqS_4h")) / n4
    f["h2_casc4h"] = (pl.col("liqL_4h") + pl.col("liqS_4h")) / n4
    f["fragility5"] = pl.col("up5") + pl.col("dn5")
    return p.with_columns(**f)


def cs_rank(cols):
    return [((pl.col(c).rank("average").over("t") - 0.5) / pl.col(c).count().over("t") - 0.5).alias(f"r_{c}")
            for c in cols]


def orthogonalize(p, fac, base=BASELINE):
    """每个时间截面：因子的排名对基准因子排名做 OLS，取残差。"""
    cols = [fac] + base
    q = p.select("t", "symbol", *cols).drop_nulls().filter(pl.all_horizontal([pl.col(c).is_finite() for c in cols]))
    q = q.with_columns(cs_rank(cols)).sort("t")
    t = q["t"].to_numpy()
    y = q[f"r_{fac}"].to_numpy()
    X = np.column_stack([np.ones(len(q))] + [q[f"r_{c}"].to_numpy() for c in base])
    res = np.full(len(q), np.nan)
    idx = np.flatnonzero(np.r_[True, t[1:] != t[:-1], True])
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a < MIN_NAMES:
            continue
        beta, *_ = np.linalg.lstsq(X[a:b], y[a:b], rcond=None)
        res[a:b] = y[a:b] - X[a:b] @ beta
    return q.select("t", "symbol").with_columns(pl.Series(f"{fac}_orth", res))


def ic_series(p, fac, target):
    q = p.select("t", fac, target).drop_nulls().filter(pl.col(fac).is_finite() & pl.col(target).is_finite())
    q = q.filter(pl.len().over("t") >= MIN_NAMES)
    return (q.group_by("t").agg(pl.corr(pl.col(fac).rank(), pl.col(target).rank()).alias("ic"),
                                pl.len().alias("n")).sort("t").drop_nulls().filter(pl.col("ic").is_not_nan()))


def spread_series(p, fac, target, q_=0.2):
    q = p.select("t", fac, target).drop_nulls().filter(pl.col(fac).is_finite() & pl.col(target).is_finite())
    q = q.filter(pl.len().over("t") >= MIN_NAMES)
    q = q.with_columns((pl.col(fac).rank().over("t") / pl.len().over("t")).alias("pct"))
    return q.group_by("t").agg((pl.col(target).filter(pl.col("pct") > 1 - q_).mean()
                                - pl.col(target).filter(pl.col("pct") <= q_).mean()).alias("ls")).sort("t")


def summarize(p, fac, horizons=(1, 4, 8, 24), period="dev"):
    if period == "dev":
        p = p.filter(pl.col("t") < HOLDOUT_START)
    out = {}
    for h in horizons:
        tgt = f"res_{h}h"
        ic = ic_series(p, fac, tgt)
        nov = ic.filter((pl.col("t").dt.hour() % h) == 0) if h <= 24 else ic
        m, s, n = nov["ic"].mean(), nov["ic"].std(), nov.height
        sp = spread_series(p, fac, tgt).filter((pl.col("t").dt.hour() % h) == 0)
        yr = ic.group_by(pl.col("t").dt.year().alias("y")).agg(pl.col("ic").mean()).sort("y")
        out[f"{h}h"] = {
            "ic": round(ic["ic"].mean(), 4),
            "t": round(m / s * np.sqrt(n), 2) if s and s > 0 else None,
            "ic_pos_frac": round((ic["ic"] > 0).mean(), 3),
            "ls_bps": round(sp["ls"].mean() * 1e4, 2),
            "by_year": {int(a): round(b, 4) for a, b in yr.iter_rows()},
        }
    return out


def log_trial(name, desc, result, extra=None):
    os.makedirs(os.path.dirname(TRIALS), exist_ok=True)
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "name": name, "desc": desc, "result": result}
    if extra:
        rec.update(extra)
    with open(TRIALS, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def n_trials():
    if not os.path.exists(TRIALS):
        return 0
    return sum(1 for _ in open(TRIALS))


def evaluate_factor(p, fac, desc, orth=True, log=True, horizons=(1, 4, 8, 24)):
    raw = summarize(p, fac, horizons)
    res = {"raw": raw}
    if orth:
        o = orthogonalize(p.filter(pl.col("t") < HOLDOUT_START), fac)
        po = p.join(o, on=["t", "symbol"], how="inner")
        res["orth"] = summarize(po, f"{fac}_orth", horizons)
    if log:
        log_trial(fac, desc, res)
    return res


def fmt(res):
    lines = []
    for k, v in res.items():
        row = "  ".join(f"{h}: IC={d['ic']:+.4f} t={d['t']} 多空={d['ls_bps']:+.1f}bp" for h, d in v.items())
        lines.append(f"  [{k}] {row}")
    return "\n".join(lines)
