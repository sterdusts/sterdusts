# Independent re-implementation (verifier) of D1P-spot-lead-flow features + G0.
# Spec: /home/user/sterdusts/liqmap/research/flow_prereg_2026-10-10.json (commit b3f95f8), id=D1P-spot-lead-flow
# Features only. No fwd_/res_/fund_ columns are read here.
import json, hashlib, sys, os
from datetime import date
import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

OUT = "/home/user/data/flow/D1P-spot-lead-flow_verify"
FLOW = "/home/user/data/flow"
D0, DEND = date(2022, 1, 1), date(2026, 3, 22)

PCOLS = ["date", "symbol", "bar_status", "high", "low", "close", "ret_1d", "quote_volume", "taker_buy_quote_volume",
         "univ_excluded", "in_universe", "in_holdout", "beta_60d", "funding_1d", "oi_value", "top_ls_pos",
         "taker_ls_mean", "metrics_n"]
assert not any(c.startswith(("fwd_", "res_", "fund_")) for c in PCOLS)

# ---- holdout guard: features only up to the last eligible signal day
p = (pl.scan_parquet(f"{FLOW}/perp_daily.parquet").select(PCOLS)
     .filter((pl.col("date") <= DEND) & (~pl.col("in_holdout"))).collect())
assert p["date"].max() <= DEND
# EXCL_FROZEN: static classification column over all perp_daily symbols (no label/price data used)
excl = sorted(pl.scan_parquet(f"{FLOW}/perp_daily.parquet").filter(pl.col("univ_excluded")).select("symbol").unique().collect()["symbol"].to_list())
assert len(excl) == 189, len(excl)

trade = pl.col("bar_status") == "trade"
p = p.with_columns([pl.when(trade).then(pl.col(c)).otherwise(None).alias(c)
                    for c in ["close", "high", "low", "ret_1d", "quote_volume", "taker_buy_quote_volume"]])
p = p.with_columns(trade.alias("trade")).sort("symbol", "date")

btc = p.filter(pl.col("symbol") == "BTCUSDT").select("date", pl.col("ret_1d").alias("btc_r"))
p = p.join(btc, on="date", how="left")

# ---- perp epoch
p = p.with_columns(pl.when(pl.col("trade")).then(pl.col("date")).otherwise(None).alias("_td"))
p = p.with_columns(pl.col("_td").forward_fill().shift(1).over("symbol").alias("_prevtd"))
p = p.with_columns((pl.col("trade") & (pl.col("_prevtd").is_null() |
                    ((pl.col("date") - pl.col("_prevtd")).dt.total_days() >= 15))).alias("_newep"))
p = p.with_columns(pl.col("_newep").cast(pl.Int32).cum_sum().over("symbol").alias("eid"))
p = p.with_columns(pl.col("date").filter(pl.col("_newep")).min().over(["symbol", "eid"]).alias("epoch_start"))
p = p.with_columns((pl.col("date") - pl.col("epoch_start")).dt.total_days().alias("age_epoch"))

# ---- spot join
sp = (pl.scan_parquet(f"{FLOW}/spot_daily_perp.parquet")
      .select("date", pl.col("perp_symbol").alias("symbol"), "spot_base", "has_spot", "px_ok", "px_logdiff",
              "qv_usdt", "qv_usdt_USDT").filter(pl.col("date") <= DEND).collect())
sb = (pl.scan_parquet(f"{FLOW}/spot_daily_base.parquet")
      .select("date", "base", "tbqv_usdt_USDT", "tbqv_usdt", "days_since_prev").filter(pl.col("date") <= DEND).collect())
p = p.join(sp, on=["date", "symbol"], how="left").join(sb, left_on=["date", "spot_base"], right_on=["date", "base"], how="left")
assert p.height == p.select("date", "symbol").unique().height

# spot epoch start: base-layer restarts (days_since_prev>=4 or base first day) asof per base
brs = (sb.sort("base", "date").with_columns(pl.col("date").min().over("base").alias("_bfirst"))
       .filter((pl.col("days_since_prev") >= 4) | (pl.col("date") == pl.col("_bfirst")))
       .select(pl.col("base").alias("spot_base"), pl.col("date").alias("_brs_date")).sort("_brs_date"))
p = p.sort("date").join_asof(brs, left_on="date", right_on="_brs_date", by="spot_base", strategy="backward").sort("symbol", "date")
# base change (vs last non-null spot_base of this perp)
p = p.with_columns(pl.col("spot_base").forward_fill().shift(1).over("symbol").alias("_prevbase"))
p = p.with_columns(pl.when(pl.col("spot_base").is_not_null() &
                           (pl.col("_prevbase").is_null() | (pl.col("spot_base") != pl.col("_prevbase"))))
                   .then(pl.col("date")).otherwise(None).alias("_chg"))
p = p.with_columns(pl.col("_chg").forward_fill().over("symbol").alias("_lastchg"))
p = p.with_columns(pl.when(pl.col("spot_base").is_not_null())
                   .then(pl.max_horizontal("_brs_date", "_lastchg")).otherwise(None).alias("spot_epoch_start"))
p = p.with_columns(pl.col("spot_epoch_start").forward_fill().over("symbol").alias("spot_epoch_start"))
# monotone check (needed for contiguous segment grouping)
mono = p.select((pl.col("spot_epoch_start").diff().over("symbol").dt.total_days() < 0).sum()).item()
assert mono == 0, mono
p = p.with_columns(pl.col("spot_epoch_start").fill_null(date(1970, 1, 1)))
G1 = ["symbol", "eid"]
G2 = ["symbol", "eid", "spot_epoch_start"]
p = p.with_columns(pl.int_range(pl.len()).over(G1).alias("pos1"), pl.int_range(pl.len()).over(G2).alias("pos2"))

# ---- base quantities
p = p.with_columns((pl.col("ret_1d") - pl.col("beta_60d").fill_null(1.0) * pl.col("btc_r")).alias("rr"))
p = p.with_columns(
    pl.col("ret_1d").rolling_std(30, min_samples=20).over(G1).alias("sig30"),
    pl.col("quote_volume").rolling_median(30, min_samples=20).over(G1).alias("qv_med30"),
    (pl.col("trade") & pl.col("ret_1d").is_not_null() & pl.col("funding_1d").is_not_null()).cast(pl.Int32).alias("_pok"),
    (pl.col("has_spot").fill_null(False) & pl.col("px_ok").fill_null(False) & (pl.col("qv_usdt_USDT") > 0).fill_null(False)
     & (pl.col("qv_usdt_USDT") >= 0.5 * pl.col("qv_usdt")).fill_null(False) & (pl.col("days_since_prev") == 1).fill_null(False)).alias("s_ok"),
    (pl.col("has_spot").fill_null(False) & pl.col("px_ok").fill_null(False) & (pl.col("days_since_prev") == 1).fill_null(False)).alias("_sall_ok"),
)
p = p.with_columns((pl.col("_pok").rolling_sum(5, min_samples=5).over(G1) == 5).fill_null(False).alias("perp5_ok"),
                   (pl.col("s_ok").cast(pl.Int32).rolling_sum(5, min_samples=5).over(G2) == 5).fill_null(False).alias("_s5"),
                   (pl.col("_sall_ok").cast(pl.Int32).rolling_sum(5, min_samples=5).over(G2) == 5).fill_null(False).alias("_sall5"))
p = p.with_columns((pl.col("perp5_ok") & pl.col("_s5")).alias("win5_ok"))

def rs(c, n, g, m=None):
    return pl.col(c).rolling_sum(n, min_samples=n if m is None else m).over(g)

p = p.with_columns(
    pl.when(pl.col("win5_ok")).then(rs("tbqv_usdt_USDT", 5, G2) / rs("qv_usdt_USDT", 5, G2)).alias("S5"),
    pl.when(pl.col("perp5_ok")).then(rs("taker_buy_quote_volume", 5, G1) / rs("quote_volume", 5, G1)).alias("P5"),
    pl.when(pl.col("win5_ok")).then((rs("qv_usdt_USDT", 5, G2) / (rs("qv_usdt_USDT", 5, G2) + rs("quote_volume", 5, G2))).log()).alias("SH5"),
    pl.when(pl.col("_sall5") & (rs("qv_usdt", 5, G2) > 0)).then(rs("tbqv_usdt", 5, G2) / rs("qv_usdt", 5, G2)).alias("S5all"),
    pl.when(pl.col("high") > pl.col("low")).then((pl.col("high") / pl.col("low")).log().log()).otherwise(None).alias("lrng"),
)

def robust_z(x, pos, lo, hi, minn, chunk=100000):
    """z of x[j] vs baseline x[j-hi..j-lo] restricted to same segment (lag<=pos)."""
    N = len(x); L = hi - lo + 1
    out = np.full(N, np.nan)
    pad = np.concatenate([np.full(hi, np.nan), x])
    sw = sliding_window_view(pad, L)  # row j -> x[j-hi .. j-lo]
    lags = hi - np.arange(L)
    for j0 in range(0, N, chunk):
        j1 = min(N, j0 + chunk)
        W = sw[j0:j1].copy()
        W[lags[None, :] > pos[j0:j1, None]] = np.nan
        cnt = (~np.isnan(W)).sum(1)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = np.nanmedian(W, 1)
                mad = np.nanmedian(np.abs(W - med[:, None]), 1)
        s = 1.4826 * mad
        z = (x[j0:j1] - med) / s
        bad = (cnt < minn) | ~(s >= 1e-4) | np.isnan(x[j0:j1])
        z[bad] = np.nan
        out[j0:j1] = z
    return out

pos1 = p["pos1"].to_numpy(); pos2 = p["pos2"].to_numpy()
zc = {}
for c, pos in [("S5", pos2), ("P5", pos1), ("SH5", pos2), ("S5all", pos2)]:
    zc["z" + c] = robust_z(p[c].fill_null(np.nan).to_numpy().astype(float), pos, 5, 64, 40)
zc["zrng"] = robust_z(p["lrng"].fill_null(np.nan).to_numpy().astype(float), pos1, 1, 60, 40)
p = p.with_columns([pl.Series(k, v).fill_nan(None) for k, v in zc.items()])

qvs = pl.when(pl.col("s_ok")).then(pl.col("qv_usdt_USDT")).otherwise(None)
p = p.with_columns(qvs.alias("_qvs"))
p = p.with_columns(
    pl.col("rr").alias("r1"),
    rs("rr", 3, G1).alias("r3"),
    pl.when(pl.col("perp5_ok")).then(rs("rr", 5, G1) / (pl.col("sig30") * np.sqrt(5))).alias("r5z"),
    rs("rr", 7, G1).alias("r7"),
    (pl.col("rr").rolling_mean(30, min_samples=25).over(G1) * 30).alias("r30"),
    (pl.col("quote_volume").rolling_mean(5, min_samples=5).over(G1) /
     pl.col("quote_volume").rolling_median(30, min_samples=20).shift(5).over(G1)).log().alias("vx5"),
    (pl.col("_qvs").rolling_mean(5, min_samples=5).over(G2) /
     pl.col("_qvs").rolling_median(30, min_samples=20).shift(5).over(G2)).log().alias("vxS5"),
    pl.when(pl.col("perp5_ok")).then(rs("funding_1d", 5, G1)).alias("F5"),
    (pl.col("taker_buy_quote_volume") / pl.col("quote_volume")).alias("tbr1"),
    pl.col("px_logdiff").alias("px"),
    (pl.col("px_logdiff") - pl.col("px_logdiff").shift(5).over(G2)).alias("dbasis5"),
    pl.col("qv_med30").log().alias("lqv"),
    pl.col("beta_60d").alias("beta"),
    pl.when(pl.col("age_epoch") > 0).then(pl.col("age_epoch").cast(pl.Float64).log()).alias("lage"),
    pl.col("in_universe").cast(pl.Float64).alias("inU"),
)
p = p.with_columns((pl.col("F5") < 0).cast(pl.Float64).alias("negF5"),
                   (pl.col("F5") - pl.col("F5").shift(5).over(G1)).alias("dF5"))
# metrics
mok = ((pl.col("metrics_n") >= 280) & (pl.col("metrics_n").shift(5).over(G1) >= 280)).fill_null(False)
oiv, oiv5 = pl.col("oi_value"), pl.col("oi_value").shift(5).over(G1)
tp, tp5 = pl.col("top_ls_pos"), pl.col("top_ls_pos").shift(5).over(G1)
p = p.with_columns(
    pl.when(mok & (oiv > 0) & (oiv5 > 0)).then((oiv / oiv5).log()).alias("doi5"),
    pl.when(mok & (oiv > 0) & (pl.col("qv_med30") > 0)).then((oiv / pl.col("qv_med30")).log()).alias("oiqv"),
    pl.when(mok & (pl.col("taker_ls_mean") > 0)).then(pl.col("taker_ls_mean")).alias("tls"),
    pl.when(mok & (tp > 0) & (tp5 > 0)).then((tp / tp5).log()).alias("dtopls5"),
)
ZC = ["r1", "r3", "r5z", "r7", "r30", "vx5", "vxS5", "sig30", "zrng", "F5", "negF5", "dF5", "P5", "zP5", "tbr1", "lqv",
      "beta", "lage", "zSH5", "px", "dbasis5", "inU"]
MC = ["doi5", "oiqv", "tls", "dtopls5"]
# replace inf/nan with null for all features
p = p.with_columns([pl.when(pl.col(c).is_finite()).then(pl.col(c)).otherwise(None).alias(c)
                    for c in ZC + MC + ["zS5", "zS5all", "S5", "SH5", "S5all"]])

E = (~pl.col("symbol").is_in(excl) & (pl.col("symbol") != "BTCUSDT") & pl.col("trade") & pl.col("perp5_ok")
     & (pl.col("age_epoch") >= 60) & pl.col("beta_60d").is_not_null() & pl.col("sig30").is_not_null()
     & pl.col("qv_med30").is_not_null() & (pl.col("qv_med30") >= 2e7) & (~pl.col("in_holdout"))
     & (pl.col("date") >= D0) & (pl.col("date") <= DEND))
p = p.with_columns(E.fill_null(False).alias("E"))
f = p.filter(pl.col("E"))
f = f.with_columns((pl.col("win5_ok") & pl.col("zS5").is_not_null() & pl.col("zP5").is_not_null()
                    & pl.col("zSH5").is_not_null()).alias("S"))
s = f.filter(pl.col("S"))
s = s.with_columns((pl.col("zS5") - pl.col("zP5")).alias("_d"))
s = s.with_columns(((pl.col("_d") - pl.col("_d").median().over("date")).clip(-5, 5)).alias("A"),
                   (pl.col("zS5") - pl.col("zS5").median().over("date")).alias("zS5_cs"),
                   pl.col("F5").median().over("date").alias("F5med"),
                   ((pl.col("zS5all") - pl.col("zP5")) - (pl.col("zS5all") - pl.col("zP5")).median().over("date")).clip(-5, 5).alias("A_all"))
s = s.with_columns(pl.all_horizontal([pl.col(c).is_not_null() for c in ZC]).alias("inC"))
s = s.with_columns(pl.col("inC").sum().over("date").alias("nC"))
s = s.with_columns((pl.col("inC") & (pl.col("nC") >= 40)).alias("C"))

# ---- events
q = (pl.col("r5z").abs() <= 1) & (pl.col("F5") <= pl.col("F5med"))
hi_s, hi_a, shp = pl.col("zS5_cs") >= 2, pl.col("A") >= 1, pl.col("zSH5") >= 0
s = s.with_columns(
    (hi_s & hi_a & q & shp).alias("raw_D1"),
    (hi_s & hi_a & (pl.col("r5z") > 1) & (pl.col("F5") <= pl.col("F5med")) & shp).alias("raw_C1"),
    (hi_s & ~hi_a & q & shp).alias("raw_C2"),
    (~hi_s & hi_a & q & shp).alias("raw_C3"),
    (~hi_s & ~hi_a & q).alias("raw_C4"),
)
G = ["D1", "C1", "C2", "C3", "C4"]
chk = s.select(pl.sum_horizontal([pl.col("raw_" + g).cast(pl.Int32) for g in G]).max()).item()
assert chk <= 1, "groups not mutually exclusive"

def decluster(df):
    df = df.sort("symbol", "date")
    kept = np.zeros(df.height, dtype=bool)
    sym = df["symbol"].to_numpy(); dd = df["date"].to_physical().to_numpy()
    last = {}
    for i in range(df.height):
        k = sym[i]
        if k in last and dd[i] - last[k] <= 6:
            continue
        kept[i] = True; last[k] = dd[i]
    return df.with_columns(pl.Series("kept", kept))

for g in G:
    ev = decluster(s.filter(pl.col("raw_" + g)).select("symbol", "date"))
    s = s.join(ev.filter(pl.col("kept")).select("symbol", "date", pl.lit(True).alias("kept_" + g)), on=["symbol", "date"], how="left")
    s = s.with_columns(pl.col("kept_" + g).fill_null(False))

# ---- G0
days = s.group_by("date").agg(pl.col("C").sum().alias("nC"), pl.len().alias("nS")).sort("date")
alldays = pl.DataFrame({"date": pl.date_range(D0, DEND, "1d", eager=True)})
days = alldays.join(days, on="date", how="left").fill_null(0)

def prank(col):
    return ((pl.col(col).rank("average").over("date") - 0.5) / pl.len().over("date"))

BIN = {"negF5", "inU"}
c = s.filter(pl.col("C")).sort("date", "symbol")
c = c.with_columns(prank("A").alias("u_A"), *[(pl.col(z) if z in BIN else prank(z)).alias("u_" + z) for z in ZC])
r2 = []
eA = np.full(c.height, np.nan)
dates = c["date"].to_numpy()
bounds = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
UA = c["u_A"].to_numpy(); UZ = c.select(["u_" + z for z in ZC]).to_numpy()
for a, b in zip(bounds[:-1], bounds[1:]):
    X = np.column_stack([np.ones(b - a), UZ[a:b]]); y = UA[a:b]
    beta_, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta_
    eA[a:b] = res
    r2.append(1 - (res @ res) / ((y - y.mean()) @ (y - y.mean())))
c = c.with_columns(pl.Series("e_A", eA))
kept_counts = {g: int(s["kept_" + g].sum()) for g in G}
raw_counts = {g: int(s["raw_" + g].sum()) for g in G}
by_year = {g: s.filter(pl.col("kept_" + g)).group_by(pl.col("date").dt.year().alias("y")).len().sort("y")
           .to_dict(as_series=False) for g in G}
merge_c12 = not (kept_counts["C1"] >= 300 and kept_counts["C2"] >= 300)
g0 = dict(
    n_days=days.height,
    C_median=float(days["nC"].median()), C_ge40_share=float((days["nC"] >= 40).mean()),
    S_median=float(days["nS"].median()),
    R2_median=float(np.median(r2)), kept=kept_counts, raw=raw_counts, kept_by_year=by_year, merge_C12=merge_c12,
    zS5_avail_by_year=f.group_by(pl.col("date").dt.year().alias("y")).agg(pl.col("zS5").is_not_null().mean()).sort("y").to_dict(as_series=False),
)
g0["passed"] = bool(g0["C_median"] >= 50 and g0["C_ge40_share"] >= 0.9 and g0["R2_median"] <= 0.70 and kept_counts["D1"] >= 300)
print(json.dumps(g0, indent=1, default=str))

keep = (["date", "symbol", "in_universe", "beta_60d", "qv_med30", "A", "A_all", "zS5", "zS5_cs", "F5med", "zS5all",
         "S", "C", "nC", "spot_base", "eid", "epoch_start", "spot_epoch_start"] + ZC + MC
        + ["raw_" + g for g in G] + ["kept_" + g for g in G])
s.select(keep).write_parquet(f"{OUT}/v_S.parquet")
c.select(["date", "symbol", "e_A", "u_A"] + ["u_" + z for z in ZC]).write_parquet(f"{OUT}/v_C_ranks.parquet")
json.dump(g0, open(f"{OUT}/v_g0.json", "w"), indent=1, default=str)
