# Independent re-implementation (verifier) of D1P-spot-lead-flow main test statistics.
# Reads labels/prices only after G0 (v_g0.json) passed. Dev period only.
import json, glob, sys
from datetime import date, datetime, timezone, timedelta
import numpy as np
import polars as pl
from scipy import stats

OUT = "/home/user/data/flow/D1P-spot-lead-flow_verify"
FLOW = "/home/user/data/flow"
g0 = json.load(open(f"{OUT}/v_g0.json"))
assert g0["passed"], "G0 failed; labels must not be read"

D0, DEND = date(2022, 1, 1), date(2026, 3, 22)
PMAX = date(2026, 3, 30)                       # last allowed price date
FRMAX = int(datetime(2026, 3, 31, tzinfo=timezone.utc).timestamp() * 1000)  # last allowed settlement ts
DAYMS = 86400000
ZC = ["r1", "r3", "r5z", "r7", "r30", "vx5", "vxS5", "sig30", "zrng", "F5", "negF5", "dF5", "P5", "zP5", "tbr1", "lqv",
      "beta", "lage", "zSH5", "px", "dbasis5", "inU"]
MC = ["doi5", "oiqv", "tls", "dtopls5"]
G = ["D1", "C1", "C2", "C3", "C4"]
res = {}

S = pl.read_parquet(f"{OUT}/v_S.parquet")
CR = pl.read_parquet(f"{OUT}/v_C_ranks.parquet")
assert S["date"].max() <= DEND

# ---------------- prices (holdout guard: date<=2026-03-30, in_holdout False)
px = (pl.scan_parquet(f"{FLOW}/perp_daily.parquet")
      .select("date", "symbol", "bar_status", "close", "funding_1d", "in_holdout", "low", "open")
      .filter((pl.col("date") <= PMAX) & (~pl.col("in_holdout"))).collect())
assert px["date"].max() <= PMAX
px = px.with_columns(pl.when(pl.col("bar_status") == "trade").then(pl.col("close")).otherwise(None).alias("close"),
                     (pl.col("bar_status") == "trade").alias("trade")).sort("symbol", "date")
# strict-label existence flag: fwd_7d non-null, read only for d<=2026-03-22 (d+7<=03-29)
f7 = (pl.scan_parquet(f"{FLOW}/perp_daily.parquet").select("date", "symbol", "fwd_7d", "fwd_7d_lv", "in_holdout")
      .filter((pl.col("date") <= DEND) & (~pl.col("in_holdout"))).collect())

# ---------------- funding settlements FR
ff = sorted(glob.glob("/home/user/data/funding_*.parquet")) + [f"{FLOW}/perp_daily_aux_funding_backfill.parquet"]
fr = pl.concat([pl.read_parquet(x).select("symbol", "calc_time", "funding_rate") for x in ff])
fr = fr.unique(["symbol", "calc_time"], keep="first").with_columns((pl.col("calc_time") // 1000 * 1000).alias("ts"))
fr = fr.filter(pl.col("ts") <= FRMAX).sort("symbol", "ts")
assert fr["ts"].max() <= FRMAX
FRD = {}
for (sym,), g in fr.group_by(["symbol"]):
    g = g.sort("ts")
    FRD[sym] = (g["ts"].to_numpy(), np.r_[0.0, np.cumsum(g["funding_rate"].to_numpy())])

def fund_exec(sym, a, b):
    """sum of funding_rate with ts in (a, b]; a, b arrays of ms."""
    a = np.asarray(a, dtype=np.int64); b = np.asarray(b, dtype=np.int64)
    assert (b <= FRMAX).all()
    if sym not in FRD:
        return np.zeros(len(a))
    ts, cs = FRD[sym]
    return cs[np.searchsorted(ts, b, "right")] - cs[np.searchsorted(ts, a, "right")]

def d2ms(d):  # date array (days since epoch int) -> ms at 00:00
    return d.astype(np.int64) * DAYMS

# per-symbol arrays (complete calendar grid)
PX = {}
for (sym,), g in px.group_by(["symbol"]):
    g = g.sort("date")
    dd = g["date"].to_physical().to_numpy()
    assert (np.diff(dd) == 1).all()
    PX[sym] = dict(d0=dd[0], close=g["close"].fill_null(np.nan).to_numpy(), trade=g["trade"].to_numpy(),
                   fnull=g["funding_1d"].is_null().to_numpy(), low=g["low"].fill_null(np.nan).to_numpy(),
                   open=g["open"].fill_null(np.nan).to_numpy())
B = PX["BTCUSDT"]

def exit_idx(P, i0, h):
    """last trade index in (i0, i0+h]; -1 if none."""
    n = len(P["close"])
    hi = min(i0 + h, n - 1)
    for k in range(hi, i0, -1):
        if P["trade"][k]:
            return k
    return -1

def label(sym, d, beta, qv, h=7, lag=0, strict=False):
    """returns (y_g, fwd, crt) for signal day d (int days). lag=1: entry close_{d+1}."""
    P = PX.get(sym)
    if P is None:
        return np.nan, np.nan
    i0 = d - P["d0"] + lag
    if i0 < 0 or i0 >= len(P["close"]) or not P["trade"][i0]:
        return np.nan, np.nan
    ie = exit_idx(P, i0, h)
    if ie < 0:
        return np.nan, np.nan
    if strict and ie != i0 + h:
        return np.nan, np.nan
    e = P["d0"] + ie
    assert e <= (PMAX - date(1970, 1, 1)).days
    if P["fnull"][i0 + 1: ie + 1].any():
        return np.nan, np.nan
    fwd = np.log(P["close"][ie] / P["close"][i0])
    a, b = d2ms(np.array([d + lag + 1])), d2ms(np.array([e + 1]))
    fi = fund_exec(sym, a, b)[0]
    bi0, bie = d + lag - B["d0"], e - B["d0"]
    if np.isnan(B["close"][bi0]) or np.isnan(B["close"][bie]) or B["fnull"][bi0 + 1: bie + 1].any():
        return np.nan, np.nan
    fb = fund_exec("BTCUSDT", a, b)[0]
    hb = min(max(beta, 0.5), 2.0)
    yg = (fwd - fi) - hb * (np.log(B["close"][bie] / B["close"][bi0]) - fb)
    return yg, fwd

def crt(beta, qv):
    hb = np.clip(beta, 0.5, 2.0)
    slip = np.where(qv >= 1e8, 0.0002, np.where(qv >= 5e7, 0.0005, 0.0010))
    return 2 * (0.0005 + slip) + 2 * 0.0005 * hb

# label rows = all S rows (C rows for IC; S rows needed for sleeve)
L = S.select("date", "symbol", "beta_60d", "qv_med30", "C").with_columns(pl.col("date").to_physical().alias("di"))
cols = {k: [] for k in ["y7_g", "fwd7", "y7_strict", "y7_lag1", "y3_g", "y14_g"]}
for sym, d, b, q in zip(L["symbol"], L["di"], L["beta_60d"], L["qv_med30"]):
    y, fw = label(sym, d, b, q); cols["y7_g"].append(y); cols["fwd7"].append(fw)
    cols["y7_strict"].append(label(sym, d, b, q, strict=True)[0])
    cols["y7_lag1"].append(label(sym, d, b, q, lag=1)[0])
    cols["y3_g"].append(label(sym, d, b, q, h=3)[0])
    cols["y14_g"].append(label(sym, d, b, q, h=14)[0] if d <= (date(2026, 3, 15) - date(1970, 1, 1)).days else np.nan)
L = L.with_columns([pl.Series(k, np.array(v, dtype=float)).fill_nan(None) for k, v in cols.items()])
L = L.with_columns(pl.Series("c_rt", crt(L["beta_60d"].to_numpy(), L["qv_med30"].to_numpy())))
L = L.with_columns((pl.col("y7_g") - pl.col("c_rt")).alias("y7_n"))
# cross-check own fwd vs perp_daily fwd_7d_lv / strict existence vs fwd_7d
chk = L.join(f7, on=["date", "symbol"], how="left")
res["label_check"] = dict(
    n=chk.height,
    maxdiff_fwd_lv=float(chk.select((pl.col("fwd7") - pl.col("fwd_7d_lv")).abs().max()).item() or 0),
    null_mismatch_fwd_lv=int(chk.select((pl.col("fwd7").is_null() != pl.col("fwd_7d_lv").is_null()).sum()).item()),
    strict_vs_fwd7d_null_mismatch=int(chk.select(((pl.col("y7_strict").is_not_null()) & pl.col("fwd_7d").is_null()).sum()).item()),
    y7g_null_in_C=int(chk.filter(pl.col("C"))["y7_g"].null_count()), C_rows=int(chk["C"].sum()))
print(res["label_check"])
L.write_parquet(f"{OUT}/v_labels.parquet")

# ---------------- IC machinery
def ic_daily(df, ycol, resid="e_A", minn=30):
    out = []
    for (d,), g in df.filter(pl.col(ycol).is_not_null()).group_by(["date"]):
        if g.height < minn:
            continue
        r = stats.spearmanr(g[resid].to_numpy(), g[ycol].to_numpy()).statistic
        out.append((d, r, g.height))
    return pl.DataFrame(out, schema=["date", "ic", "n"], orient="row").sort("date")

def nw_se(x, lag=2):
    x = np.asarray(x, float); T = len(x); u = x - x.mean()
    s = u @ u / T
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / T
    return np.sqrt(s / T)

def summarize(icd, label_):
    w = (icd.with_columns(pl.col("date").dt.truncate("1w").alias("wk"))
         .group_by("wk").agg(pl.col("ic").mean(), pl.len().alias("nd")).filter(pl.col("nd") >= 4).sort("wk"))
    W = w["ic"].to_numpy()
    m = W.mean(); se = nw_se(W); t = m / se
    ph = icd.with_columns(((pl.col("date") - pl.lit(D0)).dt.total_days() % 7).alias("ph")).group_by("ph").agg(
        pl.col("ic").mean().alias("m"), (pl.col("ic").mean() / (pl.col("ic").std() / pl.len().sqrt())).alias("t")).sort("ph")
    yr = w.group_by(pl.col("wk").dt.year().alias("y")).agg(pl.col("ic").mean()).sort("y")
    r = dict(IC=float(m), se=float(se), t=float(t), n_weeks=len(W), n_days=icd.height, mean_n=float(icd["n"].mean()),
             phases_mean=[round(x, 5) for x in ph["m"].to_list()], phases_t=[round(x, 2) for x in ph["t"].to_list()],
             phases_pos=int((ph["m"] > 0).sum()), yearly=dict(zip(yr["y"].to_list(), [round(x, 5) for x in yr["ic"].to_list()])))
    print(label_, json.dumps(r))
    return r

C = CR.join(L.select("date", "symbol", "y7_g", "y7_n", "y7_strict", "y7_lag1", "y3_g", "y14_g"), on=["date", "symbol"], how="left")
assert C.height == CR.height
icd = ic_daily(C, "y7_g")
icd.write_parquet(f"{OUT}/v_ic_daily.parquet")
res["IC_main"] = summarize(icd, "IC_main")
res["IC_strict"] = summarize(ic_daily(C, "y7_strict"), "IC_strict")
res["IC_lag1"] = summarize(ic_daily(C, "y7_lag1"), "IC_lag1")
res["IC_h3"] = summarize(ic_daily(C, "y3_g"), "IC_h3")
res["IC_h14"] = summarize(ic_daily(C, "y14_g"), "IC_h14")
res["IC_unorth"] = summarize(ic_daily(C, "y7_g", resid="u_A"), "IC_unorth_uA")

# ---------------- metrics subset (pooled OLS with date FE on within-day-demeaned ranks)
def metrics_ic(rank_within_M):
    M = S.filter(pl.col("C")).filter(pl.all_horizontal([pl.col(c).is_not_null() for c in MC]))
    BIN = {"negF5", "inU"}
    if rank_within_M:
        def pr(c): return (pl.col(c).rank("average").over("date") - 0.5) / pl.len().over("date")
        M = M.with_columns(pr("A").alias("u_A"), *[(pl.col(z) if z in BIN else pr(z)).alias("u_" + z) for z in ZC + MC])
    else:  # ranks from C_d for A and Z, metrics ranked within C_d^M
        def pr(c): return (pl.col(c).rank("average").over("date") - 0.5) / pl.len().over("date")
        M = M.join(CR.drop("e_A"), on=["date", "symbol"], how="left").with_columns(*[pr(z).alias("u_" + z) for z in MC])
    U = ["u_A"] + ["u_" + z for z in ZC + MC]
    M = M.sort("date", "symbol").with_columns([(pl.col(c) - pl.col(c).mean().over("date")).alias(c) for c in U])
    X = M.select(U[1:]).to_numpy(); y = M["u_A"].to_numpy()
    bb, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ bb
    r2 = 1 - e @ e / (y @ y)
    M = M.with_columns(pl.Series("e_M", e)).join(L.select("date", "symbol", "y7_g"), on=["date", "symbol"], how="left")
    r = summarize(ic_daily(M, "y7_g", resid="e_M", minn=20), f"IC_metrics(rank_within_M={rank_within_M})")
    r["pooled_R2"] = float(r2); r["rows"] = M.height; r["days_with_M"] = M["date"].n_unique()
    return r
res["IC_metrics"] = metrics_ic(True)
res["IC_metrics_altrank"] = metrics_ic(False)

# ---------------- S5all variant (A_all), only report: residualize A_all ranks on same C_d u_Z
json.dump(res, open(f"{OUT}/v_results_ic.json", "w"), indent=1, default=str)
