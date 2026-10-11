# Verifier part 2: 01:00 entry IC, event regression (E1-E5), G2 sleeve. Dev period only.
import json, glob, sys
from datetime import date, datetime, timezone
import numpy as np
import polars as pl
from scipy import stats

sys.path.insert(0, "/home/user/data/flow/D1P-spot-lead-flow_verify")
OUT = "/home/user/data/flow/D1P-spot-lead-flow_verify"
FLOW = "/home/user/data/flow"
assert json.load(open(f"{OUT}/v_g0.json"))["passed"]
D0, DEND, PMAX = date(2022, 1, 1), date(2026, 3, 22), date(2026, 3, 30)
EPOCH = date(1970, 1, 1)
FRMAX = int(datetime(2026, 3, 31, tzinfo=timezone.utc).timestamp() * 1000)
H1MAX = int(datetime(2026, 3, 30, 1, tzinfo=timezone.utc).timestamp() * 1000)
DAYMS, HMS = 86400000, 3600000
ZC = ["r1", "r3", "r5z", "r7", "r30", "vx5", "vxS5", "sig30", "zrng", "F5", "negF5", "dF5", "P5", "zP5", "tbr1", "lqv",
      "beta", "lage", "zSH5", "px", "dbasis5", "inU"]
G = ["D1", "C1", "C2", "C3", "C4"]
res = {}
S = pl.read_parquet(f"{OUT}/v_S.parquet")
CR = pl.read_parquet(f"{OUT}/v_C_ranks.parquet")
L = pl.read_parquet(f"{OUT}/v_labels.parquet")

# reuse loaders from test.py logic (re-implemented here minimal)
px = (pl.scan_parquet(f"{FLOW}/perp_daily.parquet").select("date", "symbol", "bar_status", "close", "funding_1d", "in_holdout", "ret_1d")
      .filter((pl.col("date") <= PMAX) & (~pl.col("in_holdout"))).collect())
assert px["date"].max() <= PMAX
px = px.with_columns(pl.when(pl.col("bar_status") == "trade").then(pl.col("close")).otherwise(None).alias("close"),
                     (pl.col("bar_status") == "trade").alias("trade"))
PX = {}
for (sym,), g in px.group_by(["symbol"]):
    g = g.sort("date"); dd = g["date"].to_physical().to_numpy()
    PX[sym] = dict(d0=dd[0], close=g["close"].fill_null(np.nan).to_numpy(), trade=g["trade"].to_numpy(),
                   fnull=g["funding_1d"].is_null().to_numpy())
ff = sorted(glob.glob("/home/user/data/funding_*.parquet")) + [f"{FLOW}/perp_daily_aux_funding_backfill.parquet"]
fr = pl.concat([pl.read_parquet(x).select("symbol", "calc_time", "funding_rate") for x in ff])
fr = fr.unique(["symbol", "calc_time"], keep="first").with_columns((pl.col("calc_time") // 1000 * 1000).alias("ts"))
fr = fr.filter(pl.col("ts") <= FRMAX)
FRD = {}
for (sym,), g in fr.group_by(["symbol"]):
    g = g.sort("ts"); FRD[sym] = (g["ts"].to_numpy(), np.r_[0.0, np.cumsum(g["funding_rate"].to_numpy())])
def fx(sym, a, b):
    assert b <= FRMAX
    if sym not in FRD: return 0.0
    ts, cs = FRD[sym]
    return cs[np.searchsorted(ts, b, "right")] - cs[np.searchsorted(ts, a, "right")]

# ---------------- 01:00 entry variant
K1 = {}
def k1(sym):
    if sym not in K1:
        k = (pl.read_parquet(f"{FLOW}/kl1h_um/{sym}.parquet").select("open_time", "open")
             .filter(pl.col("open_time") <= H1MAX))
        K1[sym] = dict(zip(k["open_time"].to_list(), k["open"].to_list()))
    return K1[sym]

def exit_idx(P, i0, h=7):
    for k in range(min(i0 + h, len(P["close"]) - 1), i0, -1):
        if P["trade"][k]: return k
    return -1

B = PX["BTCUSDT"]
rows = []
Lc = L.filter(pl.col("C") & pl.col("y7_g").is_not_null())
n_drop = 0
for sym, d, beta in zip(Lc["symbol"], Lc["di"], Lc["beta_60d"]):
    P = PX[sym]; i0 = int(d - P["d0"]); ie = exit_idx(P, i0); e = int(P["d0"]) + ie  # python int (avoid int32 overflow in ms)
    t_in = (d + 1) * DAYMS + HMS; t_out = (e + 1) * DAYMS + HMS
    K = k1(sym); KB = k1("BTCUSDT")
    pin = K.get(t_in)
    if pin is None or KB.get(t_in) is None:
        n_drop += 1; rows.append(np.nan); continue
    pout = K.get(t_out, P["close"][ie])
    bout = KB.get(t_out, B["close"][e - B["d0"]])
    hb = min(max(beta, 0.5), 2.0)
    y = (np.log(pout / pin) - fx(sym, t_in, t_out)) - hb * (np.log(bout / KB[t_in]) - fx("BTCUSDT", t_in, t_out))
    rows.append(y)
Lc = Lc.with_columns(pl.Series("y7_0100", np.array(rows, float)).fill_nan(None))
Lc.select("date","symbol","y7_0100").write_parquet(f"{OUT}/v_labels_0100.parquet")
res["0100_dropped_missing_entry"] = n_drop; res["0100_rows_start"] = Lc.height
from test_ic_lib import ic_daily, summarize  # noqa
C = CR.join(Lc.select("date", "symbol", "y7_0100"), on=["date", "symbol"], how="left")
res["IC_0100"] = summarize(ic_daily(C, "y7_0100"), "IC_0100")

# ---------------- event regression
E = S.filter(pl.col("C")).join(L.select("date", "symbol", "y7_n", "y7_g", "y7_strict", "c_rt", "y7_lag1"), on=["date", "symbol"], how="left")
E = E.join(CR.drop("e_A", "u_A"), on=["date", "symbol"], how="left")
# rule (a): raw trigger not kept (any group)
E = E.with_columns(pl.any_horizontal([pl.col("raw_" + g) & ~pl.col("kept_" + g) for g in G]).alias("_a"))
# rule (b): symbol had a kept event (any group) in [d-6, d-1]; kept events come from all of S (not only C)
kept_all = (S.filter(pl.any_horizontal([pl.col("kept_" + g) for g in G])).select("symbol", pl.col("date").to_physical().alias("kd")))
KD = {}
for s_, kd in zip(kept_all["symbol"], kept_all["kd"]): KD.setdefault(s_, []).append(kd)
KD = {k: np.sort(np.array(v)) for k, v in KD.items()}
def held(sym, d):
    a = KD.get(sym)
    if a is None: return False
    i = np.searchsorted(a, d - 6, "left")
    return bool(i < len(a) and a[i] <= d - 1)
E = E.with_columns(pl.Series("_b", [held(s_, d) for s_, d in zip(E["symbol"], E["date"].to_physical())]))
E = E.with_columns(pl.col("kept_D1").cast(pl.Float64).alias("I_D1"),
                   (pl.col("kept_C1") | pl.col("kept_C2")).cast(pl.Float64).alias("I_C12"),
                   pl.col("kept_C3").cast(pl.Float64).alias("I_C3"), pl.col("kept_C4").cast(pl.Float64).alias("I_C4"))
res["D1_kept_in_C_with_label"] = int(E.filter(pl.col("kept_D1") & pl.col("y7_n").is_not_null()).height)
Rall = E.filter(pl.col("y7_n").is_not_null())
res["R_counts"] = dict(start=Rall.height, a=int(Rall["_a"].sum()), b=int(Rall["_b"].sum()),
                       a_or_b=int((Rall["_a"] | Rall["_b"]).sum()),
                       D1_removed_by_b=int(Rall.filter(pl.col("kept_D1") & pl.col("_b")).height))
R = Rall.filter(~pl.col("_a") & ~pl.col("_b")).sort("date", "symbol")
res["R_counts"]["R"] = R.height
res["R_counts"]["D1_in_R"] = int(R["I_D1"].sum())
print(res["R_counts"])
IND = ["I_D1", "I_C12", "I_C3", "I_C4"]
UZ = ["u_" + z for z in ZC]

def block_ids(dates_phys):
    b0 = (date(2022, 1, 3) - EPOCH).days
    return np.floor((dates_phys - b0) / 14).astype(int)  # 2022-01-01/02 -> block -1

def reg(df, ycol="y7_n", nboot=5000, seed=0, ind=IND):
    df = df.filter(pl.col(ycol).is_not_null()).sort("date")
    cols = [ycol] + ind + UZ
    dm = df.with_columns([(pl.col(c) - pl.col(c).mean().over("date")).alias(c) for c in cols])
    y = dm[ycol].to_numpy(); X = dm.select(ind + UZ).to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    out = {k: float(v) for k, v in zip(ind, beta[:len(ind)])}
    out["n"] = df.height; out["n_D1"] = int(df["I_D1"].sum())
    if nboot:
        bid = block_ids(df["date"].to_physical().to_numpy())
        ub, inv = np.unique(bid, return_inverse=True)
        k = X.shape[1]
        XX = np.zeros((len(ub), k, k)); Xy = np.zeros((len(ub), k))
        for j in range(len(ub)):
            m = inv == j; XX[j] = X[m].T @ X[m]; Xy[j] = X[m].T @ y[m]
        rng = np.random.default_rng(seed)
        bs = np.empty((nboot, len(ind)))
        for r in range(nboot):
            cnt = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)).astype(float)
            A = np.tensordot(cnt, XX, 1); v = cnt @ Xy
            bs[r] = np.linalg.lstsq(A, v, rcond=None)[0][:len(ind)]
        out["n_blocks"] = len(ub)
        out["se"] = {k: float(bs[:, i].std(ddof=1)) for i, k in enumerate(ind)}
        out["p"] = {k: float((bs[:, i] <= 0).mean()) for i, k in enumerate(ind)}
        out["p_delta"] = {k: float(((bs[:, 0] - bs[:, i]) <= 0).mean()) for i, k in enumerate(ind) if i > 0}
    return out

res["E_main"] = reg(R); print("E_main", res["E_main"])
d1R = R.filter(pl.col("I_D1") == 1)
res["mean_crt_D1_in_R"] = float(d1R["c_rt"].mean())
res["mean_crt_D1_all_labelled"] = float(E.filter(pl.col("kept_D1") & pl.col("y7_n").is_not_null())["c_rt"].mean())
res["E3_yearly"] = {}
for y in [2022, 2023, 2024, 2025, 2026]:
    res["E3_yearly"][y] = reg(R.filter(pl.col("date").dt.year() == y), nboot=0)["I_D1"]
print("E3", res["E3_yearly"])
lo, hi = np.quantile(R["y7_n"].to_numpy(), [0.025, 0.975])
res["E4_winsor"] = reg(R.with_columns(pl.col("y7_n").clip(lo, hi)), nboot=0)["I_D1"]
# top-5 contributors: event y7_n minus same-day C_d mean y7_n (C_d rows with labels)
cm = E.filter(pl.col("y7_n").is_not_null()).group_by("date").agg(pl.col("y7_n").mean().alias("cmean"))
contrib = (d1R.join(cm, on="date").with_columns((pl.col("y7_n") - pl.col("cmean")).alias("x"))
           .group_by("symbol").agg(pl.col("x").sum()).sort("x", descending=True))
top5 = contrib.head(5)["symbol"].to_list()
res["E4_top5"] = top5
res["E4_drop_top5"] = reg(R.filter(~pl.col("symbol").is_in(top5)), nboot=0)["I_D1"]
# strict: rows of R with strict label (y7_n_strict = y7_strict - c_rt)
res["E5_strict"] = reg(R.with_columns((pl.col("y7_strict") - pl.col("c_rt")).alias("ys")), ycol="ys", nboot=0)["I_D1"]
# sensitivity: exempt kept event rows from rule (b)
R2 = Rall.filter(~pl.col("_a") & (~pl.col("_b") | pl.any_horizontal([pl.col("kept_" + g) for g in G])))
res["E_sens_b_exempt"] = reg(R2, nboot=2000); print("E_sens", res["E_sens_b_exempt"])
res["E_lag1"] = reg(R.with_columns((pl.col("y7_lag1") - pl.col("c_rt")).alias("yl")), ycol="yl", nboot=0)["I_D1"]
print({k: res[k] for k in ["mean_crt_D1_in_R", "mean_crt_D1_all_labelled", "E4_winsor", "E4_top5", "E4_drop_top5", "E5_strict", "E_lag1"]})

# ---------------- G2 sleeve
ev = (S.filter(pl.col("kept_D1")).select("date", "symbol", "A", "beta_60d", "qv_med30", "sig30").sort("date", pl.col("A"), descending=[False, True]))
NAV = 1e6
def slip(q): return 0.0002 if q >= 1e8 else (0.0005 if q >= 5e7 else 0.0010)
tdays = np.arange((date(2022, 1, 2) - EPOCH).days, (PMAX - EPOCH).days + 1)
t0 = tdays[0]
gross = np.zeros(len(tdays)); cost = {0.0003: np.zeros(len(tdays)), 0.0005: np.zeros(len(tdays)), 0.0007: np.zeros(len(tdays))}
notional = np.zeros(len(tdays)); npos = np.zeros(len(tdays), int)
open_pos = []  # (exit_day, w)
skipped = 0
def daily_ret(P, k):
    """simple return of day index k relative to last trade close before k (cross gaps); 0 if k not trade."""
    if not P["trade"][k]: return 0.0
    j = k - 1
    while j >= 0 and not P["trade"][j]: j -= 1
    if j < 0: return 0.0
    return P["close"][k] / P["close"][j] - 1
for d, sym, A, beta, qv, sg in ev.iter_rows():
    di = (d - EPOCH).days
    P = PX[sym]; i0 = di - P["d0"]
    w = min(0.005 / sg, 0.10, 0.005 * qv / NAV)
    hb = min(max(beta, 0.5), 2.0)
    open_now = [x for x in open_pos if x[0] >= di + 1]
    if len(open_now) >= 15 or sum(x[1] for x in open_now) + w > 1.5 + 1e-12:
        skipped += 1; continue
    ie = exit_idx(P, i0)
    if ie < 0:
        e = di + 1; days_ = []
    else:
        e = P["d0"] + ie; days_ = range(di + 1, e + 1)
    open_pos.append((e, w, hb))
    for t in days_:
        k = t - P["d0"]; kb = t - PX["BTCUSDT"]["d0"]
        a, b = t * DAYMS, (t + 1) * DAYMS
        r_i = daily_ret(P, k) - fx(sym, a, b)
        r_b = daily_ret(B, kb) - fx("BTCUSDT", a, b)
        gross[t - t0] += w * (r_i - hb * r_b)
        notional[t - t0] += w; npos[t - t0] += 1
    for fee in cost:
        c1 = (fee + slip(qv)) * w + fee * hb * w
        cost[fee][di + 1 - t0] += c1
        cost[fee][max(e, di + 1) - t0] += c1
assert tdays[-1] == (PMAX - EPOCH).days
btc = np.array([daily_ret(B, t - B["d0"]) for t in tdays])
res["G2"] = dict(n_events=ev.height, skipped_capacity=skipped, max_pos=int(npos.max()), mean_notional=float(notional.mean()))
for fee, cvec in cost.items():
    r = gross - cvec
    res["G2"][f"SR_{fee}"] = float(r.mean() / r.std(ddof=1) * np.sqrt(365))
    res["G2"][f"ann_{fee}"] = float(r.mean() * 365)
    res["G2"][f"corrBTC_{fee}"] = float(np.corrcoef(r, btc)[0, 1])
r5 = gross - cost[0.0005]
yrs = np.array([(EPOCH.toordinal() + int(t)) for t in tdays]); yy = np.array([date.fromordinal(x).year for x in yrs])
res["G2"]["SR_yearly_0.0005"] = {int(y): float(r5[yy == y].mean() / r5[yy == y].std(ddof=1) * np.sqrt(365)) for y in np.unique(yy)}
print(res["G2"])
json.dump(res, open(f"{OUT}/v_results_2.json", "w"), indent=1, default=str)
