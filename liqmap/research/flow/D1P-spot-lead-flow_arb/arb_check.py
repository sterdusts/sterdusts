"""Arbitration checks for D1P-spot-lead-flow (spec commit b3f95f8). Read-only on _run/_verify outputs and raw data.
Holdout guard: raw prices filtered to date<=2026-03-30 & in_holdout==False; funding ts<=2026-03-31 00:00 UTC;
signal days d<=2026-03-22 (from features). No file outside this _arb directory is written.
"""
import json, glob, os
from datetime import date, datetime, timedelta, timezone
import numpy as np, polars as pl
from scipy.stats import spearmanr, rankdata

RUN = "/home/user/data/flow/D1P-spot-lead-flow_run"
ARB = "/home/user/data/flow/D1P-spot-lead-flow_arb"
FLOW = "/home/user/data/flow"
OUT = f"{ARB}/arb_check.json"
assert not os.path.exists(OUT)
PRICE_END = date(2026, 3, 30)
FUND_END = int(datetime(2026, 3, 31, tzinfo=timezone.utc).timestamp() * 1000)
res = {}

F = pl.read_parquet(f"{RUN}/d1_features.parquet")
L = pl.read_parquet(f"{RUN}/labels.parquet")
assert F["date"].max() <= date(2026, 3, 22) and L["e"].max() <= date(2026, 3, 29)
Z = ["r1","r3","r5z","r7","r30","vx5","vxS5","sigma30","zrng","F5","negF5","dF5","P5","zP5","tbr1","lqv","beta","lage","zSH5","px","dbasis5","inU"]
UZ = ["u_" + c for c in Z]

# ---------------- A. IC recomputation (own code: scipy spearman, ISO weeks, NW lag 2)
def nw(x, lag=2):
    x = np.asarray(x, float); T = len(x); u = x - x.mean(); s = u @ u / T
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / T
    se = np.sqrt(s / T); return x.mean(), se, x.mean() / se

def ic(df, sc, lb, minn):
    rows = []
    for (d,), g in df.filter(pl.col(sc).is_not_null() & pl.col(lb).is_not_null()).group_by("date"):
        if g.height >= minn:
            rows.append((d, spearmanr(g[sc].to_numpy(), g[lb].to_numpy())[0]))
    ic = pl.DataFrame(rows, schema=["date", "ic"], orient="row").sort("date").filter(pl.col("ic").is_not_nan())
    iso = [d.isocalendar() for d in ic["date"]]
    ic = ic.with_columns(pl.Series("wk", [f"{a}-{b:02d}" for a, b, _ in iso]),
                         pl.Series("mon", [d - timedelta(days=d.weekday()) for d in ic["date"]]))
    W = ic.group_by("wk").agg(pl.col("ic").mean().alias("W"), pl.len().alias("n"), pl.col("mon").first()).filter(pl.col("n") >= 4).sort("mon")
    m, se, t = nw(W["W"].to_numpy())
    ph = ic.with_columns(((pl.col("date") - pl.lit(date(2022, 1, 1))).dt.total_days() % 7).alias("ph")).group_by("ph").agg(pl.col("ic").mean()).sort("ph")
    yr = W.group_by(pl.col("mon").dt.year().alias("y")).agg(pl.col("W").mean()).sort("y")
    return {"IC": m, "se": se, "t": t, "weeks": W.height, "days": ic.height, "phases_pos": int((ph["ic"] > 0).sum()),
            "phase_means": [round(v, 5) for v in ph["ic"]], "by_year": {str(a): round(b, 5) for a, b in yr.iter_rows()}}

C = F.filter(pl.col("in_C")).select("date", "symbol", "e_A", "u_A", *UZ).join(L.select("date", "symbol", "y7_g", "y7s_g", "y7_01_g", "y7lag_g"), on=["date", "symbol"], how="left")
res["IC_main"] = ic(C, "e_A", "y7_g", 30)
res["IC_strict"] = ic(C, "e_A", "y7s_g", 30)
res["IC_0100"] = ic(C, "e_A", "y7_01_g", 30)
res["IC_lag1"] = ic(C, "e_A", "y7lag_g", 30)
CM = F.filter(pl.col("in_CM")).select("date", "symbol", "e_A_M").join(L.select("date", "symbol", "y7_g"), on=["date", "symbol"], how="left")
res["IC_metrics"] = ic(CM, "e_A_M", "y7_g", 20)
res["n_0100_missing_entry_in_C_labelled"] = int(C.filter(pl.col("y7_g").is_not_null() & pl.col("y7_01_g").is_null()).height)
res["n_C_labelled"] = int(C["y7_g"].is_not_null().sum())

# ---------------- B. residual orthogonality of e_A (per day max |corr(e_A, u_Z)|)
mx = 0.0
for (d,), g in C.group_by("date"):
    E_ = g["e_A"].to_numpy(); X = g.select(UZ).to_numpy()
    mx = max(mx, float(np.abs(X.T @ (E_ - E_.mean())).max() / len(E_)))
res["max_abs_cov_eA_uZ"] = mx

# ---------------- C. naive raw-data label spot-check (500 random labelled C_d rows)
rng = np.random.default_rng(7)
samp = C.filter(pl.col("y7_g").is_not_null()).select("date", "symbol", "y7_g")
samp = samp[np.sort(rng.choice(samp.height, 500, replace=False))]
samp = samp.join(F.select("date", "symbol", "beta_60d"), on=["date", "symbol"])
syms = sorted(set(samp["symbol"]) | {"BTCUSDT"})
P = (pl.scan_parquet(f"{FLOW}/perp_daily.parquet").select("date", "symbol", "bar_status", "close", "funding_1d", "in_holdout")
     .filter(pl.col("symbol").is_in(syms) & (pl.col("date") <= PRICE_END) & ~pl.col("in_holdout")).collect())
px = {s: {r[0]: r[1] for r in g.filter(pl.col("bar_status") == "trade").select("date", "close").iter_rows()} for (s,), g in P.group_by("symbol")}
fnull = {s: {r[0] for r in g.filter(pl.col("funding_1d").is_null()).select("date").iter_rows()} for (s,), g in P.group_by("symbol")}
FR = pl.concat([pl.read_parquet(x, columns=["symbol", "calc_time", "funding_rate"]).filter(pl.col("symbol").is_in(syms))
                for x in sorted(glob.glob("/home/user/data/funding_*.parquet")) + [f"{FLOW}/perp_daily_aux_funding_backfill.parquet"]])
FR = FR.unique(["symbol", "calc_time"]).with_columns((pl.col("calc_time") // 1000 * 1000).alias("ts")).filter(pl.col("ts") <= FUND_END)
fr = {s: (g.sort("ts")["ts"].to_numpy(), g.sort("ts")["funding_rate"].to_numpy()) for (s,), g in FR.group_by("symbol")}
ms = lambda d: int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)
def fsum(s, a, b):
    t, v = fr[s]; m = (t > a) & (t <= b); return float(v[m].sum())
diffs = []
for d, s, y, b in samp.select("date", "symbol", "y7_g", "beta_60d").iter_rows():
    c = px[s]
    e = None
    for k in range(7, 0, -1):
        if d + timedelta(days=k) in c:
            e = d + timedelta(days=k); break
    assert e is not None
    a_, b_ = ms(d + timedelta(days=1)), ms(e + timedelta(days=1))
    assert b_ <= FUND_END
    hb = min(max(b, 0.5), 2.0)
    yy = (np.log(c[e] / c[d]) - fsum(s, a_, b_)) - hb * (np.log(px["BTCUSDT"][e] / px["BTCUSDT"][d]) - fsum("BTCUSDT", a_, b_))
    diffs.append(abs(yy - y))
res["label_spotcheck"] = {"n": len(diffs), "max_abs_diff": float(max(diffs))}

# ---------------- D. event regression point estimate (own FE code) on R = C & y7_n & ~contam_a & ~contam_b
R = (F.filter(pl.col("in_C")).select("date", "symbol", "ev_D1", "ev_C12", "ev_C3", "ev_C4", *UZ)
     .join(L.select("date", "symbol", "y7_n", "y7s_n", "contam_a", "contam_b", "c_rt"), on=["date", "symbol"]))
R = R.filter(pl.col("y7_n").is_not_null() & ~pl.col("contam_a") & ~pl.col("contam_b"))
def fe_beta(df, y):
    df = df.filter(pl.col(y).is_not_null())
    cols = [y, "ev_D1", "ev_C12", "ev_C3", "ev_C4"] + UZ
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in cols])
    df = df.with_columns([(pl.col(c) - pl.col(c).mean().over("date")).alias(c) for c in cols])
    X = df.select(cols[1:]).to_numpy(); b = np.linalg.lstsq(X, df[y].to_numpy(), rcond=None)[0]
    return dict(zip(["D1", "C12", "C3", "C4"], [float(v) for v in b[:4]]))
res["event_R_rows"] = R.height
res["event_D1_in_R"] = int(R["ev_D1"].sum())
res["event_beta"] = fe_beta(R, "y7_n")
res["event_beta_strict"] = fe_beta(R, "y7s_n")
res["mean_crt_D1_in_R"] = float(R.filter(pl.col("ev_D1"))["c_rt"].mean())

# ---------------- E. sleeve Sharpe from saved daily series
SL = pl.read_parquet(f"{RUN}/sleeve_daily.parquet")
sr = lambda x: float(np.mean(x) / np.std(x, ddof=1) * np.sqrt(365))
res["sleeve"] = {"days": SL.height, "first": str(SL["date"].min()), "last": str(SL["date"].max()),
                 "SR_005": sr(SL["ret_005"].to_numpy()), "SR_007": sr(SL["ret_007"].to_numpy()),
                 "corr_btc_005": float(np.corrcoef(SL["ret_005"].to_numpy(), SL["btc_ret"].to_numpy())[0, 1])}

# ---------------- F. holdout facts on 1h klines used
mx1 = 0
for f in glob.glob(f"{FLOW}/kl1h_um/*.parquet"):
    mx1 = max(mx1, int(pl.read_parquet(f, columns=["open_time"])["open_time"].max()))
res["kl1h_max_open_time_utc"] = str(datetime.fromtimestamp(mx1 / 1000, tz=timezone.utc))
json.dump(res, open(OUT, "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
