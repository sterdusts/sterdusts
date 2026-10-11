"""D1P-spot-lead-flow -- registered test d1p_spot_lead_ic_v1 (labels are read ONLY here, after G0 passed).

Spec: /home/user/sterdusts/liqmap/research/flow_prereg_2026-10-10.json (commit b3f95f8), id = D1P-spot-lead-flow.
Holdout guard (explicit):
  * daily prices: perp_daily filtered at scan to date <= 2026-03-30 and in_holdout==False
  * funding settlements: ts <= 2026-03-31 00:00 UTC (asserted on every interval used)
  * 1h / 5m klines: open_time <= 2026-03-30 01:00 / 00:30 UTC
  * signal days d <= 2026-03-22 (h=7), d <= 2026-03-26 (h=3), d <= 2026-03-15 (h=14)
  * fwd_7d / fwd_7d_lv read only for d <= 2026-03-22, only to cross-check self-built labels
"""
import hashlib, json, os, sys, time, glob
from datetime import date, datetime, timedelta, timezone
import numpy as np
import polars as pl

T0 = time.time()
RUN = "/home/user/data/flow/D1P-spot-lead-flow_run"
FLOW = "/home/user/data/flow"
TRIALS = "/home/user/sterdusts/liqmap/research/trials.jsonl"
PRICE_END = date(2026, 3, 30)
FUND_END_MS = int(datetime(2026, 3, 31, tzinfo=timezone.utc).timestamp() * 1000)
D_END = date(2026, 3, 22)
NAV = 1e6
SEED = 20261010
NBOOT = 5000
OUTDIR = os.environ.get("D1P_OUT", RUN)
OUTS = ["results.json", "labels.parquet", "ic_daily.parquet", "ic_weekly.parquet", "sleeve_daily.parquet",
        "sleeve_positions.parquet", "matched_balance.parquet", "run_test.log.json"]
for f_ in OUTS:
    assert not os.path.exists(f"{OUTDIR}/{f_}"), f"refuse to overwrite {f_}"


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ================================================================== 0. registration + hash guards
rows = [json.loads(l) for l in open(TRIALS) if l.strip()]
reg = [r for r in rows if r.get("name") == "d1p_spot_lead_ic_v1"]
assert len(reg) == 1, "trial d1p_spot_lead_ic_v1 must be pre-registered exactly once"
G0 = json.load(open(f"{RUN}/prereg.json"))["G0"]
assert G0["passed"], "G0 failed -> must not read labels"
sha = lambda fn: hashlib.sha256(open(fn, "rb").read()).hexdigest()
assert sha(f"{RUN}/d1_features.parquet") == G0["features_sha256"], "features hash mismatch"
assert sha(f"{RUN}/build_features.py") == G0["script_sha256"], "feature script hash mismatch"
MERGE_C12 = G0["C12_merged"]
log("registration row found:", reg[0]["ts"], reg[0]["result"], "| C12 merged:", MERGE_C12)

# ================================================================== 1. features (built before labels)
F = pl.read_parquet(f"{RUN}/d1_features.parquet")
assert F["date"].max() <= D_END
Z = ["r1", "r3", "r5z", "r7", "r30", "vx5", "vxS5", "sigma30", "zrng", "F5", "negF5", "dF5", "P5", "zP5", "tbr1",
     "lqv", "beta", "lage", "zSH5", "px", "dbasis5", "inU"]
UZ = ["u_" + c for c in Z]
GROUPS = ["D1", "C1", "C2", "C3", "C4"]
JUDGE = (["C12"] if MERGE_C12 else ["C1", "C2"]) + ["C3", "C4"]

# ================================================================== 2. dev-only price grid + funding table
G = (pl.scan_parquet(f"{FLOW}/perp_daily.parquet")
     .select("date", "symbol", "bar_status", "open", "low", "close", "funding_1d", "in_holdout")
     .filter((pl.col("date") <= PRICE_END) & (~pl.col("in_holdout"))).collect())
assert G["date"].max() <= PRICE_END and not G["in_holdout"].any()
G = G.with_columns((pl.col("bar_status") == "trade").alias("trade")).sort("symbol", "date")
G = G.with_columns([pl.when(pl.col("trade")).then(pl.col(c)).otherwise(None).alias(c) for c in ["open", "low", "close"]])
G = G.with_columns(pl.col("close").log().alias("lc"),
                   pl.when(pl.col("trade")).then(pl.col("date")).otherwise(None).alias("_td"))
G = G.with_columns(pl.col("_td").forward_fill().over("symbol").alias("td_ff"),
                   pl.col("lc").forward_fill().over("symbol").alias("lc_ff"),
                   pl.col("funding_1d").is_null().cast(pl.Int32).cum_sum().over("symbol").alias("fnull_cum"))
G = G.with_columns((pl.col("lc") - pl.col("lc_ff").shift(1).over("symbol")).alias("rspan"))  # log ret since previous trade close
LAST = G.group_by("symbol").agg(pl.col("date").max().alias("gend"), pl.col("_td").drop_nulls().last().alias("td_last"),
                                pl.col("lc").drop_nulls().last().alias("lc_last"))
GK = G.select("symbol", "date", "trade", "lc", "td_ff", "lc_ff", "fnull_cum")
BTC = G.filter(pl.col("symbol") == "BTCUSDT").select("date", pl.col("lc").alias("lcB"), pl.col("fnull_cum").alias("fnB"),
                                                     pl.col("rspan").alias("rB"))
log("price grid", G.shape)

syms_needed = sorted(set(F["symbol"].unique().to_list()) | {"BTCUSDT"})
ff = sorted(glob.glob("/home/user/data/funding_*.parquet")) + [f"{FLOW}/perp_daily_aux_funding_backfill.parquet"]
FR = pl.concat([pl.scan_parquet(x).select("symbol", "calc_time", "funding_rate").filter(pl.col("symbol").is_in(syms_needed)).collect()
                for x in ff])
FR = FR.unique(["symbol", "calc_time"], keep="first").with_columns((pl.col("calc_time") // 1000 * 1000).alias("ts"))
FR = FR.filter(pl.col("ts") <= FUND_END_MS)          # holdout guard: no settlement after 2026-03-31 00:00
SYMCODE = {s: i for i, s in enumerate(syms_needed)}
FR = FR.with_columns(pl.col("symbol").replace_strict(SYMCODE, return_dtype=pl.Int64).alias("sc")).sort("sc", "ts")
BIG = 10 ** 13
FR_KEY = FR["sc"].to_numpy() * BIG + FR["ts"].to_numpy()
FR_CUM = np.r_[0.0, np.cumsum(FR["funding_rate"].to_numpy())]
assert FR["ts"].max() <= FUND_END_MS
log("funding rows", FR.height)


def fund_exec(sym, a_ms, b_ms):
    """sum of funding_rate with ts in (a, b] for each row (vectorised)"""
    sc = np.array([SYMCODE[s] for s in sym], dtype=np.int64)
    a_ms = np.asarray(a_ms, dtype=np.int64)
    b_ms = np.asarray(b_ms, dtype=np.int64)
    assert (b_ms <= FUND_END_MS).all(), "settlement interval beyond 2026-03-31 00:00"
    ia = np.searchsorted(FR_KEY, sc * BIG + a_ms, side="right")
    ib = np.searchsorted(FR_KEY, sc * BIG + b_ms, side="right")
    return FR_CUM[ib] - FR_CUM[ia]


def ms_of(dcol, hours=0, minutes=0):
    return (dcol.cast(pl.Datetime("ms")).cast(pl.Int64) + (hours * 3600 + minutes * 60) * 1000)


# ================================================================== 3. labels
def lv_exit(df, d0col, h, strict=False):
    """exit day e = last trade day in (d0, d0+h] (lv rule) or d0+h if strict; returns df with e, lc_e, lc_0"""
    x = df.with_columns((pl.col(d0col) + pl.duration(days=h)).alias("_x"))
    assert x["_x"].max() <= PRICE_END, "price date beyond 2026-03-30"
    x = x.join(GK.rename({"date": "_x", "td_ff": "_tdx", "lc_ff": "_lcx", "trade": "_trx", "lc": "_lcx_raw", "fnull_cum": "_fnx"}),
               on=["symbol", "_x"], how="left")
    x = x.join(GK.select("symbol", pl.col("date").alias(d0col), pl.col("lc").alias("lc_0"), pl.col("fnull_cum").alias("_fn0")),
               on=["symbol", d0col], how="left")
    x = x.join(LAST, on="symbol", how="left")
    beyond = pl.col("_x") > pl.col("gend")   # archive ended before d0+h (delisted): last trade close
    if strict:
        e = pl.when(pl.col("_trx").fill_null(False)).then(pl.col("_x")).otherwise(None)
        lce = pl.when(pl.col("_trx").fill_null(False)).then(pl.col("_lcx_raw")).otherwise(None)
    else:
        e = pl.when(beyond).then(pl.col("td_last")).otherwise(pl.col("_tdx"))
        lce = pl.when(beyond).then(pl.col("lc_last")).otherwise(pl.col("_lcx"))
    x = x.with_columns(e.alias("e"), lce.alias("lc_e"))
    x = x.with_columns(pl.when(pl.col("e") > pl.col(d0col)).then(pl.col("e")).otherwise(None).alias("e"))
    x = x.with_columns(pl.when(pl.col("e").is_not_null()).then(pl.col("lc_e")).otherwise(None).alias("lc_e"))
    # funding_1d null anywhere in (d0, e] -> label null
    x = x.join(GK.select("symbol", pl.col("date").alias("e"), pl.col("fnull_cum").alias("_fne")), on=["symbol", "e"], how="left")
    x = x.with_columns(((pl.col("_fne") - pl.col("_fn0")) > 0).alias("_fnull"))
    return x.drop([c for c in x.columns if c.startswith("_x") or c in ("_trx", "_tdx", "_lcx", "_lcx_raw", "_fnx", "gend", "td_last", "lc_last", "_fne", "_fn0")])


def add_btc(x, d0col):
    x = x.join(BTC.select(pl.col("date").alias(d0col), pl.col("lcB").alias("lcB0"), pl.col("fnB").alias("fnB0")), on=d0col, how="left")
    x = x.join(BTC.select(pl.col("date").alias("e"), pl.col("lcB").alias("lcBe"), pl.col("fnB").alias("fnBe")), on="e", how="left")
    return x


def finish(x, d0col, a_ms, b_ms, px_in=None, px_out=None, pxB_in=None, pxB_out=None, name="y"):
    """y_g = [fwd - F_i] - hb [fwdB - F_B]; F over (a, b]"""
    ok = x["e"].is_not_null().to_numpy() & ~x["_fnull"].fill_null(True).to_numpy() & (x["fnBe"] - x["fnB0"] == 0).fill_null(False).to_numpy()
    sym = x["symbol"].to_list()
    a = a_ms.to_numpy()
    b = b_ms.to_numpy()
    a_ = np.where(ok, a, 0)
    b_ = np.where(ok, b, 0)
    Fi = fund_exec(sym, a_, b_)
    Fb = fund_exec(["BTCUSDT"] * len(sym), a_, b_)
    fwd = (x["lc_e"] - x["lc_0"]).to_numpy() if px_in is None else np.log(px_out / px_in)
    fwdB = (x["lcBe"] - x["lcB0"]).to_numpy() if pxB_in is None else np.log(pxB_out / pxB_in)
    hb = np.clip(x["beta_60d"].to_numpy(), 0.5, 2.0)
    yg = (fwd - Fi) - hb * (fwdB - Fb)
    yg = np.where(ok, yg, np.nan)
    return x.select("date", "symbol").with_columns(pl.Series(name, yg))


L = F.select("date", "symbol", "beta_60d", "qv_med30", "in_C", "in_CM", "in_Ca", "in_S").with_columns(
    pl.col("beta_60d").clip(0.5, 2.0).alias("hb"))
slip = (pl.when(pl.col("qv_med30") >= 1e8).then(0.0002).when(pl.col("qv_med30") >= 5e7).then(0.0005)
        .when(pl.col("qv_med30") >= 2e7).then(0.0010).otherwise(None))
L = L.with_columns(slip.alias("slip"))
L = L.with_columns((2 * (0.0005 + pl.col("slip")) + 2 * 0.0005 * pl.col("hb")).alias("c_rt"))

lab = {}
# --- main: lv, h=7, close_d entry, funding ((d+1)00:00, (e+1)00:00]
x = add_btc(lv_exit(L, "date", 7), "date")
assert x.height == L.height
lab["y7_g"] = finish(x, "date", ms_of(x["date"] + timedelta(days=1)), ms_of(x["e"].fill_null(x["date"]) + timedelta(days=1)), name="y7_g")
X7 = x.select("date", "symbol", "e")
assert X7["e"].drop_nulls().max() <= date(2026, 3, 29)
# --- strict (fwd_7d non-null, e = d+7)
xs = add_btc(lv_exit(L, "date", 7, strict=True), "date")
lab["y7s_g"] = finish(xs, "date", ms_of(xs["date"] + timedelta(days=1)), ms_of(xs["e"].fill_null(xs["date"]) + timedelta(days=1)), name="y7s_g")
# --- lag-1: entry close_{d+1}, exit lv over (d+1, d+8], funding ((d+2)00:00, (e'+1)00:00]
Ll = L.with_columns((pl.col("date") + pl.duration(days=1)).alias("d1"))
xl = add_btc(lv_exit(Ll, "d1", 7), "d1")
lab["y7lag_g"] = finish(xl.with_columns(pl.when(pl.col("lc_0").is_null()).then(None).otherwise(pl.col("e")).alias("e")),  # d+1 must be a trade day
                       "d1", ms_of(xl["d1"] + timedelta(days=1)), ms_of(xl["e"].fill_null(xl["d1"]) + timedelta(days=1)), name="y7lag_g")
# --- h=3 / h=14 (report only); d <= 2026-03-30 - h - 1
for h in (3, 14):
    dmax = PRICE_END - timedelta(days=h + 1)
    Lh = L.filter(pl.col("date") <= dmax)
    xh = add_btc(lv_exit(Lh, "date", h), "date")
    tmp = finish(xh, "date", ms_of(xh["date"] + timedelta(days=1)), ms_of(xh["e"].fill_null(xh["date"]) + timedelta(days=1)), name=f"y{h}_g")
    L = L.join(tmp, on=["date", "symbol"], how="left")
for k in ("y7_g", "y7s_g", "y7lag_g"):
    L = L.join(lab[k], on=["date", "symbol"], how="left")
L = L.with_columns(pl.col(c).fill_nan(None) for c in ["y7_g", "y7s_g", "y7lag_g", "y3_g", "y14_g"])
L = L.join(X7, on=["date", "symbol"], how="left")

# --- -25% catastrophic stop (report only): first day t in (d, e] with low_t <= 0.75 close_d -> exit at min(open_t, 0.75 close_d)
ex = (L.select("date", "symbol", "e").filter(pl.col("e").is_not_null())
      .with_columns(pl.int_ranges(1, 8).alias("k")).explode("k")
      .with_columns((pl.col("date") + pl.duration(days=pl.col("k"))).alias("t")).filter(pl.col("t") <= pl.col("e"))
      .join(G.select("symbol", pl.col("date").alias("t"), "low", "open"), on=["symbol", "t"], how="left")
      .join(GK.select("symbol", "date", pl.col("lc").alias("lc0")), on=["symbol", "date"], how="left")
      .filter(pl.col("low").log() <= pl.col("lc0") + np.log(0.75))
      .sort("k").group_by("date", "symbol").agg(pl.col("t").first().alias("t_stop"), pl.col("open").first().alias("open_stop"),
                                                pl.col("lc0").first()))
ex = ex.with_columns(pl.min_horizontal(pl.col("open_stop").log(), pl.col("lc0") + np.log(0.75)).alias("lstop"))
Ls = L.select("date", "symbol", "beta_60d", "e").join(ex, on=["date", "symbol"], how="inner")
Ls = Ls.with_columns(pl.col("t_stop").alias("e2"))
Ls = Ls.join(BTC.select(pl.col("date"), pl.col("lcB").alias("lcB0")), on="date", how="left").join(
    BTC.select(pl.col("date").alias("e2"), pl.col("lcB").alias("lcBe")), on="e2", how="left")
FiS = fund_exec(Ls["symbol"].to_list(), ms_of(Ls["date"] + timedelta(days=1)).to_numpy(), ms_of(Ls["e2"] + timedelta(days=1)).to_numpy())
FbS = fund_exec(["BTCUSDT"] * Ls.height, ms_of(Ls["date"] + timedelta(days=1)).to_numpy(), ms_of(Ls["e2"] + timedelta(days=1)).to_numpy())
hbS = np.clip(Ls["beta_60d"].to_numpy(), 0.5, 2)
yS = ((Ls["lstop"] - Ls["lc0"]).to_numpy() - FiS) - hbS * ((Ls["lcBe"] - Ls["lcB0"]).to_numpy() - FbS)
L = L.join(Ls.select("date", "symbol", pl.Series("y7stop_override", yS), "t_stop"), on=["date", "symbol"], how="left")
L = L.with_columns(pl.when(pl.col("y7stop_override").is_not_null() & pl.col("y7_g").is_not_null())
                   .then(pl.col("y7stop_override")).otherwise(pl.col("y7_g")).alias("y7stop_g")).drop("y7stop_override")
log("daily labels built")


# --- intraday-entry variants (01:00 from kl1h_um; 00:30 from kl5m for pool coins)
def intraday_variant(Lc, bars, off_h, off_m, tag):
    """bars: symbol, open_time(ms), open. entry = open of bar at (d+1)+off; exit = open at (e+1)+off else close_e."""
    x = Lc.select("date", "symbol", "beta_60d", "e").filter(pl.col("e").is_not_null())
    x = x.with_columns(ms_of(pl.col("date") + pl.duration(days=1), off_h, off_m).alias("t_in"),
                       ms_of(pl.col("e") + pl.duration(days=1), off_h, off_m).alias("t_out"))
    b = bars.select("symbol", "open_time", "open")
    x = x.join(b.rename({"open_time": "t_in", "open": "p_in"}), on=["symbol", "t_in"], how="left")
    x = x.join(b.rename({"open_time": "t_out", "open": "p_out"}), on=["symbol", "t_out"], how="left")
    bb = b.filter(pl.col("symbol") == "BTCUSDT").drop("symbol")
    x = x.join(bb.rename({"open_time": "t_in", "open": "pB_in"}), on="t_in", how="left")
    x = x.join(bb.rename({"open_time": "t_out", "open": "pB_out"}), on="t_out", how="left")
    x = x.join(GK.select("symbol", pl.col("date").alias("e"), pl.col("lc").alias("lc_e")), on=["symbol", "e"], how="left")
    x = x.join(BTC.select(pl.col("date").alias("e"), pl.col("lcB").alias("lcBe")), on="e", how="left")
    n_all = x.height
    n_miss_in = int(x["p_in"].is_null().sum())
    n_miss_B = int((x["p_in"].is_not_null() & x["pB_in"].is_null()).sum())
    x = x.filter(pl.col("p_in").is_not_null() & pl.col("pB_in").is_not_null())
    x = x.with_columns(pl.coalesce(pl.col("p_out"), pl.col("lc_e").exp()).alias("p_out"),
                       pl.coalesce(pl.col("pB_out"), pl.col("lcBe").exp()).alias("pB_out"))
    assert x["t_out"].max() <= ms_of(pl.Series([PRICE_END]), off_h, off_m)[0]
    Fi = fund_exec(x["symbol"].to_list(), x["t_in"].to_numpy(), x["t_out"].to_numpy())
    Fb = fund_exec(["BTCUSDT"] * x.height, x["t_in"].to_numpy(), x["t_out"].to_numpy())
    hb = np.clip(x["beta_60d"].to_numpy(), 0.5, 2)
    y = (np.log(x["p_out"].to_numpy() / x["p_in"].to_numpy()) - Fi) - hb * (np.log(x["pB_out"].to_numpy() / x["pB_in"].to_numpy()) - Fb)
    out = x.select("date", "symbol").with_columns(pl.Series(tag, y))
    return out, {"rows_with_label": n_all, "dropped_missing_entry_bar": n_miss_in, "dropped_missing_btc_bar": n_miss_B}


Lnn = L.filter(pl.col("y7_g").is_not_null())     # same null pattern (incl. funding-null rule) as the main label
kl1 = sorted(glob.glob(f"{FLOW}/kl1h_um/*.parquet"))
need1 = set(L.filter("in_C")["symbol"].unique().to_list()) | {"BTCUSDT"}
have1 = {os.path.basename(p_)[:-8] for p_ in kl1}
miss1 = sorted(need1 - have1)
LIM1 = int(datetime(2026, 3, 30, 1, tzinfo=timezone.utc).timestamp() * 1000)
B1 = pl.concat([pl.scan_parquet(p_).select("symbol", "open_time", "open")
                .filter((pl.col("open_time") <= LIM1) & ((pl.col("open_time") // 3600000) % 24 == 1)).collect() for p_ in kl1])
y01, info01 = intraday_variant(Lnn, B1, 1, 0, "y7_01_g")
info01["symbols_missing_kl1h_file"] = miss1
L = L.join(y01, on=["date", "symbol"], how="left")
del B1
LIM5 = int(datetime(2026, 3, 30, 0, 30, tzinfo=timezone.utc).timestamp() * 1000)
k5 = [f"/home/user/data/kl5m/{s}.parquet" for s in sorted(need1) if os.path.exists(f"/home/user/data/kl5m/{s}.parquet")]
B5 = pl.concat([pl.scan_parquet(p_).select("symbol", "open_time", "open")
                .with_columns(pl.when(pl.col("open_time") > 10**14).then(pl.col("open_time") // 1000).otherwise(pl.col("open_time")).alias("open_time"))
                .filter((pl.col("open_time") <= LIM5) & ((pl.col("open_time") // 60000) % 1440 == 30)).collect() for p_ in k5])
y0030, info0030 = intraday_variant(Lnn.filter(pl.col("symbol").is_in([os.path.basename(p_)[:-8] for p_ in k5])), B5, 0, 30, "y7_0030_g")
info0030["n_kl5m_symbols"] = len(k5)
L = L.join(y0030, on=["date", "symbol"], how="left")
del B5
for c in ["y7_g", "y7s_g", "y7lag_g", "y3_g", "y14_g", "y7stop_g", "y7_01_g", "y7_0030_g"]:
    L = L.with_columns((pl.col(c) - pl.col("c_rt")).alias(c.replace("_g", "_n")))
log("intraday variants built", info01, info0030)

# cross-check self-built labels against perp_daily fwd_7d_lv / fwd_7d (read only for d <= 2026-03-22)
chk = (pl.scan_parquet(f"{FLOW}/perp_daily.parquet").select("date", "symbol", "fwd_7d_lv", "fwd_7d")
       .filter(pl.col("date") <= D_END).collect())
cc = X7.join(chk, on=["date", "symbol"], how="left").join(GK.select("symbol", "date", pl.col("lc").alias("lc0")), on=["symbol", "date"], how="left").join(
    GK.select("symbol", pl.col("date").alias("e"), pl.col("lc").alias("lce")), on=["symbol", "e"], how="left")
diff = (cc["lce"] - cc["lc0"] - cc["fwd_7d_lv"]).abs()
xcheck = {"n": int(diff.is_not_null().sum()), "max_abs_diff_vs_fwd_7d_lv": float(diff.max()),
          "null_mismatch": int((cc["e"].is_null() != cc["fwd_7d_lv"].is_null()).sum())}
log("label cross-check", xcheck)
del chk, cc

# ================================================================== 4. data assembly
D = F.join(L.select("date", "symbol", "c_rt", "hb", "slip", "e", *[c for c in L.columns if c.startswith("y")]), on=["date", "symbol"], how="left")
D = D.sort("date", "symbol")
assert D.filter(pl.col("in_C") & pl.col("y7_g").is_not_null())["e"].max() <= date(2026, 3, 29)


# ================================================================== 5. IC machinery
def spearman_day(df, score, label, mask, min_rows):
    d = df.filter(pl.col(mask) & pl.col(label).is_not_null() & pl.col(score).is_not_null()).select("date", score, label)
    d = d.with_columns(pl.col(score).rank("average").over("date").alias("_rs"), pl.col(label).rank("average").over("date").alias("_rl"),
                       pl.len().over("date").alias("_n"))
    ic = d.group_by("date").agg(pl.corr("_rs", "_rl").alias("ic"), pl.col("_n").first().alias("n")).sort("date")
    return ic.filter((pl.col("n") >= min_rows) & pl.col("ic").is_not_nan() & pl.col("ic").is_not_null())


def nw_t(x, lag=2):
    x = np.asarray(x, float)
    T = len(x)
    m = x.mean()
    u = x - m
    s = u @ u / T
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / T
    se = np.sqrt(s / T)
    return m, se, m / se


def ic_stats(ic):
    if ic.height == 0:
        return {"IC": float("nan"), "t": float("nan")}, None
    w = (ic.with_columns(pl.col("date").dt.iso_year().alias("iy"), pl.col("date").dt.week().alias("iw"),
                         (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday() - 1)).alias("monday"))
         .group_by("iy", "iw").agg(pl.col("ic").mean().alias("W"), pl.len().alias("nd"), pl.col("monday").first())
         .filter(pl.col("nd") >= 4).sort("monday"))
    m, se, t = nw_t(w["W"].to_numpy())
    ph = ic.with_columns(((pl.col("date") - pl.lit(date(2022, 1, 1))).dt.total_days() % 7).alias("ph")).group_by("ph").agg(
        pl.col("ic").mean().alias("m"), pl.col("ic").std().alias("s"), pl.len().alias("n")).sort("ph")
    ph = ph.with_columns((pl.col("m") / (pl.col("s") / pl.col("n").sqrt())).alias("t"))
    yr = w.with_columns(pl.col("monday").dt.year().alias("y")).group_by("y").agg(pl.col("W").mean().alias("IC"), pl.len().alias("weeks")).sort("y")
    out = {"IC": float(m), "NW_se": float(se), "t": float(t), "n_weeks": w.height, "n_days": ic.height,
           "mean_daily_n": float(ic["n"].mean()),
           "phases": [{"phase": int(r[0]), "mean": float(r[1]), "t": float(r[4]), "n": int(r[3])} for r in ph.iter_rows()],
           "phases_pos": int((ph["m"] > 0).sum()),
           "by_year": {str(r[0]): {"IC": float(r[1]), "weeks": int(r[2])} for r in yr.iter_rows()}}
    return out, w


RES = {"trial": "d1p_spot_lead_ic_v1", "G0": {k: G0[k] for k in ["C_d_median", "frac_days_C_ge_40", "ols_R2_median", "events_kept", "C12_merged", "passed", "MDE"]},
       "label_crosscheck": xcheck, "intraday_01": info01, "intraday_0030": info0030}
IC = {}
ic_main = spearman_day(D, "e_A", "y7_g", "in_C", 30)
IC["main"], W_main = ic_stats(ic_main)
ic_daily_all = ic_main.rename({"ic": "ic_main", "n": "n_main"})
for tag, sc, lb, mk, mn in [("metrics", "e_A_M", "y7_g", "in_CM", 20), ("entry_0100", "e_A", "y7_01_g", "in_C", 30),
                            ("lag1", "e_A", "y7lag_g", "in_C", 30), ("strict", "e_A", "y7s_g", "in_C", 30),
                            ("h3", "e_A", "y3_g", "in_C", 30), ("h14", "e_A", "y14_g", "in_C", 30),
                            ("S5all", "e_A_all", "y7_g", "in_Ca", 30), ("entry_0030_pool", "e_A", "y7_0030_g", "in_C", 20),
                            ("raw_A_noorth", "u_A", "y7_g", "in_C", 30), ("main_net", "e_A", "y7_n", "in_C", 30),
                            ("stop25", "e_A", "y7stop_g", "in_C", 30)]:
    ic_ = spearman_day(D, sc, lb, mk, mn)
    IC[tag], _ = ic_stats(ic_)
    ic_daily_all = ic_daily_all.join(ic_.rename({"ic": f"ic_{tag}", "n": f"n_{tag}"}), on="date", how="full", coalesce=True)
# stratified ICs (report only)
D = D.with_columns(pl.col("btc_r60").gt(0).alias("btc60_pos"))
for tag, cond in [("btc60_pos", pl.col("btc60_pos").fill_null(False)), ("btc60_neg", ~pl.col("btc60_pos").fill_null(True)),
                  ("drop5coins", ~pl.col("symbol").is_in(["BCHUSDT", "AXSUSDT", "XRPUSDT", "NEARUSDT", "APTUSDT"])),
                  ("inU", pl.col("inU") == 1), ("not_inU", pl.col("inU") == 0)]:
    Dm = D.with_columns((pl.col("in_C") & cond).alias("_m"))
    IC["strat_" + tag], _ = ic_stats(spearman_day(Dm, "e_A", "y7_g", "_m", 20))
RES["IC"] = IC
log("IC main", {k: IC["main"][k] for k in ["IC", "t", "n_weeks", "phases_pos"]}, IC["main"]["by_year"])
for k in IC:
    log(f"  IC {k}: {IC[k].get('IC'):.4f} t={IC[k].get('t'):.2f}")

# ================================================================== 6. event regression
# contamination: (a) raw trigger (any group) not kept; (b) coin holds a kept event (any group) from [d-6, d-1]
D = D.sort("symbol", "date")
D = D.with_columns(pl.any_horizontal([pl.col(f"raw_{g}") & ~pl.col(f"ev_{g}") for g in GROUPS]).alias("contam_a"),
                   pl.any_horizontal([pl.col(f"ev_{g}") for g in GROUPS]).alias("ev_any"),
                   pl.any_horizontal([pl.col(f"raw_{g}") for g in GROUPS]).alias("raw_any"))
D = D.with_columns(pl.when(pl.col("ev_any")).then(pl.col("date")).otherwise(None).shift(1).forward_fill().over("symbol").alias("_last_ev"))
# rows of F are E rows only; a kept event can only occur on S rows (subset of E) so the per-symbol scan over E rows is complete
D = D.with_columns(((pl.col("date") - pl.col("_last_ev")).dt.total_days() <= 6).fill_null(False).alias("contam_b"))
# D1' repeat flag: another kept D1' event of the same coin in [d-14, d-1]
D = D.with_columns(pl.when(pl.col("ev_D1")).then(pl.col("date")).otherwise(None).shift(1).forward_fill().over("symbol").alias("_last_d1"))
D = D.with_columns((pl.col("ev_D1") & ((pl.col("date") - pl.col("_last_d1")).dt.total_days() <= 14).fill_null(False)).alias("ev_D1_repeat"))
# raw trigger of any group within [d-6, d] (for matched-control pool)
D = D.with_columns(pl.col("raw_any").cast(pl.Int32).rolling_sum_by("date", window_size="7d").over("symbol").alias("_raw7"))
D = D.sort("date", "symbol")
D = D.with_columns(((pl.col("date") - pl.lit(date(2022, 1, 3))).dt.total_days() // 14).alias("block"))


def event_reg(Rdf, ycol, dummies, nboot=NBOOT, seed=SEED, winsor=None):
    d = Rdf.filter(pl.col(ycol).is_not_null()).select("date", "block", ycol, *dummies, *UZ)
    d = d.with_columns([pl.col(c).cast(pl.Float64) for c in dummies])
    if winsor is not None:
        lo, hi = d[ycol].quantile(winsor[0]), d[ycol].quantile(winsor[1])
        d = d.with_columns(pl.col(ycol).clip(lo, hi))
    cols = [ycol] + dummies + UZ
    d = d.with_columns([(pl.col(c) - pl.col(c).mean().over("date")).alias(c) for c in cols])
    y = d[ycol].to_numpy()
    X = d.select(dummies + UZ).to_numpy()
    keep = np.abs(X).sum(0) > 0
    ndum = len(dummies)
    X = X[:, keep]
    kept_names = [c for c, k in zip(dummies + UZ, keep) if k]
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    blk = d["block"].to_numpy()
    ub, bi = np.unique(blk, return_inverse=True)
    K = X.shape[1]
    XtX = np.zeros((len(ub), K, K))
    Xty = np.zeros((len(ub), K))
    for j in range(len(ub)):
        m = bi == j
        XtX[j] = X[m].T @ X[m]
        Xty[j] = X[m].T @ y[m]
    rng = np.random.default_rng(seed)
    cnt = np.stack([np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)) for _ in range(nboot)]).astype(float)
    A = np.einsum("bj,jkl->bkl", cnt, XtX)
    v = np.einsum("bj,jk->bk", cnt, Xty)
    A += np.eye(K)[None] * 1e-12
    bb = np.linalg.solve(A, v[..., None])[..., 0]
    out = {"n": int(len(y)), "n_blocks": int(len(ub)), "nboot": nboot}
    draws = {}
    for c in dummies:
        if c in kept_names:
            i = kept_names.index(c)
            out[c] = {"beta": float(beta[i]), "boot_se": float(bb[:, i].std()), "p_one_sided": float((bb[:, i] <= 0).mean()),
                      "n_events": int(Rdf.filter(pl.col(ycol).is_not_null())[c].cast(pl.Int64).sum())}
            draws[c] = bb[:, i]
        else:
            out[c] = {"beta": float("nan"), "n_events": 0}
    return out, draws


R0 = D.filter(pl.col("in_C") & ~pl.col("contam_a") & ~pl.col("contam_b"))
DUMS = ["ev_D1"] + ["ev_" + k for k in JUDGE]


def full_event(Rdf, ycol, tag, dums=DUMS, **kw):
    o, dr = event_reg(Rdf, ycol, dums, **kw)
    b1 = o["ev_D1"]["beta"]
    for k in dums[1:]:
        if k in dr and "ev_D1" in dr:
            dd = dr["ev_D1"] - dr[k]
            o[k]["delta"] = float(b1 - o[k]["beta"])
            o[k]["delta_p_one_sided"] = float((dd <= 0).mean())
    o["mean_crt_D1"] = float(Rdf.filter(pl.col("ev_D1") & pl.col(ycol).is_not_null())["c_rt"].mean())
    return o


EV = {}
EV["main"] = full_event(R0, "y7_n", "main")
RES["R_sample"] = {"rows_in_C_with_label": int(D.filter(pl.col("in_C") & pl.col("y7_n").is_not_null()).height),
                   "removed_contam_a": int(D.filter(pl.col("in_C") & pl.col("y7_n").is_not_null() & pl.col("contam_a")).height),
                   "removed_contam_b": int(D.filter(pl.col("in_C") & pl.col("y7_n").is_not_null() & ~pl.col("contam_a") & pl.col("contam_b")).height),
                   "R_rows": int(R0.filter(pl.col("y7_n").is_not_null()).height),
                   "D1_kept_total": int(D["ev_D1"].sum()), "D1_kept_in_C": int(D.filter(pl.col("in_C") & pl.col("ev_D1")).height),
                   "D1_kept_in_C_labelled": int(D.filter(pl.col("in_C") & pl.col("ev_D1") & pl.col("y7_n").is_not_null()).height),
                   "D1_in_R": int(R0.filter(pl.col("ev_D1") & pl.col("y7_n").is_not_null()).height),
                   "D1_kept_removed_by_contam_b": int(D.filter(pl.col("in_C") & pl.col("ev_D1") & pl.col("y7_n").is_not_null() & pl.col("contam_b")).height)}
log("R", RES["R_sample"])
log("EV main", json.dumps(EV["main"]))
# sensitivity (report only): kept-event rows exempt from contamination rule (b) (literal spec applies (b) to all rows)
EV["sens_events_exempt_from_contam_b"] = full_event(D.filter(pl.col("in_C") & ~pl.col("contam_a") & (~pl.col("contam_b") | pl.col("ev_any"))), "y7_n", "alt")
# separate C1/C2 (report only when merged)
EV["sep_C1_C2_report"] = full_event(R0, "y7_n", "sep", dums=["ev_D1", "ev_C1", "ev_C2", "ev_C3", "ev_C4"])
# E3 per year
EV["by_year"] = {}
for yv in [2022, 2023, 2024, 2025, 2026]:
    EV["by_year"][str(yv)] = full_event(R0.filter(pl.col("date").dt.year() == yv), "y7_n", f"y{yv}", nboot=1000)
# E4 tails
EV["winsor_2.5_97.5"] = full_event(R0, "y7_n", "w", winsor=(0.025, 0.975))
cm = D.filter(pl.col("in_C") & pl.col("y7_n").is_not_null()).group_by("date").agg(pl.col("y7_n").mean().alias("cmean"))
contrib = (R0.filter(pl.col("ev_D1") & pl.col("y7_n").is_not_null()).join(cm, on="date")
           .with_columns((pl.col("y7_n") - pl.col("cmean")).alias("x")).group_by("symbol").agg(pl.col("x").sum().alias("contrib"), pl.len().alias("n"))
           .sort("contrib", descending=True))
top5 = contrib.head(5)["symbol"].to_list()
EV["drop_top5_coins"] = full_event(R0.filter(~pl.col("symbol").is_in(top5)), "y7_n", "dt5")
EV["drop_top5_coins"]["dropped"] = [{"symbol": r[0], "contrib": float(r[1]), "n": int(r[2])} for r in contrib.head(5).iter_rows()]
# E5 strict / P4 timing variants / horizons / stop
EV["strict"] = full_event(R0, "y7s_n", "strict")
EV["entry_0100"] = full_event(R0, "y7_01_n", "e01")
EV["lag1"] = full_event(R0, "y7lag_n", "lag1")
EV["h3"] = full_event(R0, "y3_n", "h3")
EV["h14"] = full_event(R0, "y14_n", "h14")
EV["stop25"] = full_event(R0, "y7stop_n", "stop")
EV["entry_0030_pool"] = full_event(R0, "y7_0030_n", "e0030")
EV["gross_y7_g"] = full_event(R0, "y7_g", "gross")
# stratifications (report only)
EV["strat_btc60_pos"] = full_event(R0.filter(pl.col("btc60_pos").fill_null(False)), "y7_n", "b+", nboot=1000)
EV["strat_btc60_neg"] = full_event(R0.filter(~pl.col("btc60_pos").fill_null(True)), "y7_n", "b-", nboot=1000)
EV["strat_drop5coins"] = full_event(R0.filter(~pl.col("symbol").is_in(["BCHUSDT", "AXSUSDT", "XRPUSDT", "NEARUSDT", "APTUSDT"])), "y7_n", "d5", nboot=1000)
Rs = R0.with_columns((pl.col("ev_D1") & pl.col("ev_D1_repeat")).alias("ev_D1_rep"), (pl.col("ev_D1") & ~pl.col("ev_D1_repeat")).alias("ev_D1_first"),
                     (pl.col("ev_D1") & (pl.col("inU") == 1)).alias("ev_D1_inU"), (pl.col("ev_D1") & (pl.col("inU") == 0)).alias("ev_D1_notinU"),
                     (pl.col("ev_D1") & (pl.col("slip") == 0.0002)).alias("ev_D1_qv_hi"), (pl.col("ev_D1") & (pl.col("slip") == 0.0005)).alias("ev_D1_qv_mid"),
                     (pl.col("ev_D1") & (pl.col("slip") == 0.0010)).alias("ev_D1_qv_lo"))
EV["strat_repeat14"] = event_reg(Rs, "y7_n", ["ev_D1_rep", "ev_D1_first"] + DUMS[1:], nboot=1000)[0]
EV["strat_inU"] = event_reg(Rs, "y7_n", ["ev_D1_inU", "ev_D1_notinU"] + DUMS[1:], nboot=1000)[0]
EV["strat_qv_tier"] = event_reg(Rs, "y7_n", ["ev_D1_qv_hi", "ev_D1_qv_mid", "ev_D1_qv_lo"] + DUMS[1:], nboot=1000)[0]
RES["EV"] = EV
log("EV variants", {k: (round(v["ev_D1"]["beta"], 5) if "ev_D1" in v else None) for k, v in EV.items() if k != "by_year"},
    {k: round(v["ev_D1"]["beta"], 5) for k, v in EV["by_year"].items()})

# ================================================================== 7. matched controls (diagnostic only)
Cc = D.filter(pl.col("in_C")).with_columns(
    pl.when(pl.col("u_F5") < 1 / 3).then(0).when(pl.col("u_F5") < 2 / 3).then(1).otherwise(2).alias("F5ter"))
MV = ["u_r30", "u_vx5", "u_sigma30", "u_lqv", "u_r1", "u_zP5", "u_zSH5", "u_zrng"]
Sinv = np.linalg.inv(np.cov(Cc.select(MV).to_numpy().T))
evs = Cc.filter(pl.col("ev_D1") & pl.col("y7_n").is_not_null() & ~pl.col("contam_b"))
pool = Cc.filter((pl.col("_raw7") == 0) & pl.col("y7_n").is_not_null())
pool_by_day = {k[0]: v for k, v in pool.partition_by("date", as_dict=True).items()}
pairs = []
nfull = 0
for r in evs.iter_rows(named=True):
    pdx = pool_by_day.get(r["date"])
    if pdx is None:
        continue
    cand = pdx.filter(((pl.col("u_r5z") - r["u_r5z"]).abs() <= 0.2) & ((pl.col("u_F5") - r["u_F5"]).abs() <= 0.2) & (pl.col("F5ter") == r["F5ter"]))
    if cand.height == 0:
        continue
    dv = cand.select(MV).to_numpy() - np.array([r[c] for c in MV])[None]
    md = np.einsum("ij,jk,ik->i", dv, Sinv, dv)
    o = np.argsort(md)[:5]
    nfull += len(o) == 5
    for j in o:
        pairs.append((r["date"], r["symbol"], cand["symbol"][int(j)], 1.0 / len(o)))
PR = pl.DataFrame(pairs, schema=["date", "symbol", "ctrl", "w"], orient="row")
ctrl = PR.join(Cc.select("date", pl.col("symbol").alias("ctrl"), "y7_n", *UZ), on=["date", "ctrl"], how="left")
trt = evs.join(PR.select("date", "symbol").unique(), on=["date", "symbol"], how="inner")
# Abadie-Imbens bias correction with mu(X)=OLS of y on u_Z fitted on matched controls
Xc = np.column_stack([np.ones(ctrl.height), ctrl.select(UZ).to_numpy()])
bc = np.linalg.lstsq(Xc, ctrl["y7_n"].to_numpy(), rcond=None)[0]
ctrl = ctrl.with_columns(pl.Series("adj", ctrl["y7_n"].to_numpy() - Xc @ bc))
trt = trt.with_columns(pl.Series("adj", trt["y7_n"].to_numpy() - np.column_stack([np.ones(trt.height), trt.select(UZ).to_numpy()]) @ bc))
cmean = ctrl.group_by("date", "symbol").agg((pl.col("adj") * pl.col("w")).sum().alias("cadj"), (pl.col("y7_n") * pl.col("w")).sum().alias("cy"))
tt = trt.join(cmean, on=["date", "symbol"]).with_columns((pl.col("adj") - pl.col("cadj")).alias("tau"), (pl.col("y7_n") - pl.col("cy")).alias("tau_raw"))
bal = []
for c in UZ:
    tv = trt[c].to_numpy()
    cv = ctrl[c].to_numpy()
    cw = ctrl["w"].to_numpy()
    mc = np.average(cv, weights=cw)
    vc = np.average((cv - mc) ** 2, weights=cw)
    smd = (tv.mean() - mc) / np.sqrt((tv.var() + vc) / 2) if (tv.var() + vc) > 0 else 0.0
    bal.append((c, float(tv.mean()), float(mc), float(smd)))
BAL = pl.DataFrame(bal, schema=["var", "mean_event", "mean_ctrl", "smd"], orient="row")
RES["matched"] = {"n_events": evs.height, "n_matched": tt.height, "n_with_5_controls": int(nfull),
                  "ATT_bias_corrected": float(tt["tau"].mean()), "ATT_se_naive": float(tt["tau"].std() / np.sqrt(tt.height)),
                  "ATT_raw": float(tt["tau_raw"].mean()), "max_abs_smd": float(BAL["smd"].abs().max()),
                  "flag": "匹配不平衡" if BAL["smd"].abs().max() >= 0.1 else "balanced",
                  "imbalanced_vars": BAL.filter(pl.col("smd").abs() >= 0.1)["var"].to_list()}
log("matched", RES["matched"])


# ================================================================== 8. G2 event sleeve (trading_rule)
def build_sleeve():
    ev = (F.filter(pl.col("ev_D1")).select("date", "symbol", "A", "sig30", "qv_med30", "beta_60d")
          .join(X7, on=["date", "symbol"], how="left").sort("date", pl.col("A"), descending=[False, True]))
    ev = ev.with_columns(pl.min_horizontal(0.005 * NAV / pl.col("sig30"), pl.lit(0.10 * NAV), 0.005 * pl.col("qv_med30")).alias("w_usd"),
                         pl.col("beta_60d").clip(0.5, 2).alias("hb"),
                         pl.when(pl.col("qv_med30") >= 1e8).then(0.0002).when(pl.col("qv_med30") >= 5e7).then(0.0005).otherwise(0.0010).alias("slip"))
    sel = []
    open_pos = []   # (last active day)
    n_skip_cap = 0
    for dday, grp in ev.group_by("date", maintain_order=True):
        dd = dday[0]
        entry = dd + timedelta(days=1)
        open_pos = [x for x in open_pos if x >= entry]
        cap = 15 - len(open_pos)
        g = grp.sort("A", descending=True)
        gross = 0.0
        for r in g.iter_rows(named=True):
            if cap <= 0:
                n_skip_cap += 1
                continue
            last = r["e"] if r["e"] is not None else entry
            sel.append({**r, "entry": entry, "last": last, "no_exit_price": r["e"] is None})
            open_pos.append(last)
            cap -= 1
    P = pl.DataFrame(sel)
    days = pl.DataFrame({"t": pl.date_range(date(2022, 1, 2), PRICE_END, "1d", eager=True)})
    # position-day rows
    PD = (P.with_row_index("pid").with_columns(pl.date_ranges("entry", "last").alias("t")).explode("t")
          .filter(pl.col("t") <= pl.col("last")))
    PD = PD.join(G.select("symbol", pl.col("date").alias("t"), "rspan", "trade"), on=["symbol", "t"], how="left")
    PD = PD.join(BTC.select(pl.col("date").alias("t"), "rB"), on="t", how="left")
    PD = PD.with_columns(pl.when(pl.col("no_exit_price")).then(0.0).otherwise(pl.col("rspan").fill_null(0.0)).alias("ri"),
                         pl.when(pl.col("no_exit_price")).then(0.0).otherwise(pl.col("rB").fill_null(0.0)).alias("rb"))
    a = ms_of(PD["t"]).to_numpy()
    b = ms_of(PD["t"] + timedelta(days=1)).to_numpy()
    assert b.max() <= FUND_END_MS
    noex = PD["no_exit_price"].to_numpy()
    Fi = np.where(noex, 0.0, fund_exec(PD["symbol"].to_list(), a, b))
    Fb = np.where(noex, 0.0, fund_exec(["BTCUSDT"] * PD.height, a, b))
    PD = PD.with_columns(pl.Series("Fi", Fi), pl.Series("Fb", Fb))
    PD = PD.with_columns(((pl.col("w_usd") / NAV) * ((pl.col("ri").exp() - 1) - pl.col("Fi") - pl.col("hb") * ((pl.col("rb").exp() - 1) - pl.col("Fb")))).alias("pnl"))
    out = days
    for fee, tag in [(0.0003, "003"), (0.0005, "005"), (0.0007, "007")]:
        cost1 = (pl.col("w_usd") / NAV) * ((fee + pl.col("slip")) + fee * pl.col("hb"))
        ce = P.select(pl.col("entry").alias("t"), cost1.alias("c")).vstack(P.select(pl.col("last").alias("t"), cost1.alias("c")))
        ce = ce.group_by("t").agg(pl.col("c").sum())
        pn = PD.group_by("t").agg(pl.col("pnl").sum())
        o = days.join(pn, on="t", how="left").join(ce, on="t", how="left").fill_null(0.0)
        out = out.join(o.select("t", (pl.col("pnl") - pl.col("c")).alias(f"ret_{tag}")), on="t", how="left")
    npos = PD.group_by("t").agg(pl.len().alias("n_pos"), (pl.col("w_usd").sum() / NAV).alias("gross_notional"))
    out = out.join(npos, on="t", how="left").with_columns(pl.col("n_pos").fill_null(0), pl.col("gross_notional").fill_null(0.0)).join(BTC.select(pl.col("date").alias("t"), (pl.col("rB").exp() - 1).alias("btc_ret")), on="t", how="left")
    return out.rename({"t": "date"}), P, n_skip_cap


SL, POS, nskip = build_sleeve()
assert SL["date"].max() <= PRICE_END and POS["last"].max() <= date(2026, 3, 29)


def sharpe(x):
    x = np.asarray(x, float)
    return float(x.mean() / x.std(ddof=1) * np.sqrt(365))


G2 = {"n_positions": POS.height, "skipped_capacity": nskip, "no_exit_price_positions": int(POS["no_exit_price"].sum()),
      "days": SL.height, "max_concurrent": int(SL["n_pos"].max()), "mean_gross_notional": float(SL["gross_notional"].mean()),
      "max_gross_notional": float(SL["gross_notional"].max())}
for tag in ["003", "005", "007"]:
    G2[f"sharpe_{tag}"] = sharpe(SL[f"ret_{tag}"])
    G2[f"ann_ret_{tag}"] = float(SL[f"ret_{tag}"].mean() * 365)
    G2[f"corr_btc_{tag}"] = float(np.corrcoef(SL[f"ret_{tag}"].to_numpy(), SL["btc_ret"].to_numpy())[0, 1])
G2["by_year_sharpe_005"] = {str(y[0]): sharpe(g["ret_005"]) for y, g in SL.group_by(pl.col("date").dt.year(), maintain_order=True)}
RES["G2"] = G2
log("G2", G2)

# ================================================================== 9. criteria
icm = IC["main"]
ev = EV["main"]
b1 = ev["ev_D1"]["beta"]
crt = ev["mean_crt_D1"]
yrs = [str(y) for y in (2022, 2023, 2024, 2025)]
crit = []


def C(name, ok, val):
    crit.append({"criterion": name, "passed": bool(ok), "value": val})


C("P1 主IC: IC>0, NW t>=3, >=6/7 相位均值>0", icm["IC"] > 0 and icm["t"] >= 3 and icm["phases_pos"] >= 6,
  f"IC={icm['IC']:.4f}, t={icm['t']:.2f}, phases_pos={icm['phases_pos']}/7")
npos_y = sum(icm["by_year"].get(y, {"IC": -1})["IC"] > 0 for y in yrs)
C("P2 分年: 2022-2025 至少3年周均IC>0", npos_y >= 3, "; ".join(f"{y}:{icm['by_year'].get(y, {}).get('IC', float('nan')):.4f}" for y in yrs) + f" -> {npos_y}/4")
m_ = IC["metrics"]
C("P3 metrics子集: IC^M>0, t>=2, IC^M>=0.5*IC", m_["IC"] > 0 and m_["t"] >= 2 and m_["IC"] >= 0.5 * icm["IC"],
  f"IC^M={m_['IC']:.4f}, t={m_['t']:.2f}, 0.5*IC={0.5*icm['IC']:.4f}")
same = lambda a, b: np.sign(a) == np.sign(b)
e01, el1 = IC["entry_0100"]["IC"], IC["lag1"]["IC"]
C("P4 执行稳健: 01:00版同号且>=0.7*IC; 滞后1天版同号且>=0.5*IC",
  same(e01, icm["IC"]) and e01 >= 0.7 * icm["IC"] and same(el1, icm["IC"]) and el1 >= 0.5 * icm["IC"],
  f"IC_0100={e01:.4f} (t={IC['entry_0100']['t']:.2f}), IC_lag1={el1:.4f} (t={IC['lag1']['t']:.2f}), IC={icm['IC']:.4f}")
es = IC["strict"]["IC"]
C("P5 标签稳健: 严格版同号且>=0.7*IC", same(es, icm["IC"]) and es >= 0.7 * icm["IC"], f"IC_strict={es:.4f}, IC={icm['IC']:.4f}")
C("E1 事件: beta_D1>0, 单侧p<=0.05, beta_D1>=2*mean c_rt", b1 > 0 and ev["ev_D1"]["p_one_sided"] <= 0.05 and b1 >= 2 * crt,
  f"beta_D1={b1:.5f}, p={ev['ev_D1']['p_one_sided']:.4f}, 2*mean c_rt={2*crt:.5f}, n_ev={ev['ev_D1']['n_events']}")
e2ok, e2txt = True, []
for k in JUDGE:
    o = ev["ev_" + k]
    ok = o["beta"] <= 0.5 * b1 and o.get("delta", -1) > 0
    if k in ("C3", "C4") or (not MERGE_C12 and k in ("C1", "C2")):
        ok = ok and o.get("delta_p_one_sided", 1) <= 0.05
    e2ok &= ok
    e2txt.append(f"{k}: beta={o['beta']:.5f}, delta={o.get('delta', float('nan')):.5f}, p_delta={o.get('delta_p_one_sided', float('nan')):.4f}")
C("E2 消融: 各判定组 beta_k<=0.5*beta_D1 且 delta>0 (C3/C4 p_delta<=0.05; C12 点估计)", e2ok, " | ".join(e2txt))
ny = sum(EV["by_year"][y]["ev_D1"]["beta"] > 0 for y in yrs)
C("E3 事件分年: 2022-2025 至少3年 beta_D1>0", ny >= 3, "; ".join(f"{y}:{EV['by_year'][y]['ev_D1']['beta']:.5f}" for y in yrs) + f" -> {ny}/4")
bw = EV["winsor_2.5_97.5"]["ev_D1"]["beta"]
bt5 = EV["drop_top5_coins"]["ev_D1"]["beta"]
C("E4 尾部: 缩尾后 beta_D1>=mean c_rt; 去掉前5币后 beta_D1>0", bw >= crt and bt5 > 0,
  f"beta_winsor={bw:.5f} vs mean c_rt={crt:.5f}; beta_drop_top5={bt5:.5f} (dropped {top5})")
bs = EV["strict"]["ev_D1"]["beta"]
C("E5 严格标签版 beta_D1 同号且>=0.7*beta_D1", same(bs, b1) and bs >= 0.7 * b1, f"beta_strict={bs:.5f}, beta_D1={b1:.5f}")
C("G2 sleeve: 0.05%净夏普>=1.0, 0.07%>0, |corr BTC|<=0.3",
  G2["sharpe_005"] >= 1.0 and G2["sharpe_007"] > 0 and abs(G2["corr_btc_005"]) <= 0.3,
  f"SR005={G2['sharpe_005']:.3f}, SR007={G2['sharpe_007']:.3f}, corr={G2['corr_btc_005']:.3f}")
kills = []


def K(name, hit, val):
    kills.append({"kill": name, "triggered": bool(hit), "value": val})


K("K1 IC<=0 或 t<3", icm["IC"] <= 0 or icm["t"] < 3, f"IC={icm['IC']:.4f}, t={icm['t']:.2f}")
K("K2 仅3d/14d显著而7d不显著", (icm["t"] < 3) and (IC["h3"]["t"] >= 3 or IC["h14"]["t"] >= 3),
  f"t7={icm['t']:.2f}, t3={IC['h3']['t']:.2f}, t14={IC['h14']['t']:.2f}")
K("K3 IC^M<=0", m_["IC"] <= 0, f"IC^M={m_['IC']:.4f}")
K("K4 任一判定组 beta_k>=0.5*beta_D1", any(ev["ev_" + k]["beta"] >= 0.5 * b1 for k in JUDGE),
  "; ".join(f"{k}:{ev['ev_'+k]['beta']:.5f}" for k in JUDGE) + f" vs 0.5*beta_D1={0.5*b1:.5f}")
K("K5 01:00版或滞后1天版反号", not same(e01, icm["IC"]) or not same(el1, icm["IC"]) or
  not same(EV["entry_0100"]["ev_D1"]["beta"], b1) or not same(EV["lag1"]["ev_D1"]["beta"], b1),
  f"IC01={e01:.4f}, IClag={el1:.4f}, b01={EV['entry_0100']['ev_D1']['beta']:.5f}, blag={EV['lag1']['ev_D1']['beta']:.5f}")
K("K6 <=1整年为正, 或缩尾/去前5币 beta_D1<=0", npos_y <= 1 or ny <= 1 or bw <= 0 or bt5 <= 0,
  f"IC年正={npos_y}, beta年正={ny}, bw={bw:.5f}, bt5={bt5:.5f}")
K("K7 严格标签版反号", not same(es, icm["IC"]) or not same(bs, b1), f"IC_strict={es:.4f}, beta_strict={bs:.5f}")
K("K8 G0 不通过", not G0["passed"], str(G0["checks"]))
RES["criteria"] = crit
RES["kills"] = kills
RES["verdict"] = "pass" if all(c["passed"] for c in crit) and not any(k["triggered"] for k in kills) else "fail"
log("VERDICT", RES["verdict"])
for c in crit:
    log(("PASS " if c["passed"] else "FAIL ") + c["criterion"], "|", c["value"])
for k in kills:
    log(("KILL " if k["triggered"] else "ok   ") + k["kill"], "|", k["value"])

# ================================================================== 10. write (dev-only outputs)
D.filter(pl.col("in_C") | pl.col("ev_any")).select(
    "date", "symbol", "in_C", "in_CM", "e_A", "e_A_M", "A", "ev_D1", "ev_C12", "ev_C3", "ev_C4", "contam_a", "contam_b", "c_rt",
    "e", *[c for c in D.columns if c.startswith("y") and c not in ("year",)]).write_parquet(f"{OUTDIR}/labels.parquet")
ic_daily_all.sort("date").write_parquet(f"{OUTDIR}/ic_daily.parquet")
W_main.write_parquet(f"{OUTDIR}/ic_weekly.parquet")
SL.write_parquet(f"{OUTDIR}/sleeve_daily.parquet")
POS.write_parquet(f"{OUTDIR}/sleeve_positions.parquet")
BAL.write_parquet(f"{OUTDIR}/matched_balance.parquet")
RES["code_sha256"] = {"run_test.py": sha(os.path.abspath(__file__)), "build_features.py": G0["script_sha256"]}
RES["holdout_guard"] = __doc__.split("Holdout guard (explicit):")[1].strip()
json.dump(RES, open(f"{OUTDIR}/results.json", "w"), indent=1, default=str, ensure_ascii=False)
log("written")
