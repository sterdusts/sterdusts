"""D1P-spot-lead-flow (trial d1p_spot_lead_ic_v1) -- G0 feature build. FEATURES ONLY, NO LABELS.

Spec: /home/user/sterdusts/liqmap/research/flow_prereg_2026-10-10.json (commit b3f95f8), id = D1P-spot-lead-flow.
Outputs (all new files, in OUT):
  excl_symbols.json, d1_features.parquet, g0_daily.parquet, prereg.json (G0 section)
Holdout guard: every input is filtered to date <= 2026-03-22 (= last eligible signal day) and in_holdout==False
before any computation; no fwd_/res_/fund_ column is ever selected (asserted).
"""
import hashlib, json, os, sys, time
from datetime import date
import numpy as np
import polars as pl

T0 = time.time()
OUT = os.environ.get("D1P_OUT", "/home/user/data/flow/D1P-spot-lead-flow_run")
FLOW = "/home/user/data/flow"
FEAT_END = date(2026, 3, 22)      # last eligible signal day d (E(d) upper bound)
D_START = date(2022, 1, 1)
for f in ["d1_features.parquet", "g0_daily.parquet", "prereg.json", "excl_symbols.json"]:
    assert not os.path.exists(f"{OUT}/{f}"), f"refuse to overwrite {f}"


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


# ------------------------------------------------------------------ 0. column guard (no labels)
PERP_COLS = ["date", "symbol", "bar_status", "high", "low", "close", "ret_1d", "quote_volume",
             "taker_buy_quote_volume", "univ_excluded", "in_universe", "in_holdout", "beta_60d",
             "funding_1d", "oi_value", "top_ls_pos", "taker_ls_mean", "metrics_n"]
SPOTP_COLS = ["date", "perp_symbol", "spot_base", "has_spot", "px_ok", "px_logdiff", "qv_usdt", "qv_usdt_USDT"]
SPOTB_COLS = ["date", "base", "tbqv_usdt_USDT", "tbqv_usdt", "days_since_prev"]
for c in PERP_COLS + SPOTP_COLS + SPOTB_COLS:
    assert not c.startswith(("fwd_", "res_", "fund_")), f"label column requested: {c}"

# ------------------------------------------------------------------ 1. load (dev-only filter applied at scan time)
perp_all = pl.scan_parquet(f"{FLOW}/perp_daily.parquet").select(PERP_COLS)
excl = sorted(perp_all.filter(pl.col("univ_excluded")).select("symbol").unique().collect()["symbol"].to_list())
assert len(excl) == 189, len(excl)
json.dump({"source": "perp_daily.parquet univ_excluded==True (static frozen list)", "n": len(excl), "symbols": excl},
          open(f"{OUT}/excl_symbols.json", "w"), indent=1)

p = (perp_all.filter((pl.col("date") <= FEAT_END) & (~pl.col("in_holdout")))
     .collect().sort("symbol", "date"))
assert p["date"].max() <= FEAT_END and not p["in_holdout"].any()
log("perp", p.shape)

trade = pl.col("bar_status") == "trade"
p = p.with_columns(trade.alias("trade"))
p = p.with_columns([pl.when(pl.col("trade")).then(pl.col(c)).otherwise(None).alias(c)
                    for c in ["close", "high", "low", "ret_1d", "quote_volume", "taker_buy_quote_volume"]])

# spot
sp = (pl.scan_parquet(f"{FLOW}/spot_daily_perp.parquet").select(SPOTP_COLS)
      .filter(pl.col("date") <= FEAT_END).collect().rename({"perp_symbol": "symbol"}))
sb = (pl.scan_parquet(f"{FLOW}/spot_daily_base.parquet").select(SPOTB_COLS)
      .filter(pl.col("date") <= FEAT_END).collect().sort("base", "date"))
# base-level spot epoch start: latest date <= d with days_since_prev>=4 (or the base's first row)
sb = sb.with_columns(pl.when(pl.col("days_since_prev").is_null() | (pl.col("days_since_prev") >= 4))
                     .then(pl.col("date")).otherwise(None).forward_fill().over("base").alias("bes"))
sp = sp.join(sb.rename({"base": "spot_base"}), on=["date", "spot_base"], how="left")
p = p.join(sp, on=["date", "symbol"], how="left").sort("symbol", "date")
log("joined", p.shape)

# ------------------------------------------------------------------ 2. epochs
p = p.with_columns(pl.when(pl.col("trade")).then(pl.col("date")).otherwise(None).alias("_td"))
p = p.with_columns(pl.col("_td").shift(1).forward_fill().over("symbol").alias("_prev_td"))
new_ep = pl.col("trade") & (pl.col("_prev_td").is_null() |
                            ((pl.col("date") - pl.col("_prev_td")).dt.total_days() - 1 >= 14))
p = p.with_columns(new_ep.alias("_new_ep"))
p = p.with_columns(pl.col("_new_ep").cast(pl.Int32).cum_sum().over("symbol").alias("epoch_id"),
                   pl.when(pl.col("_new_ep")).then(pl.col("date")).otherwise(None).forward_fill().over("symbol").alias("epoch_start"))
p = p.with_columns((pl.col("date") - pl.col("epoch_start")).dt.total_days().alias("age_epoch"))
# spot epoch: new when spot_base (non-null) differs from last non-null spot_base, or base epoch start changes
p = p.with_columns(pl.col("spot_base").shift(1).forward_fill().over("symbol").alias("_last_sb"),
                   pl.col("bes").shift(1).forward_fill().over("symbol").alias("_last_bes"))
new_sep = pl.col("spot_base").is_not_null() & (
    pl.col("_last_sb").is_null() | (pl.col("spot_base") != pl.col("_last_sb")) |
    pl.col("bes").is_null() | pl.col("_last_bes").is_null() | (pl.col("bes") != pl.col("_last_bes")))
p = p.with_columns(new_sep.fill_null(False).alias("_new_sep"))
p = p.with_columns((pl.col("_new_ep") | pl.col("_new_sep")).cast(pl.Int32).cum_sum().over("symbol").alias("seg_id"),
                   pl.when(pl.col("_new_sep")).then(pl.col("date")).otherwise(None).forward_fill().over("symbol").alias("spot_epoch_start"))
EP = ["symbol", "epoch_id"]      # perp-only windows
SG = ["symbol", "seg_id"]        # windows that involve spot quantities (perp epoch AND spot epoch)

# ------------------------------------------------------------------ 3. base quantities
btc = p.filter(pl.col("symbol") == "BTCUSDT").select("date", pl.col("ret_1d").alias("btc_r"))
p = p.join(btc, on="date", how="left").sort("symbol", "date")
p = p.with_columns((pl.col("ret_1d") - pl.col("beta_60d").fill_null(1.0) * pl.col("btc_r")).alias("rr"))
p = p.with_columns(
    pl.col("ret_1d").rolling_std(30, min_samples=20).over(EP).alias("sig30"),
    pl.col("quote_volume").rolling_median(30, min_samples=20).over(EP).alias("qv_med30"),
    (pl.col("trade") & pl.col("ret_1d").is_not_null() & pl.col("funding_1d").is_not_null())
    .cast(pl.Int32).rolling_sum(5, min_samples=5).over(EP).alias("_p5n"),
)
p = p.with_columns((pl.col("_p5n") == 5).fill_null(False).alias("perp5_ok"))
s_ok = (pl.col("has_spot") & pl.col("px_ok") & (pl.col("qv_usdt_USDT") > 0) &
        (pl.col("qv_usdt_USDT") >= 0.5 * pl.col("qv_usdt")) & (pl.col("days_since_prev") == 1)).fill_null(False)
s_all_ok = (pl.col("has_spot") & pl.col("px_ok") & (pl.col("days_since_prev") == 1)).fill_null(False)
p = p.with_columns(s_ok.alias("s_ok"), s_all_ok.alias("s_all_ok"))
p = p.with_columns(
    pl.col("s_ok").cast(pl.Int32).rolling_sum(5, min_samples=5).over(SG).alias("_s5n"),
    pl.col("s_all_ok").cast(pl.Int32).rolling_sum(5, min_samples=5).over(SG).alias("_sa5n"),
)
# perp5_ok within the seg group: recompute the 5-day trade/ret/funding count inside SG so pre-spot-epoch rows are excluded
p = p.with_columns((pl.col("trade") & pl.col("ret_1d").is_not_null() & pl.col("funding_1d").is_not_null())
                   .cast(pl.Int32).rolling_sum(5, min_samples=5).over(SG).alias("_p5n_sg"))
p = p.with_columns((pl.col("perp5_ok") & (pl.col("_p5n_sg") == 5).fill_null(False) & (pl.col("_s5n") == 5).fill_null(False)).alias("win5_ok"),
                   (pl.col("_sa5n") == 5).fill_null(False).alias("sall5_ok"))


def rs5(c, grp):
    return pl.col(c).rolling_sum(5, min_samples=5).over(grp)


p = p.with_columns(
    rs5("tbqv_usdt_USDT", SG).alias("_s_tb"), rs5("qv_usdt_USDT", SG).alias("_s_qv"),
    rs5("quote_volume", SG).alias("_p_qv_sg"),
    rs5("taker_buy_quote_volume", EP).alias("_p_tb"), rs5("quote_volume", EP).alias("_p_qv"),
    rs5("tbqv_usdt", SG).alias("_sa_tb"), rs5("qv_usdt", SG).alias("_sa_qv"),
    rs5("funding_1d", EP).alias("_f5"),
)
p = p.with_columns(
    pl.when(pl.col("win5_ok")).then(pl.col("_s_tb") / pl.col("_s_qv")).otherwise(None).alias("S5"),
    pl.when(pl.col("perp5_ok")).then(pl.col("_p_tb") / pl.col("_p_qv")).otherwise(None).alias("P5"),
    pl.when(pl.col("win5_ok")).then((pl.col("_s_qv") / (pl.col("_s_qv") + pl.col("_p_qv_sg"))).log()).otherwise(None).alias("SH5"),
    pl.when(pl.col("sall5_ok") & (pl.col("_sa_qv") > 0)).then(pl.col("_sa_tb") / pl.col("_sa_qv")).otherwise(None).alias("S5all"),
    pl.when(pl.col("perp5_ok")).then(pl.col("_f5")).otherwise(None).alias("F5"),
    pl.when(pl.col("high") > pl.col("low")).then((pl.col("high") / pl.col("low")).log().log()).otherwise(None).alias("lrng"),
)
p = p.with_columns([pl.col(c).fill_nan(None) for c in ["S5", "P5", "SH5", "S5all", "lrng"]])
log("base quantities done")


# ------------------------------------------------------------------ 4. robust z (numpy, grouped, calendar-row windows)
def robust_z(x, gid, lo, hi, min_n=40):
    out = np.full(len(x), np.nan)
    w = lo - hi + 1
    brk = np.flatnonzero(gid[1:] != gid[:-1]) + 1
    for s, e in zip(np.r_[0, brk], np.r_[brk, len(x)]):
        seg = x[s:e]
        n = e - s
        if np.isfinite(seg).sum() < min_n:
            continue
        pad = np.r_[np.full(lo, np.nan), seg]
        win = np.lib.stride_tricks.sliding_window_view(pad, w)[:n]   # row j -> seg[j-lo .. j-hi]
        cnt = np.isfinite(win).sum(1)
        ok = (cnt >= min_n) & np.isfinite(seg)
        if not ok.any():
            continue
        wv = win[ok]
        med = np.nanmedian(wv, axis=1)
        mad = np.nanmedian(np.abs(wv - med[:, None]), axis=1)
        sc = 1.4826 * mad
        z = (seg[ok] - med) / np.where(sc > 0, sc, np.nan)
        z[~(sc >= 1e-4)] = np.nan
        tmp = out[s:e]
        tmp[ok] = z
    return out


def gid_of(cols):
    brk = pl.lit(False)
    for c in cols:
        brk = brk | (pl.col(c) != pl.col(c).shift(1)).fill_null(True)
    return p.select(brk.cast(pl.Int64).cum_sum().alias("g"))["g"].to_numpy()


g_ep = gid_of(EP)
g_sg = gid_of(SG)
# p is sorted by symbol,date and ids are monotone within symbol -> contiguous groups
assert (np.diff(g_ep) != 0).sum() == p.select(pl.struct(EP).n_unique()).item() - 1
assert (np.diff(g_sg) != 0).sum() == p.select(pl.struct(SG).n_unique()).item() - 1
zz = {}
for X, g in [("S5", g_sg), ("SH5", g_sg), ("S5all", g_sg), ("P5", g_ep)]:
    zz["z" + X] = robust_z(p[X].to_numpy().astype(float), g, 64, 5)
zz["zrng"] = robust_z(p["lrng"].to_numpy().astype(float), g_ep, 60, 1)
p = p.with_columns([pl.Series(k, v).fill_nan(None) for k, v in zz.items()])
log("robust z done")

# ------------------------------------------------------------------ 5. remaining features
p = p.with_columns(
    pl.col("rr").alias("r1"),
    pl.col("rr").rolling_sum(3, min_samples=3).over(EP).alias("r3"),
    pl.when(pl.col("perp5_ok")).then(pl.col("rr").rolling_sum(5, min_samples=5).over(EP) / (pl.col("sig30") * np.sqrt(5))).otherwise(None).alias("r5z"),
    pl.col("rr").rolling_sum(7, min_samples=7).over(EP).alias("r7"),
    (pl.col("rr").rolling_sum(30, min_samples=25).over(EP) * 30.0 /
     pl.col("rr").is_not_null().cast(pl.Float64).rolling_sum(30, min_samples=1).over(EP)).alias("_r30"),
    pl.col("rr").is_not_null().cast(pl.Int32).rolling_sum(30, min_samples=1).over(EP).alias("_r30n"),
    (pl.col("quote_volume").rolling_mean(5, min_samples=5).over(EP) /
     pl.col("quote_volume").rolling_median(30, min_samples=20).shift(5).over(EP)).log().alias("vx5"),
    pl.when(pl.col("s_ok")).then(pl.col("qv_usdt_USDT")).otherwise(None).alias("_qvS"),
)
p = p.with_columns(
    pl.when(pl.col("_r30n") >= 25).then(pl.col("_r30")).otherwise(None).alias("r30"),
    (pl.col("_qvS").rolling_mean(5, min_samples=5).over(SG) /
     pl.col("_qvS").rolling_median(30, min_samples=20).shift(5).over(SG)).log().alias("vxS5"),
    pl.when(pl.col("F5").is_not_null()).then((pl.col("F5") < 0).cast(pl.Float64)).otherwise(None).alias("negF5"),
    (pl.col("F5") - pl.col("F5").shift(5).over(EP)).alias("dF5"),
    (pl.col("taker_buy_quote_volume") / pl.col("quote_volume")).alias("tbr1"),
    pl.col("px_logdiff").alias("px"),
    (pl.col("px_logdiff") - pl.col("px_logdiff").shift(5).over(SG)).alias("dbasis5"),
    pl.col("qv_med30").log().alias("lqv"),
    pl.col("beta_60d").alias("beta"),
    pl.when(pl.col("age_epoch") > 0).then(pl.col("age_epoch").cast(pl.Float64).log()).otherwise(None).alias("lage"),
    pl.col("in_universe").cast(pl.Float64).alias("inU"),
    pl.col("sig30").alias("sigma30"),
)
# metrics-only controls
mok = (pl.col("metrics_n") >= 280) & (pl.col("metrics_n").shift(5).over(EP) >= 280)
pos = lambda c: (pl.col(c) > 0)
pos5 = lambda c: (pl.col(c).shift(5).over(EP) > 0)
p = p.with_columns(
    pl.when(mok & pos("oi_value") & pos5("oi_value")).then((pl.col("oi_value") / pl.col("oi_value").shift(5).over(EP)).log()).otherwise(None).alias("doi5"),
    pl.when(mok & pos("oi_value") & pos("qv_med30")).then((pl.col("oi_value") / pl.col("qv_med30")).log()).otherwise(None).alias("oiqv"),
    pl.when(mok & pos("taker_ls_mean")).then(pl.col("taker_ls_mean")).otherwise(None).alias("tls"),
    pl.when(mok & pos("top_ls_pos") & pos5("top_ls_pos")).then((pl.col("top_ls_pos") / pl.col("top_ls_pos").shift(5).over(EP)).log()).otherwise(None).alias("dtopls5"),
)
FEATS_FLOAT = ["r1", "r3", "r5z", "r7", "r30", "vx5", "vxS5", "sigma30", "zrng", "F5", "negF5", "dF5", "P5", "zP5", "tbr1",
               "lqv", "beta", "lage", "zSH5", "px", "dbasis5", "inU", "S5", "SH5", "S5all", "zS5", "zS5all",
               "doi5", "oiqv", "tls", "dtopls5"]
p = p.with_columns([pl.col(c).cast(pl.Float64).fill_nan(None) for c in FEATS_FLOAT])
p = p.with_columns([pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in FEATS_FLOAT])
log("features done")

# BTC past-60d return (stratification only): sum of BTC ret_1d over [d-59, d] (calendar, BTC always trades)
btc60 = (p.filter(pl.col("symbol") == "BTCUSDT").sort("date")
         .select("date", pl.col("ret_1d").rolling_sum(60, min_samples=60).alias("btc_r60")))
p = p.join(btc60, on="date", how="left")

# ------------------------------------------------------------------ 6. eligibility, cross-sections
E = (~pl.col("symbol").is_in(excl) & (pl.col("symbol") != "BTCUSDT") & pl.col("trade") & pl.col("perp5_ok") &
     (pl.col("age_epoch") >= 60) & pl.col("beta_60d").is_not_null() & pl.col("sig30").is_not_null() &
     pl.col("qv_med30").is_not_null() & (pl.col("qv_med30") >= 2e7) & (~pl.col("in_holdout")) &
     (pl.col("date") >= D_START) & (pl.col("date") <= FEAT_END))
p = p.with_columns(E.fill_null(False).alias("in_E"))
p = p.with_columns((pl.col("in_E") & pl.col("win5_ok") & pl.col("zS5").is_not_null() & pl.col("zP5").is_not_null()
                    & pl.col("zSH5").is_not_null()).alias("in_S"))
Z = ["r1", "r3", "r5z", "r7", "r30", "vx5", "vxS5", "sigma30", "zrng", "F5", "negF5", "dF5", "P5", "zP5", "tbr1",
     "lqv", "beta", "lage", "zSH5", "px", "dbasis5", "inU"]
assert len(Z) == 22
BIN = {"negF5", "inU"}
MET = ["doi5", "oiqv", "tls", "dtopls5"]

keep_cols = (["date", "symbol", "in_universe", "beta_60d", "qv_med30", "sig30", "age_epoch", "spot_base", "btc_r60",
              "S5", "SH5", "S5all", "zS5", "zS5all", "win5_ok", "in_E", "in_S"] + [c for c in Z if c not in ("sigma30", "beta")] + MET)
f = p.filter(pl.col("in_E")).select(keep_cols).with_columns(pl.col("sig30").alias("sigma30"), pl.col("beta_60d").alias("beta"))
del p
log("E rows", f.height, "S rows", f["in_S"].sum())

Sx = pl.col("in_S")
med_S = lambda e: e.filter(Sx).median().over("date")
f = f.with_columns((pl.col("zS5") - pl.col("zP5")).alias("_dA"))
f = f.with_columns(
    pl.when(Sx).then((pl.col("_dA") - med_S(pl.col("_dA"))).clip(-5, 5)).otherwise(None).alias("A"),
    pl.when(Sx).then(pl.col("zS5") - med_S(pl.col("zS5"))).otherwise(None).alias("zS5_cs"),
    pl.when(Sx).then(med_S(pl.col("F5"))).otherwise(None).alias("F5med"),
)
# S5all robustness version of A (S_d restricted to zS5all non-null)
Sa = Sx & pl.col("zS5all").is_not_null()
f = f.with_columns((pl.col("zS5all") - pl.col("zP5")).alias("_dAa"))
f = f.with_columns(pl.when(Sa).then((pl.col("_dAa") - pl.col("_dAa").filter(Sa).median().over("date")).clip(-5, 5)).otherwise(None).alias("A_all"))
f = f.with_columns((Sx & pl.all_horizontal([pl.col(c).is_not_null() for c in Z])).alias("in_C0"))
f = f.with_columns(pl.col("in_C0").sum().over("date").alias("nC"))
f = f.with_columns((pl.col("in_C0") & (pl.col("nC") >= 40)).alias("in_C"))
f = f.with_columns((pl.col("in_C") & pl.all_horizontal([pl.col(c).is_not_null() for c in MET])).alias("in_CM"))

# ------------------------------------------------------------------ 7. events (on S_d) + forward-only declustering
quiet = (pl.col("r5z").abs() <= 1) & (pl.col("F5") <= pl.col("F5med"))
c_zs, c_a, c_sh = pl.col("zS5_cs") >= 2, pl.col("A") >= 1, pl.col("zSH5") >= 0
raw = {
    "D1": Sx & c_zs & c_a & quiet & c_sh,
    "C1": Sx & c_zs & c_a & (pl.col("r5z") > 1) & (pl.col("F5") <= pl.col("F5med")) & c_sh,
    "C2": Sx & c_zs & (pl.col("A") < 1) & quiet & c_sh,
    "C3": Sx & (pl.col("zS5_cs") < 2) & c_a & quiet & c_sh,
    "C4": Sx & quiet & (pl.col("zS5_cs") < 2) & (pl.col("A") < 1),
}
f = f.with_columns([e.fill_null(False).alias(f"raw_{k}") for k, e in raw.items()])
f = f.sort("symbol", "date")
assert f.select(pl.sum_horizontal([pl.col(f"raw_{k}").cast(pl.Int32) for k in raw]).max()).item() <= 1, "groups not mutually exclusive"


def decluster(df, flag):
    sym = df["symbol"].to_numpy()
    dn = df["date"].cast(pl.Int32).to_numpy()
    fl = df[flag].to_numpy()
    keep = np.zeros(len(df), bool)
    last = {}
    for i in np.flatnonzero(fl):          # rows sorted by symbol, date -> ascending dates per symbol
        s = sym[i]
        if s not in last or dn[i] - last[s] >= 7:
            keep[i] = True
            last[s] = dn[i]
    return keep


for k in raw:
    f = f.with_columns(pl.Series(f"ev_{k}", decluster(f, f"raw_{k}")))
n_keep = {k: int(f[f"ev_{k}"].sum()) for k in raw}
n_raw = {k: int(f[f"raw_{k}"].sum()) for k in raw}
merge_c12 = not (n_keep["C1"] >= 300 and n_keep["C2"] >= 300)
f = f.with_columns((pl.col("ev_C1") | pl.col("ev_C2")).alias("ev_C12"), (pl.col("raw_C1") | pl.col("raw_C2")).alias("raw_C12"))
log("events kept", n_keep, "raw", n_raw, "merge C12:", merge_c12)

# ------------------------------------------------------------------ 8. ranks + orthogonalisation (features only)
f = f.sort("date", "symbol")


def rank_u2(col, mask):
    # rank within the masked set per date; polars rank of filtered expr over window -> do via helper column
    return pl.when(pl.col(mask)).then(pl.col(col)).otherwise(None).rank("average").over("date")


def add_ranks(df, mask, cols, prefix):
    exprs = []
    n = pl.col(mask).sum().over("date")
    for c in cols:
        if c in BIN:
            exprs.append(pl.when(pl.col(mask)).then(pl.col(c)).otherwise(None).alias(prefix + c))
        else:
            exprs.append(pl.when(pl.col(mask)).then((rank_u2(c, mask) - 0.5) / n).otherwise(None).alias(prefix + c))
    return df.with_columns(exprs)


f = add_ranks(f, "in_C", ["A"] + Z, "u_")
f = add_ranks(f, "in_CM", ["A"] + Z + MET, "uM_")
f = f.with_columns((pl.col("in_C") & pl.col("A_all").is_not_null()).alias("in_Ca"))
f = add_ranks(f, "in_Ca", ["A_all"] + Z, "ua_")


def daily_resid(df, mask, ycol, xcols):
    """per-day OLS y ~ 1 + X on mask rows -> residual array and per-day R2"""
    d = df.filter(pl.col(mask)).select("date", "symbol", ycol, *xcols)
    res = np.full(d.height, np.nan)
    r2 = []
    dates = d["date"].to_numpy()
    Y = d[ycol].to_numpy().astype(float)
    X = d.select(xcols).to_numpy().astype(float)
    brk = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    for s, e in zip(np.r_[0, brk], np.r_[brk, len(dates)]):
        y = Y[s:e]
        x = np.column_stack([np.ones(e - s), X[s:e]])
        b, *_ = np.linalg.lstsq(x, y, rcond=None)
        r = y - x @ b
        res[s:e] = r
        r2.append((int(s), 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean())), e - s))
    out = d.select("date", "symbol").with_columns(pl.Series("e", res))
    r2df = pl.DataFrame({"date": d["date"].gather([t[0] for t in r2]), "r2": [t[1] for t in r2],
                         "n": [t[2] for t in r2]})
    return out, r2df


eA, r2d = daily_resid(f, "in_C", "u_A", ["u_" + c for c in Z])
f = f.join(eA.rename({"e": "e_A"}), on=["date", "symbol"], how="left")
eAa, r2a = daily_resid(f, "in_Ca", "ua_A_all", ["ua_" + c for c in Z])
f = f.join(eAa.rename({"e": "e_A_all"}), on=["date", "symbol"], how="left")

# metrics subset: pooled OLS with date FE (within-day demeaned ranks)
dm = f.filter(pl.col("in_CM")).select("date", "symbol", *["uM_" + c for c in ["A"] + Z + MET])
dm = dm.with_columns([(pl.col(c) - pl.col(c).mean().over("date")).alias(c) for c in dm.columns[2:]])
Ym = dm["uM_A"].to_numpy()
Xm = dm.select(["uM_" + c for c in Z + MET]).to_numpy()
bm, *_ = np.linalg.lstsq(Xm, Ym, rcond=None)
resm = Ym - Xm @ bm
r2_M_pooled = 1 - (resm @ resm) / (Ym @ Ym)
f = f.join(dm.select("date", "symbol").with_columns(pl.Series("e_A_M", resm)), on=["date", "symbol"], how="left")
log("residuals done; pooled metrics R2", r2_M_pooled)

# ------------------------------------------------------------------ 9. G0 gate
days = pl.DataFrame({"date": pl.date_range(D_START, FEAT_END, "1d", eager=True)})
nC = f.filter(pl.col("in_C0")).group_by("date").agg(pl.len().alias("nC"))
nS = f.filter(pl.col("in_S")).group_by("date").agg(pl.len().alias("nS"))
nCM = f.filter(pl.col("in_CM")).group_by("date").agg(pl.len().alias("nCM"))
g0d = (days.join(nC, on="date", how="left").join(nS, on="date", how="left").join(nCM, on="date", how="left")
       .join(r2d.rename({"n": "n_ols"}), on="date", how="left")
       .with_columns([pl.col(c).fill_null(0) for c in ["nC", "nS", "nCM", "n_ols"]]))
med_C = float(g0d["nC"].median())
frac40 = float((g0d["nC"] >= 40).mean())
r2_med = float(r2d["r2"].median())
f = f.with_columns(pl.col("date").dt.year().alias("year"))
by_year = {k: {str(r[0]): r[1] for r in f.filter(pl.col(f"ev_{k}")).group_by("year").len().sort("year").iter_rows()} for k in list(raw) + ["C12"]}
g0 = {
    "C_d_median": med_C, "frac_days_C_ge_40": frac40, "n_days": g0d.height,
    "ols_R2_median": r2_med, "ols_R2_days": r2d.height,
    "events_kept": {**n_keep, "C12": int(f["ev_C12"].sum())}, "events_raw": n_raw, "events_kept_by_year": by_year,
    "C12_merged": merge_c12,
    "C12_rule": "C1 and C2 kept both >=300 -> judged separately; else C12 = C1kept OR C2kept judged, C1/C2 reported only",
    "S_d_median": float(g0d["nS"].median()), "C_dM_median": float(g0d["nCM"].median()),
    "metrics_pooled_R2": float(r2_M_pooled),
    "zS5_availability_in_E_by_year": {str(r[0]): r[1] for r in f.group_by("year").agg(pl.col("zS5").is_not_null().mean()).sort("year").iter_rows()},
}
checks = {
    "C_d_median>=50": med_C >= 50, "frac_days_C>=40 >=0.90": frac40 >= 0.90,
    "R2_median<=0.70": r2_med <= 0.70, "D1_kept>=300": n_keep["D1"] >= 300,
}
g0["checks"] = checks
g0["passed"] = all(checks.values())
# MDE (spec formula): daily IC noise 1/sqrt(n-1), weekly-mean variance ~0.6x daily, NW inflation ~1.15
n_w_exp = g0d.height / 7
se = np.sqrt(0.6 / (med_C - 1)) * 1.15 / np.sqrt(n_w_exp)
g0["MDE"] = {"n_cs": med_C, "weeks_expected": n_w_exp, "SE_ICbar": float(se), "IC_at_t3": float(3 * se)}

# ------------------------------------------------------------------ 10. write
drop = [c for c in f.columns if c.startswith("_")]
f = f.drop(drop).sort("date", "symbol")
f.write_parquet(f"{OUT}/d1_features.parquet")
g0d.write_parquet(f"{OUT}/g0_daily.parquet")
sha = lambda fn: hashlib.sha256(open(fn, "rb").read()).hexdigest()
g0["features_sha256"] = sha(f"{OUT}/d1_features.parquet")
g0["script_sha256"] = sha(os.path.abspath(__file__))
g0["spec"] = {"file": "/home/user/sterdusts/liqmap/research/flow_prereg_2026-10-10.json", "commit": "b3f95f8", "id": "D1P-spot-lead-flow",
              "trial_name": "d1p_spot_lead_ic_v1"}
g0["holdout_guard"] = "perp/spot inputs filtered at scan to date<=2026-03-22 & in_holdout==False; label columns never selected (asserted)"
json.dump({"G0": g0}, open(f"{OUT}/prereg.json", "w"), indent=1, default=str)
log("G0", json.dumps(g0, indent=1, default=str))
