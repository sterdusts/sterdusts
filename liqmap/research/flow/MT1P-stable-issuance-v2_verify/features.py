"""Independent re-implementation of MT1P-stable-issuance-v2 signals, controls and G0.

Reads ONLY feature columns (no fwd_*/fund_* labels). Every input is truncated to
date <= 2026-03-31 immediately after reading. Output written only to this verify dir.
"""
import json
import pathlib
import datetime as dt

import numpy as np
import polars as pl

FLOW = pathlib.Path("/home/user/data/flow")
OUTD = FLOW / "MT1P-stable-issuance-v2_verify"
DEV_END = dt.date(2026, 3, 31)
EXCL = ["BTC", "USDC", "FDUSD", "BUSD", "TUSD", "U", "USD1", "USTC", "PAXG", "XAUT"]
CLIP = 0.01


def ffill_after_first(s: pl.Series) -> pl.Series:
    # forward fill after first non-null; 0 before first non-null
    return s.forward_fill().fill_null(0.0)


def T_np(X: np.ndarray) -> np.ndarray:
    """T(X)(D) = [X(D)-X(D-14)] - 14/166 [X(D-14)-X(D-180)] on a contiguous daily array."""
    n = len(X)
    out = np.full(n, np.nan)
    for i in range(180, n):
        out[i] = (X[i] - X[i - 14]) - (14.0 / 166.0) * (X[i - 14] - X[i - 180])
    return out


def cum_clip_dl(F: np.ndarray) -> np.ndarray:
    lf = np.log(F)
    dl = np.empty_like(lf)
    dl[0] = 0.0
    dl[1:] = lf[1:] - lf[:-1]
    dl = np.where(np.isfinite(dl), dl, np.nan)
    dlc = np.clip(dl, -CLIP, CLIP)
    return np.cumsum(dlc), dlc, dl


def sgn(x):
    return np.where(np.isnan(x), np.nan, np.where(x > 0, 1.0, -1.0))


def build():
    # ---------------- stablecoins ----------------
    s = (pl.scan_parquet(FLOW / "stablecoins_daily.parquet")
         .filter(pl.col("date") <= DEV_END)
         .select("date", "usdt_total", "usdc_total", "busd_total", "fdusd_total", "tusd_total",
                 "dai_total", "usds_total", "usde_total")
         .collect().sort("date"))
    assert s["date"].max() <= DEV_END
    assert (s["date"].diff().drop_nulls().dt.total_days() == 1).all()
    main = ["usdt_total", "usdc_total", "busd_total", "fdusd_total", "tusd_total"]
    xs = {c: ffill_after_first(s[c]).to_numpy() for c in main}
    F = sum(xs[c] for c in main)
    L, dlc, dl = cum_clip_dl(F)
    g14 = np.full(len(L), np.nan); g166 = np.full(len(L), np.nan)
    g14[14:] = L[14:] - L[:-14]
    g166[180:] = L[166:-14] - L[:-180]
    I = g14 - (14 / 166) * g166
    st = sgn(I)
    # components
    LU, _, _ = cum_clip_dl(xs["usdt_total"])
    LC, _, _ = cum_clip_dl(xs["usdc_total"])
    cry = sum(ffill_after_first(s[c]).to_numpy() for c in ["dai_total", "usds_total", "usde_total"])
    LK, _, _ = cum_clip_dl(cry)
    sig = pl.DataFrame({
        "date": s["date"], "F": F, "dl": dl, "dlc": dlc, "L": L, "g14": g14, "g166": g166, "I": I,
        "mt1_state": st, "state_USDT": sgn(T_np(LU)), "state_USDC": sgn(T_np(LC)),
        "state_cry": sgn(T_np(LK)),
    })

    # ---------------- BTC perp features ----------------
    p = (pl.scan_parquet(FLOW / "perp_daily.parquet")
         .filter((pl.col("symbol") == "BTCUSDT") & (pl.col("date") <= DEV_END))
         .select("date", "bar_status", "close", "ret_1d", "funding_1d", "oi", "oi_value", "quote_volume")
         .collect().sort("date"))
    assert p["date"].max() <= DEV_END
    assert (p["date"].diff().drop_nulls().dt.total_days() == 1).all()
    assert (p["bar_status"] == "trade").all()
    lc = p["close"].log()
    # oi_value forward fill <=3 days
    p = p.with_columns(pl.col("oi_value").forward_fill(limit=3).alias("oi_value_ff"))
    p = p.with_columns(
        (lc - lc.shift(7)).alias("c_mom7"),
        (lc - lc.shift(30)).alias("c_mom30"),
        pl.col("ret_1d").rolling_std(30, min_samples=25).alias("c_rv30"),
        pl.col("funding_1d").rolling_sum(7, min_samples=7).alias("c_fund7"),
        (pl.col("oi").log() - pl.col("oi").log().shift(7)).alias("c_doi7"),
        (pl.col("oi_value").log() - pl.col("oi_value").log().shift(7)).alias("c_doi7_val"),
        pl.col("funding_1d").rolling_sum(30, min_samples=30).alias("fund30"),
        (lc - lc.shift(90)).alias("mom90"),
        pl.when(pl.col("close").rolling_mean(200, min_samples=200).is_null()).then(None)
        .otherwise((pl.col("close") > pl.col("close").rolling_mean(200, min_samples=200)).cast(pl.Float64))
        .alias("ma200"),
        (pl.col("quote_volume").rolling_mean(7, min_samples=7).log()
         - pl.col("quote_volume").rolling_mean(90, min_samples=90).log()).alias("dvol30"),
    )
    P_I = T_np(lc.to_numpy())
    oiv = p["oi_value_ff"].to_numpy().astype(float)
    TOI = T_np(np.log(oiv))
    p = p.with_columns(pl.Series("P_I", P_I), pl.Series("state_P", sgn(P_I)), pl.Series("state_OI", sgn(TOI)))

    # ---------------- spot breadth (causal B_t) ----------------
    sm = (pl.read_parquet(FLOW / "spot_map.parquet")
          .filter(pl.col("status") == "ok").group_by("spot_base")
          .agg(pl.col("perp_first_live").min().alias("pfl")))
    sb = (pl.scan_parquet(FLOW / "spot_daily_base.parquet")
          .filter(pl.col("date") <= DEV_END)
          .select("date", "base", "qv_usdt_USDT", "tbqv_usdt_USDT")
          .filter(~pl.col("base").is_in(EXCL))
          .join(sm.lazy(), left_on="base", right_on="spot_base", how="inner")
          .filter(pl.col("pfl") <= pl.col("date"))
          .group_by("date").agg(pl.col("qv_usdt_USDT").sum().alias("qv"),
                                pl.col("tbqv_usdt_USDT").sum().alias("tb"),
                                pl.len().alias("nB"))
          .collect().sort("date"))
    assert sb["date"].max() <= DEV_END
    cal = pl.DataFrame({"date": pl.date_range(dt.date(2021, 1, 1), DEV_END, "1d", eager=True)})
    sb = cal.join(sb, on="date", how="left")
    sb = sb.with_columns(
        (pl.col("tb").rolling_sum(7, min_samples=7) / pl.col("qv").rolling_sum(7, min_samples=7) - 0.5).alias("c_tkr7"),
        pl.col("qv").rolling_sum(7, min_samples=7).log().alias("SV"))
    sb = sb.with_columns(pl.Series("state_SV", sgn(T_np(sb["SV"].to_numpy().astype(float)))))

    df = (sig.join(p.select("date", "close", "ret_1d", "funding_1d", "c_mom7", "c_mom30", "c_rv30",
                            "c_fund7", "c_doi7", "c_doi7_val", "fund30", "P_I", "mom90", "ma200",
                            "dvol30", "state_P", "state_OI"), on="date", how="left")
          .join(sb.select("date", "c_tkr7", "SV", "state_SV", "nB"), on="date", how="left"))
    # state_3c: switch only after 3 consecutive same-sign days
    m = df["mt1_state"].to_numpy()
    s3 = np.full(len(m), np.nan)
    for i in range(len(m)):
        if np.isnan(m[i]):
            continue
        if np.isnan(s3[i - 1]) if i > 0 else True:
            s3[i] = m[i]
            continue
        if i >= 2 and m[i] == m[i - 1] == m[i - 2] and m[i] != s3[i - 1]:
            s3[i] = m[i]
        else:
            s3[i] = s3[i - 1]
    df = df.with_columns(pl.Series("state_3c", s3))
    df = df.with_columns([pl.col(c).fill_nan(None) for c in df.columns if c.startswith("state") or c in ("P_I", "I", "g14", "g166")])
    df = df.with_columns(
        pl.when(pl.col("date") < dt.date(2023, 1, 1)).then(1)
        .when(pl.col("date") < dt.date(2024, 1, 1)).then(2)
        .when(pl.col("date") < dt.date(2025, 1, 1)).then(3).otherwise(4).alias("seg"))
    return df


XCOLS = ["c_mom7", "c_mom30", "c_rv30", "c_fund7", "c_doi7", "c_tkr7", "fund30", "P_I", "mom90", "ma200", "dvol30"]


def runs(x):
    """list of (start_idx, end_idx, value)."""
    out = []
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and x[j + 1] == x[i]:
            j += 1
        out.append((i, j, x[i]))
        i = j + 1
    return out


def ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    r2 = 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    return b, r2


def logit_auc(y01, X):
    import statsmodels.api as sm
    from scipy.stats import rankdata
    res = sm.Logit(y01, X).fit(disp=0, maxiter=200)
    sc = res.predict(X)
    rk = rankdata(sc)
    n1 = y01.sum(); n0 = len(y01) - n1
    return (rk[y01 == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def g0(df):
    res = {}
    dev = df.filter((pl.col("date") >= dt.date(2022, 1, 1)) & (pl.col("date") <= DEV_END))
    v1 = pl.read_parquet(FLOW / "mkt_timing_prereg" / "mkt_timing_signals.parquet", columns=["date", "mt1_state", "mt1_I"])
    j = dev.select("date", "mt1_state", "I").join(v1, on="date", how="left")
    agree = (j["mt1_state"] == j["mt1_state_right"].cast(pl.Float64))
    res["v1_agree"] = float(agree.mean())
    res["v1_diff_dates"] = [str(d) for d in j.filter(~agree)["date"].to_list()]
    st = dev["mt1_state"].to_numpy()
    assert not np.isnan(st).any()
    sw = np.where(st[1:] != st[:-1])[0] + 1
    yrs = dev["date"].dt.year().to_numpy()
    res["switches"] = int(len(sw))
    res["switches_by_year"] = {int(y): int((yrs[sw] == y).sum()) for y in np.unique(yrs)}
    R = runs(st)
    interior = R[1:-1]
    res["flicker_1d_interior"] = int(sum(1 for a, b, v in interior if b == a))
    res["flicker_1d_all"] = int(sum(1 for a, b, v in R if b == a))
    res["runs_ge30"] = int(sum(1 for a, b, v in R if b - a + 1 >= 30))
    res["runs_ge14"] = int(sum(1 for a, b, v in R if b - a + 1 >= 14))
    plus = (st > 0).sum()
    res["frac_plus_in_ge14_runs"] = float(sum(b - a + 1 for a, b, v in R if b - a + 1 >= 14 and v > 0) / plus)
    res["frac_plus"] = float((st > 0).mean())
    for name, end in [("main", dt.date(2026, 3, 17)), ("dev", DEV_END)]:
        d = dev.filter(pl.col("date") <= end)
        Xm = d.select(XCOLS).to_numpy().astype(float)
        assert not np.isnan(Xm).any(), name
        segd = np.column_stack([(d["seg"].to_numpy() == k).astype(float) for k in (2, 3, 4)])
        X = np.column_stack([np.ones(d.height), Xm, segd])
        y = d["mt1_state"].to_numpy()
        _, r2 = ols(y, X)
        n, k = X.shape
        res[f"lpm_r2_{name}"] = float(r2)
        res[f"lpm_adjr2_{name}"] = float(1 - (1 - r2) * (n - 1) / (n - k))
        res[f"n_{name}"] = int(n)
        res[f"logit_auc_{name}"] = float(logit_auc((y > 0).astype(float), X))
    res["g0_pass"] = max(res["lpm_r2_main"], res["lpm_r2_dev"]) <= 0.50
    # other-state agreement on main sample
    d = dev.filter(pl.col("date") <= dt.date(2026, 3, 17))
    for c in ["state_3c", "state_USDT", "state_USDC", "state_cry", "state_P", "state_SV", "state_OI"]:
        x = d.select("mt1_state", c).drop_nulls()
        res[f"agree_{c}"] = float((x["mt1_state"] == x[c]).mean())
    return res


if __name__ == "__main__":
    df = build()
    out = OUTD / "verify_features.parquet"
    assert not out.exists()
    df.write_parquet(out)
    r = g0(df)
    print(json.dumps(r, indent=1, default=str))
    p = OUTD / "verify_g0.json"
    assert not p.exists()
    p.write_text(json.dumps(r, indent=1, default=str))
