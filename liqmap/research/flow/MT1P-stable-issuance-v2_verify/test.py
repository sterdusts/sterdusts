"""Independent gate-1 re-computation for MT1P-stable-issuance-v2 (verify).

Labels are recomputed from close / funding_1d truncated to date <= 2026-03-31 and
cross-checked with fwd_14d - fund_14d. Only rows with d + h <= 2026-03-31 enter.
"""
import json
import pathlib
import datetime as dt

import numpy as np
import polars as pl

FLOW = pathlib.Path("/home/user/data/flow")
OUTD = FLOW / "MT1P-stable-issuance-v2_verify"
DEV_END = dt.date(2026, 3, 31)
D0 = dt.date(2022, 1, 1)
XCOLS = ["c_mom7", "c_mom30", "c_rv30", "c_fund7", "c_doi7", "c_tkr7", "fund30", "P_I", "mom90", "ma200", "dvol30"]
NW = 28


# ---------------------------------------------------------------- regression core
def nw_ols(y, X, lag, correction=False):
    """OLS + Newey-West (Bartlett) HAC. Returns beta, se. Uses pinv for rank-deficient X."""
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    V = XtXi @ S @ XtXi
    if correction:
        V *= n / (n - np.linalg.matrix_rank(X))
    se = np.sqrt(np.diag(V))
    return b, se


def design(d, state_col="mt1_state", xcols=XCOLS, seg=True, extra=()):
    cols = [np.ones(d.height), d[state_col].to_numpy().astype(float)]
    for c in extra:
        cols.append(d[c].to_numpy().astype(float))
    for c in xcols:
        cols.append(d[c].to_numpy().astype(float))
    if seg:
        sg = d["seg"].to_numpy()
        for kk in (2, 3, 4):
            if (sg == kk).any():
                cols.append((sg == kk).astype(float))
    return np.column_stack(cols)


def m0(d, ycol="y14", state_col="mt1_state", lag=NW, **kw):
    need = [ycol, state_col] + list(kw.get("xcols", XCOLS)) + list(kw.get("extra", ()))
    d = d.drop_nulls(need)
    X = design(d, state_col, **kw)
    y = d[ycol].to_numpy()
    b, se = nw_ols(y, X, lag)
    bc, sec = nw_ols(y, X, lag, correction=True)
    return {"beta": float(b[1]), "t": float(b[1] / se[1]), "t_corr": float(bc[1] / sec[1]), "n": d.height,
            "_b": b, "_se": se}


def fmt(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


# ---------------------------------------------------------------- labels
def labels():
    p = (pl.scan_parquet(FLOW / "perp_daily.parquet")
         .filter((pl.col("symbol") == "BTCUSDT") & (pl.col("date") <= DEV_END))
         .select("date", "close", "funding_1d", "fwd_14d", "fund_14d", "fwd_7d", "fund_7d")
         .collect().sort("date"))
    assert p["date"].max() <= DEV_END
    assert (p["date"].diff().drop_nulls().dt.total_days() == 1).all()
    lc = p["close"].log()
    fsum = p["funding_1d"].cum_sum()
    out = p.select("date")
    for h in (7, 14, 30):
        # fwd from truncated close; funding d+1..d+h = cum(d+h) - cum(d)
        y = (lc.shift(-h) - lc) - (fsum.shift(-h) - fsum)
        out = out.with_columns(y.alias(f"y{h}"))
    out = out.with_columns(
        (p["fwd_14d"] - p["fund_14d"]).alias("y14_file"), (p["fwd_7d"] - p["fund_7d"]).alias("y7_file"))
    # keep only d + h <= DEV_END
    for h in (7, 14, 30):
        out = out.with_columns(pl.when(pl.col("date") + pl.duration(days=h) <= DEV_END)
                               .then(pl.col(f"y{h}")).otherwise(None).alias(f"y{h}"))
    chk = out.filter(pl.col("y14").is_not_null() & (pl.col("date") >= D0))
    assert np.allclose(chk["y14"].to_numpy(), chk["y14_file"].to_numpy(), atol=1e-9)
    chk7 = out.filter(pl.col("y7").is_not_null() & (pl.col("date") >= D0))
    assert np.allclose(chk7["y7"].to_numpy(), chk7["y7_file"].to_numpy(), atol=1e-9)
    assert out.filter(pl.col("y14").is_not_null())["date"].max() == dt.date(2026, 3, 17)
    return out.drop("y14_file", "y7_file")


def label_0100():
    """y14_x(d) = ln(P[(d+15) 01:00]/P[(d+1) 01:00]) - sum FR BTCUSDT ts in ((d+1)01:00, (d+15)01:00]."""
    cut = int(dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
    k = (pl.scan_parquet("/home/user/data/kl5m/BTCUSDT.parquet").filter(pl.col("open_time") < cut)
         .select("open_time", "open").collect())
    assert k["open_time"].max() < cut
    px = dict(zip(k["open_time"].to_list(), k["open"].to_list()))
    fr = (pl.concat([pl.scan_parquet("/home/user/data/funding_*.parquet"),
                     pl.scan_parquet(FLOW / "perp_daily_aux_funding_backfill.parquet")], how="diagonal_relaxed")
          .filter(pl.col("symbol") == "BTCUSDT")
          .select("symbol", "calc_time", "funding_rate").unique()
          .with_columns((pl.col("calc_time") // 1000 * 1000).alias("ts"))
          .filter(pl.col("ts") < cut).collect().sort("ts"))
    assert fr["ts"].is_unique().all(), "dup ts with differing rates"
    assert fr["ts"].max() < cut
    ts = fr["ts"].to_numpy(); rate = fr["funding_rate"].to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(rate)])
    rows = []
    d = D0
    H = 3600_000
    while d <= dt.date(2026, 3, 16):
        a = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000) + 86400_000 + H
        b = a + 14 * 86400_000
        assert b < cut
        pa, pb = px.get(a), px.get(b)
        if pa is None or pb is None:
            rows.append((d, None)); d += dt.timedelta(days=1); continue
        ia = np.searchsorted(ts, a, side="right"); ib = np.searchsorted(ts, b, side="right")
        f = cs[ib] - cs[ia]
        rows.append((d, float(np.log(pb / pa) - f)))
        d += dt.timedelta(days=1)
    # also validate FR daily sum vs funding_1d
    return pl.DataFrame(rows, schema={"date": pl.Date, "y14_x": pl.Float64}, orient="row"), fr


def stationary_bootstrap_idx(n, p, rng):
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(n)
    for i in range(1, n):
        if rng.random() < p:
            idx[i] = rng.integers(n)
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


def main():
    f = pl.read_parquet(OUTD / "verify_features.parquet")
    lab = labels()
    lx, fr = label_0100()
    df = f.join(lab, on="date", how="left").join(lx, on="date", how="left")
    df = df.with_columns(pl.col("mt1_state").shift(1).alias("state_lag1"))
    R = {}
    main_s = df.filter((pl.col("date") >= D0) & (pl.col("date") <= dt.date(2026, 3, 17)))
    assert main_s["y14"].null_count() == 0
    r = m0(main_s); R["M0"] = fmt(r)
    b0, t0 = r["beta"], r["t"]
    R["P1_pass"] = b0 > 0 and t0 >= 1.65

    # P2 circular shift
    d = main_s.drop_nulls(["y14", "mt1_state"] + XCOLS)
    T = d.height
    X = design(d); y = d["y14"].to_numpy(); s0 = X[:, 1].copy()
    ts_ = []
    for k in range(90, T - 90 + 1):
        X[:, 1] = np.roll(s0, k)
        b, se = nw_ols(y, X, NW)
        ts_.append(b[1] / se[1])
    X[:, 1] = s0
    ts_ = np.array(ts_)
    R["P2"] = {"n_shifts": len(ts_), "q95": float(np.percentile(ts_, 95)), "median": float(np.median(ts_)),
               "frac_ge_obs": float((ts_ >= t0).mean())}
    R["P2_pass"] = t0 > R["P2"]["q95"]

    # P3 offsets
    off = (main_s["date"] - pl.lit(D0)).dt.total_days() % 14
    P3 = []
    for o in range(14):
        rr = m0(main_s.filter(off == o), lag=2)
        P3.append((rr["beta"], rr["t"]))
    R["P3"] = {"t": [round(x[1], 3) for x in P3], "beta": [x[0] for x in P3],
               "median_t": float(np.median([x[1] for x in P3])), "n_pos": int(sum(x[0] > 0 for x in P3))}
    R["P3_pass"] = R["P3"]["median_t"] >= 1.3 and R["P3"]["n_pos"] >= 12

    # P4 segments (no seg FE)
    P4 = {}
    for sgv in (1, 2, 3, 4):
        rr = m0(main_s.filter(pl.col("seg") == sgv), seg=False)
        P4[sgv] = {"beta": rr["beta"], "t": rr["t"], "n": rr["n"]}
    R["P4"] = P4
    npos = sum(P4[k]["beta"] > 0 for k in P4); npos234 = sum(P4[k]["beta"] > 0 for k in (2, 3, 4))
    R["P4_pass"] = npos >= 3 and npos234 >= 2

    # P5 lag1
    r5 = m0(main_s, state_col="state_lag1"); R["P5"] = fmt(r5)
    R["P5_pass"] = np.sign(r5["beta"]) == np.sign(b0) and r5["beta"] >= 0.5 * b0 if b0 > 0 else False

    # P6 no controls
    r6 = m0(main_s, xcols=[], seg=False); R["P6"] = fmt(r6)
    R["P6_pass"] = b0 >= 0.5 * r6["beta"]

    # P7 bootstrap
    rng = np.random.default_rng(20261010)
    bs = []
    for _ in range(2000):
        ix = stationary_bootstrap_idx(T, 1 / 60, rng)
        Xb = X[ix]; yb = y[ix]
        bb = np.linalg.pinv(Xb.T @ Xb) @ Xb.T @ yb
        bs.append(bb[1])
    bs = np.array(bs)
    R["P7"] = {"q05": float(np.percentile(bs, 5)), "median": float(np.median(bs)),
               "q95": float(np.percentile(bs, 95)), "frac_pos": float((bs > 0).mean())}
    R["P7_pass"] = R["P7"]["q05"] > 0

    # P8
    r8 = m0(main_s.filter(pl.col("date") >= dt.date(2022, 11, 8))); R["P8"] = fmt(r8)
    R["P8_pass"] = r8["beta"] > 0 and r8["t"] >= 1.0

    # P9 race
    s9 = main_s.filter(pl.col("date") >= dt.date(2022, 4, 30)).drop_nulls(["state_P", "state_OI", "state_SV"])
    r9a = m0(s9)
    r9b = m0(s9, extra=("state_P", "state_OI", "state_SV"))
    R["P9"] = {"same_sample_M0": fmt(r9a), "race_beta": r9b["beta"], "race_t": r9b["t"], "n": r9b["n"],
               "placebo_beta": [float(x) for x in r9b["_b"][2:5]],
               "placebo_t": [float(x) for x in (r9b["_b"][2:5] / r9b["_se"][2:5])]}
    R["P9_pass"] = r9b["beta"] >= 0.5 * r9a["beta"] and r9b["t"] >= 1.65

    # P10 MA200 strata
    xc10 = [c for c in XCOLS if c != "ma200"]
    R["P10"] = {}
    for mv in (1.0, 0.0):
        sub = main_s.filter(pl.col("ma200") == mv)
        a = m0(sub, xcols=xc10); bn = m0(sub, xcols=xc10, seg=False)
        R["P10"][str(int(mv))] = {"segFE": fmt(a), "noFE": fmt(bn)}
    R["P10_pass"] = all(R["P10"][k]["segFE"]["beta"] > 0 for k in ("1", "0"))

    # P11 g14 split
    s11 = main_s.with_columns(
        (pl.col("mt1_state") * (pl.col("g14") > 0).cast(pl.Float64)).alias("s_in"),
        (pl.col("mt1_state") * (pl.col("g14") <= 0).cast(pl.Float64)).alias("s_out"),
        (pl.col("g14") > 0).cast(pl.Float64).alias("g_pos"))
    r11 = m0(s11, state_col="s_in", extra=("s_out", "g_pos"))
    R["P11"] = {"beta_in": r11["beta"], "t_in": r11["t"], "beta_out": float(r11["_b"][2]),
                "t_out": float(r11["_b"][2] / r11["_se"][2]), "delta": float(r11["_b"][3]),
                "t_delta": float(r11["_b"][3] / r11["_se"][3])}
    R["P11_pass"] = r11["beta"] > 0 and r11["beta"] >= 0.5 * b0

    # P12 jackknife
    st = main_s["mt1_state"].to_numpy(); dates = main_s["date"].to_list()
    i = 0; dels = []
    while i < len(st):
        j = i
        while j + 1 < len(st) and st[j + 1] == st[i]:
            j += 1
        if j - i + 1 >= 30:
            dels.append((f"run {dates[i]}..{dates[j]}", dates[i], dates[j]))
        i = j + 1
    dels.append(("block 2023-10-07..2024-04-17", dt.date(2023, 10, 7), dt.date(2024, 4, 17)))
    for y_ in range(2022, 2027):
        for q in range(4):
            a = dt.date(y_, 3 * q + 1, 1)
            if a > dt.date(2026, 3, 17):
                break
            bq = (dt.date(y_ + (q == 3), (3 * q + 4) % 12 or 12, 1) - dt.timedelta(days=1)) if q < 3 else dt.date(y_, 12, 31)
            dels.append((f"{y_}Q{q+1}", a, bq))
    P12 = []
    for nm, a, b in dels:
        rr = m0(main_s.filter(~pl.col("date").is_between(a, b)))
        P12.append({"del": nm, "beta": rr["beta"], "t": rr["t"]})
    R["P12"] = {"n_del": len(P12), "n_runs_ge30": sum(1 for x in P12 if x["del"].startswith("run")),
                "n_flip": int(sum(x["beta"] <= 0 for x in P12)),
                "n_ok": int(sum(x["beta"] > 0 and x["t"] >= 1.0 for x in P12)),
                "min_beta": min(P12, key=lambda x: x["beta"]), "min_t": min(x["t"] for x in P12),
                "detail": P12}
    R["P12_pass"] = R["P12"]["n_ok"] == len(P12)

    # P13 01:00 entry
    s13 = main_s.filter(pl.col("y14_x").is_not_null())
    r13 = m0(s13, ycol="y14_x"); r13b = m0(s13)
    R["P13"] = {"y14_x": fmt(r13), "same_sample_M0": fmt(r13b), "missing_y14_x": int(main_s.filter(pl.col("date") <= dt.date(2026, 3, 16))["y14_x"].null_count())}
    R["P13_pass"] = np.sign(r13["beta"]) == np.sign(r13b["beta"]) and abs(r13["beta"]) >= 0.5 * abs(r13b["beta"])

    # falsification + components
    rc = m0(main_s, state_col="state_cry"); R["cry"] = fmt(rc)
    ru = m0(main_s, state_col="state_USDT"); R["USDT"] = fmt(ru)
    rcc = m0(main_s, state_col="state_USDC"); R["USDC"] = fmt(rcc)

    # kills
    K = {}
    K["K1"] = b0 <= 0 or abs(t0) < 1.0
    K["K2"] = (not R["P6_pass"]) or (r9b["beta"] < 0.5 * r9a["beta"]) or \
              (rc["beta"] > 0 and (rc["beta"] >= 0.8 * b0 or rc["t"] >= t0))
    K["K2_race_vs_fullM0"] = r9b["beta"] < 0.5 * b0
    K["K3"] = any(x["beta"] <= 0 for x in P12) if b0 > 0 else True
    K["K4"] = np.sign(r5["beta"]) != np.sign(b0)
    K["K6"] = r8["beta"] <= 0 or r8["t"] < 1.0
    K["K9"] = np.sign(ru["beta"]) != np.sign(rcc["beta"])
    K["K10"] = r11["beta"] <= 0
    K["K11"] = any(R["P10"][k]["segFE"]["beta"] <= 0 for k in ("1", "0"))
    R["kills"] = {k: bool(v) for k, v in K.items()}

    # report items
    R["h7"] = fmt(m0(df.filter((pl.col("date") >= D0) & pl.col("y7").is_not_null()), ycol="y7", lag=14))
    R["h30"] = fmt(m0(df.filter((pl.col("date") >= D0) & pl.col("y30").is_not_null()), ycol="y30", lag=60))
    R["state_3c"] = fmt(m0(main_s, state_col="state_3c"))
    R["c_doi7_oi_value"] = fmt(m0(main_s.with_columns(pl.col("c_doi7_val").alias("c_doi7"))))
    tab = (main_s.with_columns((pl.col("c_mom30") > 0).alias("tsm_pos"))
           .group_by("tsm_pos", "mt1_state").agg(pl.col("y14").mean(), pl.len()).sort("tsm_pos", "mt1_state"))
    R["tsmom_table"] = tab.to_dicts()
    # FR vs funding_1d check
    frd = (fr.with_columns(pl.from_epoch("ts", time_unit="ms").dt.date().alias("date"))
           .group_by("date").agg(pl.col("funding_rate").sum().alias("frsum")))
    cmp = frd.join(df.select("date", "funding_1d"), on="date", how="inner").filter(pl.col("date") >= D0)
    R["fr_vs_funding1d_maxabs"] = float((cmp["frsum"] - cmp["funding_1d"]).abs().max())
    R["fr_max_ts"] = int(fr["ts"].max())
    R["gate1_pass"] = all(R[f"P{i}_pass"] for i in range(1, 14))
    R["pass_flags"] = {f"P{i}": bool(R[f"P{i}_pass"]) for i in range(1, 14)}
    out = OUTD / "verify_gate1.json"
    assert not out.exists()
    out.write_text(json.dumps(R, indent=1, default=str))
    print(json.dumps({k: v for k, v in R.items() if k not in ("P12",)}, indent=1, default=str))
    print(json.dumps({k: v for k, v in R["P12"].items() if k != "detail"}, indent=1, default=str))


if __name__ == "__main__":
    main()
