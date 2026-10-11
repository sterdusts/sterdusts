"""MT1P-stable-issuance-v2 仲裁复算（独立实现，不复用实现者/复算者代码）。

规格：/home/user/sterdusts/liqmap/research/flow_prereg_2026-10-10.json, id="MT1P-stable-issuance-v2"（commit b3f95f8）。
争议点：P5/K4 的滞后状态样本（实现者在 date>=2022-01-01 过滤后再 shift(1)，丢掉 2022-01-01，n=1536；
复算者按日历滞后，2022-01-01 取 2021-12-31 的状态，n=1537）。本脚本两种都算，并重算其余全部关卡 1 项与淘汰条件。

锁定集保护：
- stablecoins_daily / perp_daily(BTCUSDT) / spot_daily_base 在 scan 时即过滤 date<=2026-03-31，并断言；
- 标签只由过滤后的 close 与 funding_1d 计算，且只保留 d+h<=2026-03-31（y14: d<=2026-03-17）；与 fwd_14d-fund_14d 逐行核对；
- kl5m 过滤 open_time<2026-04-01 00:00 UTC，资金费结算表过滤 calc_time<2026-04-01 00:00 UTC，断言每个窗口终点 <2026-04-01；
- 平移/自助/起点子样本都只用同一个截断主样本。
输出：本目录 arb_result.json（拒绝覆盖）。运行：python3 -I arb.py
"""
import datetime as dt
import glob
import json
import pathlib

import numpy as np
import polars as pl

FLOW = pathlib.Path("/home/user/data/flow")
ARB = FLOW / "MT1P-stable-issuance-v2_arb"
IMPL_SIG = FLOW / "MT1P-stable-issuance-v2_run" / "mkt_timing_signals_v2.parquet"
OUT = ARB / "arb_result.json"
DEV0, DEV_END, END14, ENDX = dt.date(2022, 1, 1), dt.date(2026, 3, 31), dt.date(2026, 3, 17), dt.date(2026, 3, 16)
HOLD_MS = int(dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
EXCL = {"BTC", "USDC", "FDUSD", "BUSD", "TUSD", "U", "USD1", "USTC", "PAXG", "XAUT"}
XC = ["c_mom7", "c_mom30", "c_rv30", "c_fund7", "c_doi7", "c_tkr7", "fund30", "P_I", "mom90", "ma200", "dvol30"]
LAG = 28
SEED = 20261010


# ------------------------------------------------------------------ 基础数组工具（numpy，独立实现）
def lagged(x, k):
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


def Tx(x):
    """T(X)(D) = [X(D)-X(D-14)] - 14/166 [X(D-14)-X(D-180)]。"""
    return (x - lagged(x, 14)) - (14.0 / 166.0) * (lagged(x, 14) - lagged(x, 180))


def sgn(x):
    return np.where(np.isnan(x), np.nan, np.where(x > 0, 1.0, -1.0))


def ffill_first0(v):
    v = v.astype(float).copy()
    last = np.nan
    for i in range(len(v)):
        if np.isnan(v[i]):
            v[i] = last
        else:
            last = v[i]
    return np.where(np.isnan(v), 0.0, v)


def ffill_limit(v, lim):
    v = v.astype(float).copy(); out = v.copy(); last, age = np.nan, 0
    for i in range(len(v)):
        if np.isnan(v[i]):
            age += 1
            out[i] = last if age <= lim else np.nan
        else:
            last, age = v[i], 0
    return out


def L_of(level):
    """ln level -> 日差（首日 0）-> clip ±1% -> 累加。"""
    ln = np.log(level)
    dl = np.zeros(len(ln)); dl[1:] = ln[1:] - ln[:-1]
    return np.cumsum(np.clip(dl, -0.01, 0.01))


def roll(x, w, minp, fn):
    out = np.full(len(x), np.nan)
    for i in range(w - 1, len(x)):
        seg = x[i - w + 1:i + 1]; seg = seg[~np.isnan(seg)]
        if len(seg) >= minp:
            out[i] = fn(seg)
    return out


# ------------------------------------------------------------------ 特征
def build_features():
    cal = pl.DataFrame({"date": pl.date_range(dt.date(2020, 1, 1), DEV_END, "1d", eager=True)})
    s = (pl.scan_parquet(FLOW / "stablecoins_daily.parquet").filter(pl.col("date") <= DEV_END)
         .select("date", "usdt_total", "usdc_total", "busd_total", "fdusd_total", "tusd_total",
                 "dai_total", "usds_total", "usde_total").collect().sort("date"))
    assert s["date"].max() <= DEV_END and s["date"].min() == dt.date(2020, 1, 1) and s.height == cal.height
    x = {c: ffill_first0(s[c].to_numpy()) for c in s.columns if c != "date"}
    F = sum(x[c] for c in ["usdt_total", "usdc_total", "busd_total", "fdusd_total", "tusd_total"])
    L = L_of(F)
    I = (L - lagged(L, 14)) - (14 / 166) * (lagged(L, 14) - lagged(L, 180))
    g14 = L - lagged(L, 14)
    f = {"date": s["date"].to_numpy(), "I": I, "g14": g14, "mt1_state": sgn(I),
         "state_USDT": sgn(Tx(L_of(x["usdt_total"]))), "state_USDC": sgn(Tx(L_of(x["usdc_total"]))),
         "state_cry": sgn(Tx(L_of(x["dai_total"] + x["usds_total"] + x["usde_total"])))}

    p = (pl.scan_parquet(FLOW / "perp_daily.parquet")
         .filter((pl.col("symbol") == "BTCUSDT") & (pl.col("date") <= DEV_END))
         .select("date", "close", "ret_1d", "funding_1d", "oi", "oi_value", "quote_volume").collect().sort("date"))
    assert p["date"].max() <= DEV_END
    p = cal.join(p, on="date", how="left")
    C = p["close"].to_numpy().astype(float); lc = np.log(C)
    r1 = p["ret_1d"].to_numpy().astype(float); fu = p["funding_1d"].to_numpy().astype(float)
    oi = p["oi"].to_numpy().astype(float); oiv = p["oi_value"].to_numpy().astype(float)
    qv = p["quote_volume"].to_numpy().astype(float)
    sma200 = roll(C, 200, 200, np.mean)
    f.update({
        "c_mom7": lc - lagged(lc, 7), "c_mom30": lc - lagged(lc, 30),
        "c_rv30": roll(r1, 30, 25, lambda a: np.std(a, ddof=1)),
        "c_fund7": roll(fu, 7, 7, np.sum), "c_doi7": np.log(oi) - lagged(np.log(oi), 7),
        "c_doi7_oivalue": np.log(oiv) - lagged(np.log(oiv), 7),
        "fund30": roll(fu, 30, 30, np.sum), "P_I": Tx(lc), "mom90": lc - lagged(lc, 90),
        "ma200": np.where(np.isnan(sma200), np.nan, (C > sma200).astype(float)),
        "dvol30": np.log(roll(qv, 7, 7, np.mean) / roll(qv, 90, 90, np.mean)),
        "state_P": sgn(Tx(lc)), "state_OI": sgn(Tx(np.log(ffill_limit(oiv, 3)))),
        "close": C, "funding_1d": fu,
    })

    # 现货：B_t 按「存在 spot_map 行 spot_base==base & status=='ok' & perp_first_live<=t」逐行判定
    smap = pl.read_parquet(FLOW / "spot_map.parquet").filter(pl.col("status") == "ok").select("spot_base", "perp_first_live")
    b = (pl.scan_parquet(FLOW / "spot_daily_base.parquet").filter(pl.col("date") <= DEV_END)
         .select("date", "base", "qv_usdt_USDT", "tbqv_usdt_USDT").collect())
    assert b["date"].max() <= DEV_END
    b = b.filter(~pl.col("base").is_in(list(EXCL)))
    elig = (b.select("date", "base").unique()
            .join(smap, left_on="base", right_on="spot_base", how="inner")
            .filter(pl.col("perp_first_live") <= pl.col("date"))
            .select("date", "base").unique())
    b = b.join(elig, on=["date", "base"], how="semi")
    agg = b.group_by("date").agg(pl.col("qv_usdt_USDT").sum().alias("qv"), pl.col("tbqv_usdt_USDT").sum().alias("tb"))
    agg = cal.join(agg, on="date", how="left")
    qvs, tbs = agg["qv"].to_numpy().astype(float), agg["tb"].to_numpy().astype(float)
    q7, t7 = roll(qvs, 7, 7, np.sum), roll(tbs, 7, 7, np.sum)
    SV = np.log(q7)
    f.update({"c_tkr7": t7 / q7 - 0.5, "state_SV": sgn(Tx(SV))})
    f["seg"] = np.array([min(d.year, 2025) if d.year >= 2022 else 2022 for d in s["date"].to_list()])
    return pl.DataFrame(f).with_columns(pl.col("date").cast(pl.Date))


# ------------------------------------------------------------------ 标签
def labels(feat):
    p = (pl.scan_parquet(FLOW / "perp_daily.parquet")
         .filter((pl.col("symbol") == "BTCUSDT") & (pl.col("date") <= DEV_END))
         .select("date", "fwd_14d", "fund_14d").collect().sort("date"))
    assert p["date"].max() <= DEV_END
    d = feat["date"].to_list(); C = feat["close"].to_numpy(); fu = feat["funding_1d"].to_numpy()
    idx = {x: i for i, x in enumerate(d)}
    y14 = np.full(len(d), np.nan)
    for i, x in enumerate(d):
        if DEV0 <= x and x + dt.timedelta(days=14) <= DEV_END:
            assert d[i + 14] <= DEV_END
            y14[i] = np.log(C[i + 14] / C[i]) - fu[i + 1:i + 15].sum()
    out = feat.select("date").with_columns(pl.Series("y14", y14).fill_nan(None))
    chk = out.join(p, on="date").filter(pl.col("y14").is_not_null())
    assert np.allclose(chk["y14"].to_numpy(), (chk["fwd_14d"] - chk["fund_14d"]).to_numpy(), atol=1e-9)
    assert chk["date"].max() == END14
    return out, float(np.abs(chk["y14"].to_numpy() - (chk["fwd_14d"] - chk["fund_14d"]).to_numpy()).max())


def labels_0100():
    H = 3_600_000; DAY = 86_400_000
    k = (pl.scan_parquet("/home/user/data/kl5m/BTCUSDT.parquet")
         .filter((pl.col("open_time") < HOLD_MS) & ((pl.col("open_time") % DAY) == H))
         .select("open_time", "open").collect())
    assert k["open_time"].max() < HOLD_MS
    px = dict(zip(k["open_time"].to_list(), k["open"].to_list()))
    srcs = sorted(glob.glob("/home/user/data/funding_*.parquet")) + [str(FLOW / "perp_daily_aux_funding_backfill.parquet")]
    fr = (pl.concat([pl.scan_parquet(x).filter((pl.col("symbol") == "BTCUSDT") & (pl.col("calc_time") < HOLD_MS))
                     .select("calc_time", "funding_rate") for x in srcs]).unique().collect().sort("calc_time"))
    assert fr["calc_time"].is_unique().all() and fr["calc_time"].max() < HOLD_MS
    ct = fr["calc_time"].to_numpy(); cr = np.concatenate([[0.0], np.cumsum(fr["funding_rate"].to_numpy())])
    rows, d = [], DEV0
    while d <= ENDX:
        a = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp() * 1000) + DAY + H
        b = a + 14 * DAY
        assert b < HOLD_MS
        pa, pb = px.get(a), px.get(b)
        fs = cr[np.searchsorted(ct, b, side="right")] - cr[np.searchsorted(ct, a, side="right")]
        rows.append((d, None if pa is None or pb is None else float(np.log(pb / pa) - fs)))
        d += dt.timedelta(days=1)
    return pl.DataFrame(rows, schema={"date": pl.Date, "y14_x": pl.Float64}, orient="row"), int(fr["calc_time"].max())


# ------------------------------------------------------------------ 回归
def nw(y, Z, lag=LAG):
    """OLS + Newey-West Bartlett（无小样本修正，与 statsmodels HAC 默认一致），伪逆处理秩亏。"""
    XtXi = np.linalg.pinv(Z.T @ Z); b = XtXi @ (Z.T @ y); u = y - Z @ b
    g = Z * u[:, None]; S = g.T @ g
    for l in range(1, lag + 1):
        G = g[l:].T @ g[:-l]; S += (1 - l / (lag + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi))
    return b, se


def Zmat(d, regs, xc=XC, seg=True):
    cols = [np.ones(d.height)] + [d[r].to_numpy().astype(float) for r in regs] + [d[c].to_numpy().astype(float) for c in xc]
    if seg:
        sg = d["seg"].to_numpy()
        cols += [(sg == k).astype(float) for k in (2023, 2024, 2025) if (sg == k).any()]
    return np.column_stack(cols)


def reg(d, regs=("mt1_state",), y="y14", xc=XC, seg=True, lag=LAG):
    d = d.drop_nulls([y, *regs, *xc])
    b, se = nw(d[y].to_numpy().astype(float), Zmat(d, regs, xc, seg), lag)
    out = {"beta": float(b[1]), "t": float(b[1] / se[1]), "n": d.height}
    if len(regs) > 1:
        out["all"] = {r: {"beta": float(b[1 + i]), "t": float(b[1 + i] / se[1 + i])} for i, r in enumerate(regs)}
    return out


def main():
    assert not OUT.exists(), "输出已存在，拒绝覆盖"
    R = {"spec_id": "MT1P-stable-issuance-v2", "spec_commit": "b3f95f8"}
    feat = build_features()
    lab, ymax = labels(feat)
    lx, frmax = labels_0100()
    R["holdout"] = {"y14_vs_file_maxabs": ymax, "fr_max_calc_time": frmax, "last_label_d": str(END14)}

    # ---- 与实现者信号文件逐列核对（全期 2020-01-01..2026-03-31 的状态；主样本的控制）
    imp = pl.read_parquet(IMPL_SIG).filter(pl.col("date") <= DEV_END)
    j = feat.join(imp.select("date", *[pl.col(c).alias(c + "_imp") for c in
                                      ["mt1_state", "state_USDT", "state_USDC", "state_cry", "state_P", "state_OI",
                                       "state_SV", *XC]]), on="date", how="left")
    cmp = {}
    for c in ["mt1_state", "state_USDT", "state_USDC", "state_cry", "state_P", "state_OI", "state_SV"]:
        q = j.select(c, c + "_imp").drop_nulls()
        cmp[c] = {"n": q.height, "mismatch": int((q[c] != q[c + "_imp"].cast(pl.Float64)).sum())}
    jm = j.filter(pl.col("date").is_between(DEV0, END14))
    for c in XC:
        a, bb = jm[c].to_numpy().astype(float), jm[c + "_imp"].to_numpy().astype(float)
        cmp[c] = {"max_absdiff": float(np.nanmax(np.abs(a - bb))), "null_mismatch": int((np.isnan(a) != np.isnan(bb)).sum())}
    R["feature_check_vs_impl"] = cmp

    df = feat.join(lab, on="date", how="left").join(lx, on="date", how="left")
    # P5 两种滞后：(a) 日历滞后，在全期特征上 shift（2022-01-01 取 2021-12-31 状态）；(b) 实现者做法：先过滤到 >=2022-01-01 再 shift
    df = df.with_columns(pl.col("mt1_state").shift(1).alias("lag1_calendar"))
    df = df.with_columns(
        pl.when(pl.col("date") >= DEV0).then(pl.col("mt1_state")).otherwise(None).shift(1).alias("lag1_insample"),
        (pl.col("g14") > 0).cast(pl.Float64).alias("gpos"))
    df = df.with_columns((pl.col("mt1_state") * pl.col("gpos")).alias("s_in"),
                         (pl.col("mt1_state") * (1 - pl.col("gpos"))).alias("s_out"))
    M = df.filter(pl.col("date").is_between(DEV0, END14)).drop_nulls(["y14", "mt1_state", *XC])
    assert M.height == 1537 and M["date"].max() == END14 and M["date"].min() == DEV0
    st2021 = df.filter(pl.col("date") == dt.date(2021, 12, 31))["mt1_state"].item()
    st2022 = df.filter(pl.col("date") == DEV0)["mt1_state"].item()

    m0 = reg(M); b0, t0 = m0["beta"], m0["t"]; R["P1_M0"] = m0
    try:
        import statsmodels.api as sm
        d_ = M; Z_ = Zmat(d_, ["mt1_state"])
        R["P1_M0_statsmodels_t"] = float(sm.OLS(d_["y14"].to_numpy(), Z_).fit(cov_type="HAC", cov_kwds={"maxlags": LAG}).tvalues[1])
    except Exception as e:  # noqa
        R["P1_M0_statsmodels_t"] = str(e)
    P = {"P1": b0 > 0 and t0 >= 1.65}

    # P2
    Z = Zmat(M, ["mt1_state"]); y = M["y14"].to_numpy(); s0 = Z[:, 1].copy(); T = len(y); ts = []
    for k in range(90, T - 90 + 1):
        Z[:, 1] = np.roll(s0, k); b, se = nw(y, Z); ts.append(b[1] / se[1])
    Z[:, 1] = s0; ts = np.array(ts)
    R["P2"] = {"n_shifts": len(ts), "q95": float(np.quantile(ts, 0.95)), "median": float(np.median(ts)),
               "frac_ge_obs": float((ts >= t0).mean())}
    P["P2"] = t0 > R["P2"]["q95"]

    # P3
    off = M.select(((pl.col("date") - pl.lit(DEV0)).dt.total_days() % 14).alias("o"))["o"]
    p3 = [reg(M.filter(off == o), lag=2) for o in range(14)]
    R["P3"] = {"t": [round(r["t"], 3) for r in p3], "median_t": float(np.median([r["t"] for r in p3])),
               "n_pos": int(sum(r["beta"] > 0 for r in p3))}
    P["P3"] = R["P3"]["median_t"] >= 1.3 and R["P3"]["n_pos"] >= 12

    # P4（不加 seg FE）
    p4 = {sg: reg(M.filter(pl.col("seg") == sg), seg=False) for sg in (2022, 2023, 2024, 2025)}
    R["P4"] = {str(k): v for k, v in p4.items()}
    P["P4"] = sum(v["beta"] > 0 for v in p4.values()) >= 3 and sum(p4[k]["beta"] > 0 for k in (2023, 2024, 2025)) >= 2

    # P5（争议点）
    p5a = reg(M, regs=("lag1_calendar",)); p5b = reg(M, regs=("lag1_insample",))
    R["P5"] = {"calendar_lag_spec_literal": p5a, "insample_lag_impl_style": p5b,
               "state_2021-12-31": st2021, "state_2022-01-01": st2022}
    P["P5"] = bool(np.sign(p5a["beta"]) == np.sign(b0) and p5a["beta"] >= 0.5 * b0)
    R["P5_pass_under_insample_lag"] = bool(np.sign(p5b["beta"]) == np.sign(b0) and p5b["beta"] >= 0.5 * b0)

    # P6
    p6 = reg(M, xc=[], seg=False); R["P6"] = p6; P["P6"] = b0 >= 0.5 * p6["beta"]

    # P7（平稳块自助，平均块长 60，2000 次，按行联合重抽）
    rng = np.random.default_rng(SEED); bs = np.empty(2000)
    for i in range(2000):
        ix = np.empty(T, dtype=np.int64); ix[0] = rng.integers(T)
        new = rng.random(T) < 1 / 60; jump = rng.integers(T, size=T)
        for t_ in range(1, T):
            ix[t_] = jump[t_] if new[t_] else (ix[t_ - 1] + 1) % T
        bs[i] = np.linalg.lstsq(Z[ix], y[ix], rcond=None)[0][1]
    R["P7"] = {"q05": float(np.quantile(bs, 0.05)), "median": float(np.median(bs)), "q95": float(np.quantile(bs, 0.95)),
               "frac_pos": float((bs > 0).mean())}
    P["P7"] = R["P7"]["q05"] > 0

    # P8
    p8 = reg(M.filter(pl.col("date") >= dt.date(2022, 11, 8))); R["P8"] = p8
    P["P8"] = p8["beta"] > 0 and p8["t"] >= 1.0

    # P9
    M9 = M.filter(pl.col("date") >= dt.date(2022, 4, 30))
    assert M9.select("state_P", "state_OI", "state_SV").null_count().sum_horizontal().item() == 0
    m9 = reg(M9); r9 = reg(M9, regs=("mt1_state", "state_P", "state_OI", "state_SV"))
    R["P9"] = {"M0_same_sample": m9, "race": r9}
    P["P9"] = r9["beta"] >= 0.5 * m9["beta"] and r9["t"] >= 1.65

    # P10（判定用 M0 形式：保留 seg FE、X 去 ma200；另报不加 seg FE）
    xc10 = [c for c in XC if c != "ma200"]
    p10 = {v: reg(M.filter(pl.col("ma200") == v), xc=xc10) for v in (1.0, 0.0)}
    p10n = {v: reg(M.filter(pl.col("ma200") == v), xc=xc10, seg=False) for v in (1.0, 0.0)}
    R["P10"] = {"segFE": {str(int(k)): v for k, v in p10.items()}, "noFE": {str(int(k)): v for k, v in p10n.items()}}
    P["P10"] = all(v["beta"] > 0 for v in p10.values())

    # P11
    r11 = reg(M, regs=("s_in", "s_out", "gpos")); R["P11"] = r11
    P["P11"] = r11["beta"] > 0 and r11["beta"] >= 0.5 * b0

    # P12
    st = M["mt1_state"].to_numpy(); dts = M["date"].to_list(); dels = []; i = 0
    while i < len(st):
        j_ = i
        while j_ + 1 < len(st) and st[j_ + 1] == st[i]:
            j_ += 1
        if j_ - i + 1 >= 30:
            dels.append((f"run {dts[i]}..{dts[j_]}", dts[i], dts[j_]))
        i = j_ + 1
    nruns = len(dels)
    dels.append(("2023-10-07..2024-04-17", dt.date(2023, 10, 7), dt.date(2024, 4, 17)))
    for yy in range(2022, 2027):
        for q in range(4):
            a = dt.date(yy, 3 * q + 1, 1)
            if a > END14:
                break
            e = dt.date(yy, 12, 31) if q == 3 else dt.date(yy, 3 * q + 4, 1) - dt.timedelta(days=1)
            dels.append((f"{yy}Q{q + 1}", a, e))
    jk = []
    for nm, a, e in dels:
        r = reg(M.filter(~pl.col("date").is_between(a, e))); r["drop"] = nm; jk.append(r)
    R["P12"] = {"n_runs_ge30": nruns, "n_del": len(jk), "n_flip": int(sum(r["beta"] <= 0 for r in jk)),
                "n_ok": int(sum(r["beta"] > 0 and r["t"] >= 1.0 for r in jk)),
                "min_beta": min(jk, key=lambda r: r["beta"]), "detail": jk}
    P["P12"] = R["P12"]["n_ok"] == len(jk)

    # P13
    M13 = M.filter(pl.col("y14_x").is_not_null())
    x13 = reg(M13, y="y14_x"); m13 = reg(M13)
    R["P13"] = {"y14_x": x13, "M0_same_sample": m13,
                "missing": int(df.filter(pl.col("date").is_between(DEV0, ENDX))["y14_x"].null_count())}
    P["P13"] = bool(np.sign(x13["beta"]) == np.sign(m13["beta"]) and x13["beta"] >= 0.5 * m13["beta"])

    # 证伪 / 成分 / c_doi7 敏感性
    rc = reg(M, regs=("state_cry",)); ru = reg(M, regs=("state_USDT",)); rC = reg(M, regs=("state_USDC",))
    R["falsify_cry"], R["comp_USDT"], R["comp_USDC"] = rc, ru, rC
    R["sens_c_doi7_oi_value"] = reg(M.with_columns(pl.col("c_doi7_oivalue").alias("c_doi7")))

    R["gate1"] = {k: bool(v) for k, v in P.items()}
    R["gate1_pass"] = bool(all(P.values()))
    K = {
        "K1": b0 <= 0 or abs(t0) < 1.0,
        "K2": (not P["P6"]) or (r9["beta"] < 0.5 * m9["beta"]) or (rc["beta"] > 0 and (rc["beta"] >= 0.8 * b0 or rc["t"] >= t0)),
        "K3": any(r["beta"] <= 0 for r in jk),
        "K4": bool(np.sign(p5a["beta"]) != np.sign(b0)),
        "K6": p8["beta"] <= 0 or p8["t"] < 1.0,
        "K9": bool(np.sign(ru["beta"]) != np.sign(rC["beta"])),
        "K10": r11["beta"] <= 0,
        "K11": any(v["beta"] <= 0 for v in p10.values()),
    }
    R["kill"] = {k: bool(v) for k, v in K.items()}
    R["kill_K2_legs"] = {"P6_fail": bool(not P["P6"]), "race_lt_half_sameM0": bool(r9["beta"] < 0.5 * m9["beta"]),
                         "race_lt_half_fullM0": bool(r9["beta"] < 0.5 * b0),
                         "cry_leg": bool(rc["beta"] > 0 and (rc["beta"] >= 0.8 * b0 or rc["t"] >= t0))}
    R["K4_under_insample_lag"] = bool(np.sign(p5b["beta"]) != np.sign(b0))
    R["killed"] = bool(any(K.values()))
    OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=str))
    short = {k: v for k, v in R.items() if k not in ("P12",)}
    print(json.dumps(short, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({k: v for k, v in R["P12"].items() if k != "detail"}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
