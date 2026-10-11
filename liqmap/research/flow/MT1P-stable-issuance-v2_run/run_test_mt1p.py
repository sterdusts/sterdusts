"""MT1' (MT1P-stable-issuance-v2) 关卡 1 主检验 + 次检验 + 报告项 + 淘汰条件判定。

试验名 mt1p_stable_issuance_v2（trials.jsonl 已由主会话预登记；本脚本只读断言登记存在，不写 trials.jsonl）。
规格：/home/user/sterdusts/liqmap/research/flow_prereg_2026-10-10.json（commit b3f95f8）。

锁定集保护（holdout guard）：
- 所有收益相关输入（perp_daily BTCUSDT 的 close/funding/fwd/fund、kl5m、资金费结算表）在读取时即过滤到
  日期 <= 2026-03-31（kl5m / 资金费：时间戳 < 2026-04-01 00:00 UTC），并断言读到的最大价格日期、资金费日期 <= 2026-03-31。
- 主标签只取 d <= 2026-03-17（d+14 <= 2026-03-31，断言）；01:00 版 d <= 2026-03-16；h=7 d <= 2026-03-24；h=30 d <= 2026-03-01。
- 不使用 v1 的 in_dev 列。标签另由过滤后的 close/funding 独立重算并与 fwd_14d-fund_14d 逐行核对。
运行：python3 -I run_test_mt1p.py
"""
import datetime as dt
import glob
import json
import pathlib

import numpy as np
import polars as pl
import statsmodels.api as sm

RUN = pathlib.Path("/home/user/data/flow/MT1P-stable-issuance-v2_run")
FLOW = pathlib.Path("/home/user/data/flow")
SIG = RUN / "mkt_timing_signals_v2.parquet"
TRIALS = pathlib.Path("/home/user/sterdusts/liqmap/research/trials.jsonl")
OUT = RUN / "gate1_result.json"
DEV0, DEV_END = dt.date(2022, 1, 1), dt.date(2026, 3, 31)
END = {14: dt.date(2026, 3, 17), 7: dt.date(2026, 3, 24), 30: dt.date(2026, 3, 1), "x": dt.date(2026, 3, 16)}
HOLD_MS = int(dt.datetime(2026, 4, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
XCOLS = ["c_mom7", "c_mom30", "c_rv30", "c_fund7", "c_doi7", "c_tkr7", "fund30", "P_I", "mom90", "ma200", "dvol30"]
LAG = 28
SEED = 20261010


# ----------------------------------------------------------------------------- 回归工具
def design(df: pl.DataFrame, regs, xcols=XCOLS, seg_fe=True):
    cols = [np.ones(df.height)] + [df[c].to_numpy().astype(float) for c in regs]
    cols += [df[c].to_numpy().astype(float) for c in xcols]
    if seg_fe:
        segs = sorted(df["seg"].unique().to_list())
        cols += [(df["seg"] == s).to_numpy().astype(float) for s in segs[1:]]
    return np.column_stack(cols)


def ols(df: pl.DataFrame, y="y14", regs=("mt1_state",), xcols=XCOLS, seg_fe=True, lag=LAG):
    """返回 regs[0] 的 β、NW(Bartlett) t、n；以及全部 regs 的系数。"""
    d = df.drop_nulls([y, *regs, *xcols])
    Z = design(d, list(regs), xcols, seg_fe)
    yy = d[y].to_numpy().astype(float)
    f = sm.OLS(yy, Z).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    out = {"beta": float(f.params[1]), "t": float(f.tvalues[1]), "n": int(d.height)}
    if len(regs) > 1:
        out["all"] = {r: {"beta": float(f.params[1 + i]), "t": float(f.tvalues[1 + i])} for i, r in enumerate(regs)}
    return out


def beta_only(yy, Z):
    return np.linalg.lstsq(Z, yy, rcond=None)[0][1]


def runs(x):
    out, s = [], 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[s]:
            out.append((x[s], s, i - s))
            s = i
    return out


# ----------------------------------------------------------------------------- 数据
def assert_registered():
    rows = [json.loads(l) for l in TRIALS.read_text().splitlines() if l.strip()]
    hit = [r for r in rows if r.get("name") == "mt1p_stable_issuance_v2"]
    assert hit, "trials.jsonl 中没有 mt1p_stable_issuance_v2 登记，拒绝运行"
    return {"n_trials_rows": len(rows), "registration": hit[-1]}


def load_labels() -> pl.DataFrame:
    p = (pl.scan_parquet(FLOW / "perp_daily.parquet")
         .filter((pl.col("symbol") == "BTCUSDT") & (pl.col("date") <= DEV_END))  # 锁定集保护
         .select("date", "close", "funding_1d", "fwd_7d", "fund_7d", "fwd_14d", "fund_14d")
         .collect().sort("date"))
    assert p["date"].max() <= DEV_END and p["close"].null_count() == 0 and p["funding_1d"].null_count() == 0
    assert (p["date"].diff().drop_nulls() == dt.timedelta(days=1)).all()
    lc = pl.col("close").log()
    fcum = pl.col("funding_1d").cum_sum()
    p = p.with_columns(
        *[((lc.shift(-h) - lc) - (fcum.shift(-h) - fcum)).alias(f"yre{h}") for h in (7, 14, 30)],
        pl.col("date").dt.offset_by("14d").alias("d14"))
    p = p.with_columns(
        pl.when(pl.col("date") <= END[14]).then(pl.col("fwd_14d") - pl.col("fund_14d")).alias("y14"),
        pl.when(pl.col("date") <= END[7]).then(pl.col("fwd_7d") - pl.col("fund_7d")).alias("y7"),
        pl.when(pl.col("date") <= END[30]).then(pl.col("yre30")).alias("y30"),
    )
    # 独立重算核对（重算只用到 <= 2026-03-31 的 close/funding）
    q = p.filter(pl.col("date").is_between(DEV0, END[14]))
    assert q["y14"].null_count() == 0
    assert np.allclose(q["y14"].to_numpy(), q["yre14"].to_numpy(), atol=1e-9), "y14 与独立重算不一致"
    q7 = p.filter(pl.col("date").is_between(DEV0, END[7]))
    assert np.allclose(q7["y7"].to_numpy(), q7["yre7"].to_numpy(), atol=1e-9)
    assert (q["d14"] <= DEV_END).all(), "d+14 越过开发期"
    assert p.filter(pl.col("y30").is_not_null())["date"].max() <= END[30]
    return p.select("date", "y7", "y14", "y30")


def load_exec_labels() -> pl.DataFrame:
    """y14_x(d) = ln(P[(d+15)01:00]/P[(d+1)01:00]) - fund_exec(BTCUSDT, ((d+1)01:00, (d+15)01:00]]。"""
    k = (pl.scan_parquet("/home/user/data/kl5m/BTCUSDT.parquet")
         .filter(pl.col("open_time") < HOLD_MS)  # 锁定集保护
         .select("open_time", "open").collect())
    assert k["open_time"].max() < HOLD_MS
    fs = sorted(glob.glob("/home/user/data/funding_*.parquet")) + [str(FLOW / "perp_daily_aux_funding_backfill.parquet")]
    fr = pl.concat([pl.scan_parquet(x).filter((pl.col("symbol") == "BTCUSDT") & (pl.col("calc_time") < HOLD_MS))
                    .select("calc_time", "funding_rate").collect() for x in fs]).unique("calc_time").sort("calc_time")
    assert fr["calc_time"].max() < HOLD_MS
    px = dict(zip(k["open_time"].to_list(), k["open"].to_list()))
    ct = fr["calc_time"].to_numpy(); cr = np.concatenate([[0.0], np.cumsum(fr["funding_rate"].to_numpy())])

    def fsum(a, b):  # (a, b]
        return cr[np.searchsorted(ct, b, side="right")] - cr[np.searchsorted(ct, a, side="right")]

    rows = []
    d = DEV0
    while d <= END["x"]:
        a = int(dt.datetime(d.year, d.month, d.day, 1, tzinfo=dt.timezone.utc).timestamp() * 1000) + 86400000
        b = a + 14 * 86400000
        assert b < HOLD_MS
        pa, pb = px.get(a), px.get(b)
        y = None if (pa is None or pb is None) else float(np.log(pb / pa) - fsum(a, b))
        rows.append((d, y))
        d += dt.timedelta(days=1)
    # 资金费表与 funding_1d 一致性核对（开发期，按 [d 00:00, d+1 00:00) 归属）
    return pl.DataFrame(rows, schema={"date": pl.Date, "y14_x": pl.Float64}, orient="row"), fr


def check_fr(fr: pl.DataFrame):
    p = (pl.scan_parquet(FLOW / "perp_daily.parquet")
         .filter((pl.col("symbol") == "BTCUSDT") & pl.col("date").is_between(DEV0, DEV_END))
         .select("date", "funding_1d").collect())
    g = (fr.with_columns(pl.from_epoch((pl.col("calc_time") // 1000), time_unit="s").dt.date().alias("date"))
         .group_by("date").agg(pl.col("funding_rate").sum().alias("fr_sum")))
    j = p.join(g, on="date", how="left").with_columns(pl.col("fr_sum").fill_null(0.0))
    return float((j["funding_1d"] - j["fr_sum"]).abs().max())


# ----------------------------------------------------------------------------- 主程序
def main():
    assert not OUT.exists(), "输出已存在，拒绝覆盖"
    res = {"spec_id": "MT1P-stable-issuance-v2", "trial_name": "mt1p_stable_issuance_v2"}
    res["registration_check"] = assert_registered()
    sig = pl.read_parquet(SIG).filter(pl.col("date").is_between(DEV0, DEV_END))
    lab = load_labels()
    ex, fr = load_exec_labels()
    res["fr_vs_funding_1d_max_absdiff"] = check_fr(fr)
    df = sig.join(lab, on="date", how="left").join(ex, on="date", how="left")
    df = df.with_columns(pl.col("mt1_state").shift(1).alias("state_lag1"),
                         (pl.col("mt1_g14") > 0).cast(pl.Float64).alias("g14pos"))
    df = df.with_columns((pl.col("mt1_state") * pl.col("g14pos")).alias("st_in"),
                         (pl.col("mt1_state") * (1 - pl.col("g14pos"))).alias("st_out"))
    M = df.filter(pl.col("date").is_between(DEV0, END[14])).drop_nulls(["y14", "mt1_state", *XCOLS])
    assert M.height == 1537 and M["date"].max() == END[14]
    res["holdout_guard"] = {
        "max_label_row_date": str(M["date"].max()), "max_price_date_read": str(lab["date"].max()),
        "kl5m_max_open_time_lt": "2026-04-01T00:00Z", "fr_max_calc_time": int(fr["calc_time"].max()),
        "y14_x_last_d": str(df.filter(pl.col("y14_x").is_not_null())["date"].max()),
        "y14_x_missing_rows": int(df.filter(pl.col("date") <= END["x"])["y14_x"].null_count())}

    # ---------------- P1 主回归 M0
    m0 = ols(M); res["P1_M0"] = m0
    b0 = m0["beta"]
    res["P1_pass"] = bool(b0 > 0 and m0["t"] >= 1.65)

    # ---------------- P2 循环平移
    Z = design(M, ["mt1_state"]); yy = M["y14"].to_numpy(); st = M["mt1_state"].to_numpy().astype(float)
    T = len(st); ts = []
    for k in range(90, T - 90 + 1):
        Z[:, 1] = np.roll(st, k)
        ts.append(float(sm.OLS(yy, Z).fit(cov_type="HAC", cov_kwds={"maxlags": LAG}).tvalues[1]))
    Z[:, 1] = st
    ts = np.array(ts)
    res["P2_circshift"] = {"n_shifts": int(len(ts)), "t_obs": m0["t"], "q95": float(np.quantile(ts, 0.95)),
                           "q50": float(np.median(ts)), "frac_ge_obs": float((ts >= m0["t"]).mean())}
    res["P2_pass"] = bool(m0["t"] > res["P2_circshift"]["q95"])

    # ---------------- P3 非重叠起点
    off = M.select(((pl.col("date") - pl.lit(DEV0)).dt.total_days() % 14).alias("o"))["o"]
    p3 = []
    for o in range(14):
        r = ols(M.filter(off == o), lag=2); r["o"] = o; p3.append(r)
    t3 = [r["t"] for r in p3]; npos = sum(r["beta"] > 0 for r in p3)
    res["P3_offsets"] = {"per_offset": p3, "median_t": float(np.median(t3)), "n_beta_pos": int(npos)}
    res["P3_pass"] = bool(np.median(t3) >= 1.3 and npos >= 12)

    # ---------------- P4 分段（不加 seg FE）
    segdef = {"seg1_2022(重建期)": (dt.date(2022, 1, 1), dt.date(2022, 12, 31)),
              "seg2_2023": (dt.date(2023, 1, 1), dt.date(2023, 12, 31)),
              "seg3_2024": (dt.date(2024, 1, 1), dt.date(2024, 12, 31)),
              "seg4_2025-01..2026-03-17": (dt.date(2025, 1, 1), END[14])}
    p4 = {k: ols(M.filter(pl.col("date").is_between(*v)), seg_fe=False) for k, v in segdef.items()}
    pos = [p4[k]["beta"] > 0 for k in segdef]
    res["P4_segments"] = p4
    res["P4_pass"] = bool(sum(pos) >= 3 and sum(pos[1:]) >= 2)

    # ---------------- P5 滞后 1 天
    p5 = ols(M, regs=("state_lag1",)); res["P5_lag1"] = p5
    res["P5_pass"] = bool(np.sign(p5["beta"]) == np.sign(b0) and p5["beta"] >= 0.5 * b0)

    # ---------------- P6 不加控制
    p6 = ols(M, xcols=[], seg_fe=False); res["P6_nocontrol"] = p6
    res["P6_pass"] = bool(b0 >= 0.5 * p6["beta"])

    # ---------------- P7 平稳块自助
    rng = np.random.default_rng(SEED); p = 1 / 60; B = 2000; bs = np.empty(B)
    Zf = design(M, ["mt1_state"]); n = len(yy)
    for b in range(B):
        idx = np.empty(n, dtype=np.int64); idx[0] = rng.integers(n)
        new = rng.random(n) < p; jumps = rng.integers(n, size=n)
        for i in range(1, n):
            idx[i] = jumps[i] if new[i] else (idx[i - 1] + 1) % n
        bs[b] = beta_only(yy[idx], Zf[idx])
    res["P7_bootstrap"] = {"B": B, "mean_block": 60, "seed": SEED, "q05": float(np.quantile(bs, 0.05)),
                           "q50": float(np.median(bs)), "q95": float(np.quantile(bs, 0.95)),
                           "frac_pos": float((bs > 0).mean())}
    res["P7_pass"] = bool(res["P7_bootstrap"]["q05"] > 0)

    # ---------------- P8 d>=2022-11-08
    p8 = ols(M.filter(pl.col("date") >= dt.date(2022, 11, 8))); res["P8_post_20221108"] = p8
    res["P8_pass"] = bool(p8["beta"] > 0 and p8["t"] >= 1.0)

    # ---------------- P9 安慰剂赛马（d>=2022-04-30）
    M9 = M.filter(pl.col("date") >= dt.date(2022, 4, 30))
    assert M9["state_OI"].null_count() == 0
    m9 = ols(M9); r9 = ols(M9, regs=("mt1_state", "state_P", "state_OI", "state_SV"))
    res["P9_race"] = {"M0_same_sample": m9, "race": r9}
    res["P9_pass"] = bool(r9["beta"] >= 0.5 * m9["beta"] and r9["t"] >= 1.65)

    # ---------------- P10 MA200 分层（X 去掉 ma200，保留 seg FE）
    xc10 = [c for c in XCOLS if c != "ma200"]
    p10 = {f"ma200={v}": ols(M.filter(pl.col("ma200") == v), xcols=xc10) for v in (1, 0)}
    p10_nofe = {f"ma200={v}": ols(M.filter(pl.col("ma200") == v), xcols=xc10, seg_fe=False) for v in (1, 0)}
    res["P10_ma200"] = {"with_seg_fe": p10, "report_no_seg_fe": p10_nofe}
    res["P10_pass"] = bool(all(r["beta"] > 0 for r in p10.values()))

    # ---------------- P11 g14 拆分
    r11 = ols(M, regs=("st_in", "st_out", "g14pos"))
    res["P11_g14split"] = r11
    b_in = r11["all"]["st_in"]["beta"]
    res["P11_pass"] = bool(b_in > 0 and b_in >= 0.5 * b0)

    # ---------------- P12 刀切
    rr = runs(st)
    jk = []
    for v, s0, L in rr:
        if L >= 30:
            d0, d1 = M["date"][s0], M["date"][s0 + L - 1]
            r = ols(M.filter(~pl.col("date").is_between(d0, d1))); r.update(drop=f"run {d0}..{d1} state={int(v)} len={L}")
            jk.append(r)
    r = ols(M.filter(~pl.col("date").is_between(dt.date(2023, 10, 7), dt.date(2024, 4, 17))))
    r.update(drop="2023-10-07..2024-04-17"); jk.append(r)
    for y in range(2022, 2027):
        for qn in range(1, 5):
            if (y, qn) > (2026, 1):
                break
            q0 = dt.date(y, 3 * qn - 2, 1)
            q1 = (dt.date(y + (qn == 4), (3 * qn) % 12 + 1, 1) - dt.timedelta(days=1))
            r = ols(M.filter(~pl.col("date").is_between(q0, q1))); r.update(drop=f"{y}Q{qn}"); jk.append(r)
    res["P12_jackknife"] = {"n_runs_ge30_in_sample": int(sum(1 for x in rr if x[2] >= 30)), "drops": jk,
                            "min_beta": float(min(r["beta"] for r in jk)), "min_t": float(min(r["t"] for r in jk)),
                            "n_fail": int(sum(not (r["beta"] > 0 and r["t"] >= 1.0) for r in jk)),
                            "n_sign_flip": int(sum(r["beta"] <= 0 for r in jk))}
    res["P12_pass"] = bool(res["P12_jackknife"]["n_fail"] == 0)

    # ---------------- P13 01:00 入场
    M13 = M.filter(pl.col("y14_x").is_not_null())
    m13 = ols(M13); x13 = ols(M13, y="y14_x")
    res["P13_exec0100"] = {"M0_same_sample": m13, "y14_x": x13}
    res["P13_pass"] = bool(np.sign(x13["beta"]) == np.sign(m13["beta"]) and x13["beta"] >= 0.5 * m13["beta"])

    # ---------------- 证伪 / 成分一致性
    res["falsify_cry"] = ols(M, regs=("state_cry",))
    res["comp_USDT"] = ols(M, regs=("state_USDT",))
    res["comp_USDC"] = ols(M, regs=("state_USDC",))

    # ---------------- 报告项
    rep = {}
    g0 = json.loads((RUN / "g0_result.json").read_text())
    rep["state_on_X_R2_AUC"] = {"lpm_r2": g0["r2_main"]["lpm_r2"], "logit_auc": g0["r2_main"]["logit_auc_in_sample"]}
    rep["state_3c"] = {**ols(M, regs=("state_3c",)), "agree": float((M["state_3c"] == M["mt1_state"]).mean())}
    rep["state_ext"] = {**ols(M, regs=("state_ext",)), "agree": float((M["state_ext"] == M["mt1_state"]).mean())}
    rep["state_causal"] = {**ols(M, regs=("state_causal",)),
                           "agree": float((M["state_causal"] == M["mt1_state"]).mean())}
    M7 = df.filter(pl.col("date").is_between(DEV0, END[7])).drop_nulls(["y7", "mt1_state", *XCOLS])
    M30 = df.filter(pl.col("date").is_between(DEV0, END[30])).drop_nulls(["y30", "mt1_state", *XCOLS])
    rep["h7"] = ols(M7, y="y7", lag=14); rep["h30"] = ols(M30, y="y30", lag=60)
    rep["h7"]["nw_lag"] = 14; rep["h30"]["nw_lag"] = 60
    t2 = (M.with_columns(pl.when(pl.col("tsmom30") > 0).then(1).otherwise(-1).alias("tsm"))
          .group_by(["tsm", "mt1_state"]).agg(pl.len().alias("n"), pl.col("y14").mean().alias("mean_y14"))
          .sort(["tsm", "mt1_state"]))
    rep["tsmom30_x_state_mean_y14"] = t2.to_dicts()
    # 原 K3：删去对 β 贡献最大的 60 天连续块（FWL 分解：β = Σ s̃_i y_i / Σ s̃_i²）
    Zx = np.delete(Zf, 1, axis=1)
    s_t = st - Zx @ np.linalg.lstsq(Zx, st, rcond=None)[0]
    contrib = s_t * yy / (s_t @ s_t)
    c60 = np.convolve(contrib, np.ones(60), mode="valid")
    i0 = int(np.argmax(c60)); d0, d1 = M["date"][i0], M["date"][i0 + 59]
    rk3 = ols(M.filter(~pl.col("date").is_between(d0, d1)))
    rep["old_K3_drop_max60"] = {"block": f"{d0}..{d1}", "block_contrib": float(c60[i0]), "beta_full": b0, **rk3}
    rr_m = runs(st)
    rep["flicker_1day_main_sample"] = int(sum(1 for k, r in enumerate(rr_m) if r[2] == 1 and 0 < k < len(rr_m) - 1))
    rep["flicker_1day_dev"] = g0["flicker_1day_runs"]
    res["report"] = rep

    # ---------------- 关卡 1 汇总
    P = {f"P{i}": res[f"P{i}_pass"] for i in range(1, 14)}
    res["gate1"] = P
    res["gate1_pass"] = bool(all(P.values()))

    # ---------------- 淘汰条件
    bc, tc = res["falsify_cry"]["beta"], res["falsify_cry"]["t"]
    Kk = {
        "K1": bool(b0 <= 0 or abs(m0["t"]) < 1.0),
        "K2": bool((not res["P6_pass"]) or (r9["beta"] < 0.5 * m9["beta"])
                   or (bc > 0 and (bc >= 0.8 * b0 or tc >= m0["t"]))),
        "K3": bool(res["P12_jackknife"]["n_sign_flip"] > 0 or np.sign(res["P12_jackknife"]["min_beta"]) != np.sign(b0)),
        "K4": bool(np.sign(p5["beta"]) != np.sign(b0)),
        "K6": bool(p8["beta"] <= 0 or p8["t"] < 1.0),
        "K9": bool(np.sign(res["comp_USDT"]["beta"]) != np.sign(res["comp_USDC"]["beta"])),
        "K10": bool(b_in <= 0),
        "K11": bool(any(r["beta"] <= 0 for r in p10.values())),
    }
    res["kill"] = Kk
    res["killed"] = bool(any(Kk.values()))
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    short = {k: v for k, v in res.items() if k not in ("P3_offsets", "P12_jackknife", "registration_check")}
    print(json.dumps(short, ensure_ascii=False, indent=1, default=str))
    print("P12 summary", {k: v for k, v in res["P12_jackknife"].items() if k != "drops"})
    for r in res["P12_jackknife"]["drops"]:
        print("  P12", r["drop"], round(r["beta"], 5), round(r["t"], 2), r["n"])
    print("P3 median t", res["P3_offsets"]["median_t"], "npos", res["P3_offsets"]["n_beta_pos"],
          [round(r["t"], 2) for r in res["P3_offsets"]["per_offset"]])


if __name__ == "__main__":
    main()
