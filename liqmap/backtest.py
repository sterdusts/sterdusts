"""扣费组合回测 + 按成功标准对比 BTC 持有。

组合：每 H 小时调仓；按得分做多前 q、做空后 q（各占一半总敞口），剩余贝塔用 BTC 对冲。
成本：换手 × 单边费率（默认 taker 0.05% + 滑点 0.02%）；资金费按实际结算计入（多头支付为正费率）。
"""
import numpy as np
import polars as pl

from evaluate import HOLDOUT_START


def weights_at(g, score, q, beta_hedge=True, inv_vol=False):
    g = g.filter(pl.col(score).is_not_null() & pl.col(score).is_finite())
    n = g.height
    if n < 15:
        return {}
    k = max(int(round(n * q)), 1)
    g = g.sort(score)
    short = g.head(k)["symbol"].to_list()
    long = g.tail(k)["symbol"].to_list()
    beta = dict(zip(g["symbol"], g["beta"]))
    if inv_vol:  # 腿内按波动率倒数分配，避免高波动币主导风险
        vol = dict(zip(g["symbol"], g["vol_24h"].fill_null(g["vol_24h"].median())))
        iv = lambda ss: {s: 1 / max(vol.get(s) or 0.05, 0.01) for s in ss}  # noqa: E731
        lw, sw = iv(long), iv(short)
        w = {s: 0.5 * x / sum(lw.values()) for s, x in lw.items()}
        for s, x in sw.items():
            w[s] = w.get(s, 0) - 0.5 * x / sum(sw.values())
    else:
        w = {s: 0.5 / k for s in long}
        for s in short:
            w[s] = w.get(s, 0) - 0.5 / k
    if beta_hedge:
        b = sum(wi * beta.get(s, 1.0) for s, wi in w.items())
        w["BTCUSDT"] = w.get("BTCUSDT", 0) - b
    return w


def run_backtest(p, score, H=8, q=0.2, fee=0.0007, smooth=0.0, beta_hedge=True, start=None, end=None,
                 inv_vol=False):
    """返回每期净收益序列（DataFrame: t, gross, cost, funding, net, btc）。"""
    df = p.filter(pl.col("t").dt.hour() % H == 0)
    if start is not None:
        df = df.filter(pl.col("t") >= start)
    if end is not None:
        df = df.filter(pl.col("t") < end)
    df = df.sort("t")
    fwd, fund = f"fwd_{H}h", f"fund_{H}h"
    if smooth > 0:  # 对得分做 EMA 平滑，降低换手
        df = df.sort("symbol", "t").with_columns(
            pl.col(score).ewm_mean(alpha=1 - smooth, ignore_nulls=True).over("symbol").alias(score)).sort("t")
    btc_by_t = dict(df.filter(pl.col("symbol") == "BTCUSDT").select("t", fwd).iter_rows())
    prev = {}
    rows = []
    for (t,), g in df.group_by("t", maintain_order=True):
        w = weights_at(g, score, q, beta_hedge, inv_vol)
        if not w or btc_by_t.get(t) is None:
            continue
        r = dict(zip(g["symbol"], g[fwd]))
        f = dict(zip(g["symbol"], g[fund]))
        gross = 0.0
        funding = 0.0
        for s, wi in w.items():
            ri = r.get(s)
            if ri is None or not np.isfinite(ri):
                ri = 0.0  # 缺失（如下架）按 0 收益处理，偏保守地不额外加分
            gross += wi * (np.exp(ri) - 1)
            funding -= wi * (f.get(s) or 0.0)
        turnover = sum(abs(w.get(s, 0) - prev.get(s, 0)) for s in set(w) | set(prev))
        cost = turnover * fee
        rows.append((t, gross, cost, funding, gross - cost + funding, np.exp(btc_by_t[t]) - 1, turnover))
        prev = w
    return pl.DataFrame(rows, schema=["t", "gross", "cost", "funding", "net", "btc", "turnover"], orient="row")


def perf(r, H):
    """r: 每期简单收益 numpy 数组。"""
    ppy = 8760 / H
    r = np.asarray(r, dtype=float)
    if len(r) < 2:
        return {}
    eq = np.cumprod(1 + r)
    dd = 1 - eq / np.maximum.accumulate(eq)
    vol = r.std(ddof=1) * np.sqrt(ppy)
    return {"cum": eq[-1] - 1, "ann_ret": eq[-1] ** (ppy / len(r)) - 1, "vol": vol,
            "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(ppy) if r.std() > 0 else np.nan,
            "mdd": dd.max()}


def criteria(bt, H):
    """按交接文档第十节的成功标准逐条判断。"""
    s, b = bt["net"].to_numpy(), bt["btc"].to_numpy()
    ps, pb = perf(s, H), perf(b, H)
    k = pb["vol"] / ps["vol"] if ps["vol"] > 0 else np.nan
    pv = perf(s * k, H)  # 波动率对齐到 BTC
    res = {"strategy": ps, "btc": pb, "vol_matched": pv, "scale": k}
    c = {}
    if pb["cum"] > 0:
        c["1_sharpe>=1.5xBTC"] = ps["sharpe"] >= 1.5 * pb["sharpe"]
        c["2_volmatched_cum>=1.5xBTC"] = pv["cum"] >= 1.5 * pb["cum"]
    else:
        c["5_btc_down:abs_ret>0&sharpe>=1"] = (ps["cum"] > 0) and (ps["sharpe"] >= 1.0)
    c["3_volmatched_mdd<=BTC"] = pv["mdd"] <= pb["mdd"]
    # 4) 稳定性：自然年 & 滚动 6 个月窗口的风险调整后胜率
    yrs = bt.with_columns(pl.col("t").dt.year().alias("y")).group_by("y").agg("net", "btc").sort("y")
    wins, tot = 0, 0
    yr_detail = {}
    for y, n_, b_ in yrs.iter_rows():
        if len(n_) < 30:
            continue
        a, bb = perf(np.array(n_), H), perf(np.array(b_), H)
        win = a["sharpe"] > bb["sharpe"]
        yr_detail[int(y)] = (round(a["sharpe"], 2), round(bb["sharpe"], 2))
        wins += win
        tot += 1
    per = int(182 * 24 / H)
    step = int(30 * 24 / H)
    w6, t6 = 0, 0
    for i in range(0, len(s) - per + 1, step):
        a, bb = perf(s[i:i + per], H), perf(b[i:i + per], H)
        w6 += a["sharpe"] > bb["sharpe"]
        t6 += 1
    c["4_stability(years&6m>50%)"] = (tot > 0 and wins / tot > 0.5) and (t6 > 0 and w6 / t6 > 0.5)
    res["years_sharpe(strat,btc)"] = yr_detail
    res["win_6m"] = f"{w6}/{t6}"
    res["criteria"] = c
    res["pass"] = all(c.values())
    return res


def show(res):
    f = lambda d: f"累计 {d['cum']*100:+.1f}%  年化 {d['ann_ret']*100:+.1f}%  波动 {d['vol']*100:.1f}%  夏普 {d['sharpe']:.2f}  最大回撤 {d['mdd']*100:.1f}%"  # noqa: E731
    print("  策略     :", f(res["strategy"]))
    print("  BTC 持有 :", f(res["btc"]))
    print(f"  波动对齐 (x{res['scale']:.2f}):", f(res["vol_matched"]))
    print("  分年夏普(策略,BTC):", res["years_sharpe(strat,btc)"], " 6个月窗口胜率:", res["win_6m"])
    print("  标准:", {k: bool(v) for k, v in res["criteria"].items()}, "→ 通过" if res["pass"] else "→ 未通过")
