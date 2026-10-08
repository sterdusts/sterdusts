"""把每个币的 5 分钟数据 + 清算地图输出整理成小时级面板：特征、基准因子、前瞻收益。

时间约定：
- 行时间 t = 整点 H。特征只用 H 之前收盘的 K 线和时间戳 ≤ H 的 OI。
- 前瞻收益从 H+5 分钟的收盘价开始算（多留一根 K 线的执行延迟），到 H+5 分钟+h 结束。
"""
import os
import sys
import multiprocessing as mp

import numpy as np
import polars as pl

from liqsim import run_symbol, DATA, CALIB_LEV, CALIB_W

HORIZONS = [1, 4, 8, 24]  # 小时


def symbol_panel(sym, churn=0.05, lev_w=None, lev=None):
    kw = {"churn": churn}
    if lev_w is not None:
        kw["lev_w"] = lev_w
    if lev is not None:
        kw["lev"] = lev
    df = run_symbol(sym, **kw)
    if df is None:
        return None
    # 规则化到完整 5 分钟网格，保证 shift 等于真实时间位移
    df = df.sort("t_end").upsample("t_end", every="5m").with_columns(pl.col("symbol").fill_null(sym))
    lc = pl.col("close").log()
    df = df.with_columns(
        (lc - lc.shift(12)).alias("ret_1h"),
        (lc - lc.shift(48)).alias("ret_4h"),
        (lc - lc.shift(288)).alias("ret_24h"),
        (lc - lc.shift(2016)).alias("ret_7d"),
        (lc.diff().rolling_std(288, min_samples=200) * np.sqrt(288)).alias("vol_24h"),
        pl.col("quote_volume").rolling_sum(288, min_samples=200).alias("qv_24h"),
        (pl.col("taker_buy_volume").rolling_sum(48, min_samples=30)
         / pl.col("volume").rolling_sum(48, min_samples=30)).alias("buy_4h"),
        (pl.col("oi").log() - pl.col("oi").log().shift(48)).alias("doi_4h"),
        (pl.col("oi").log() - pl.col("oi").log().shift(288)).alias("doi_24h"),
        pl.col("liqL").rolling_sum(12, min_samples=6).alias("liqL_1h"),
        pl.col("liqS").rolling_sum(12, min_samples=6).alias("liqS_1h"),
        pl.col("liqL").rolling_sum(48, min_samples=24).alias("liqL_4h"),
        pl.col("liqS").rolling_sum(48, min_samples=24).alias("liqS_4h"),
        pl.col("liqL").rolling_sum(288, min_samples=144).alias("liqL_24h"),
        pl.col("liqS").rolling_sum(288, min_samples=144).alias("liqS_24h"),
        (pl.col("liqL") + pl.col("liqS")).rolling_mean(288 * 30, min_samples=288 * 5).alias("liq_avg30"),
        (pl.col("top_ls_pos").log() - pl.col("top_ls_pos").log().shift(288)).alias("dtopls_24h"),
        (pl.col("ls_acct").log()).alias("ls_acct_log"),
        # 前瞻收益：入场 = 下一根 K 线收盘 (H+5m)
        *[(lc.shift(-1 - 12 * h) - lc.shift(-1)).alias(f"fwd_{h}h") for h in HORIZONS],
    )
    df = df.filter(pl.col("t_end").dt.minute() == 0).rename({"t_end": "t"})
    keep = ["t", "symbol", "close", "oi_value", "ret_1h", "ret_4h", "ret_24h", "ret_7d", "vol_24h", "qv_24h",
            "buy_4h", "doi_4h", "doi_24h", "dtopls_24h", "ls_acct_log",
            "liqL_1h", "liqS_1h", "liqL_4h", "liqS_4h", "liqL_24h", "liqS_24h", "liq_avg30",
            "gup", "gdn"] + [c for c in df.columns if c[:2] in ("up", "dn") and c[2:].isdigit()] \
        + [f"fwd_{h}h" for h in HORIZONS]
    return df.select(keep)


def _job(args):
    sym, churn, lev_w, lev = args
    try:
        return symbol_panel(sym, churn, lev_w, lev)
    except Exception as e:  # 个别币数据异常不影响整体
        print("fail", sym, e, file=sys.stderr)
        return None


def build_panel(churn=0.05, lev_w=None, lev=None, tag="base", procs=4):
    u = pl.read_parquet(f"{DATA}/universe.parquet").with_columns(pl.col("d").alias("month"))
    syms = sorted(u["symbol"].unique().to_list())
    syms = [s for s in syms if os.path.exists(f"{DATA}/metrics/{s}.parquet")
            and os.path.exists(f"{DATA}/kl5m/{s}.parquet")]
    with mp.get_context("spawn").Pool(procs) as p:  # fork 会让 polars 线程池死锁
        parts = [x for x in p.map(_job, [(s, churn, lev_w, lev) for s in syms]) if x is not None]
    panel = pl.concat(parts)
    # 只保留当月在池内的观测
    panel = panel.with_columns(pl.col("t").dt.truncate("1mo").dt.date().alias("month")).join(
        u.select("month", "symbol"), on=["month", "symbol"], how="inner")
    panel = add_market(panel)
    panel = add_funding(panel)
    panel.write_parquet(f"{DATA}/panel_{tag}.parquet")
    return panel


def add_market(panel):
    """BTC 收益、滚动贝塔（168 小时）、残差前瞻收益。"""
    btc = panel.filter(pl.col("symbol") == "BTCUSDT").select(
        "t", pl.col("ret_1h").alias("btc_1h"), *[pl.col(f"fwd_{h}h").alias(f"btc_fwd_{h}h") for h in HORIZONS])
    p = panel.join(btc, on="t", how="left").sort("symbol", "t")
    cov = (pl.col("ret_1h") * pl.col("btc_1h")).rolling_mean(168, min_samples=72) \
        - pl.col("ret_1h").rolling_mean(168, min_samples=72) * pl.col("btc_1h").rolling_mean(168, min_samples=72)
    var = pl.col("btc_1h").rolling_var(168, min_samples=72, ddof=0)
    p = p.with_columns((cov / var).over("symbol").clip(-1, 4).alias("beta"))
    p = p.with_columns(pl.col("beta").fill_null(1.0))
    return p.with_columns(*[(pl.col(f"fwd_{h}h") - pl.col("beta") * pl.col(f"btc_fwd_{h}h")).alias(f"res_{h}h")
                            for h in HORIZONS])


def add_funding(panel):
    """最近一次已结算资金费率（as-of），以及持仓期间应付的资金费（回测用）。"""
    import glob
    f = pl.concat([pl.read_parquet(x) for x in glob.glob(f"{DATA}/funding_*.parquet")]).with_columns(
        pl.from_epoch("calc_time", time_unit="ms").dt.cast_time_unit("us").alias("ft")).sort("symbol", "ft")
    f = f.unique(["symbol", "ft"], keep="last").sort("symbol", "ft")
    p = panel.sort("t").with_columns(pl.col("t").dt.cast_time_unit("us"))
    p = p.join_asof(f.select("symbol", "ft", "funding_rate"), left_on="t", right_on="ft", by="symbol",
                    strategy="backward").drop("ft")
    # 未来 h 小时内累计资金费（多头支付为正）
    fc = f.with_columns(pl.col("funding_rate").cum_sum().over("symbol").alias("fcum"))
    for h in HORIZONS:
        a = p.select("symbol", "t").join_asof(fc.select("symbol", "ft", "fcum"), left_on="t", right_on="ft",
                                              by="symbol", strategy="backward")["fcum"]
        p2 = p.select("symbol", (pl.col("t") + pl.duration(hours=h)).alias("t2")).join_asof(
            fc.select("symbol", "ft", "fcum"), left_on="t2", right_on="ft", by="symbol", strategy="backward")["fcum"]
        p = p.with_columns((p2 - a).fill_null(0.0).alias(f"fund_{h}h"))
    return p


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    churn = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    if tag.startswith("calib"):
        p = build_panel(churn=churn, lev_w=CALIB_W, lev=CALIB_LEV, tag=tag)
    else:
        p = build_panel(churn=churn, tag=tag)
    print(p.shape, p["t"].min(), p["t"].max(), p["symbol"].n_unique())
