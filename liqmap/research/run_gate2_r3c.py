"""关卡 2 第三轮（只改组合构建，不重训模型）：
R3a 防轧空：空头腿排除上线不满 90 天的币（2026Q1 亏损主要来自新上 meme 币被轧空）。
R3b BTC + alpha 组合：两条腿按事先固定的 50/50 风险预算，各自用过去 90 天已实现波动率缩放到 20% 年化目标
     （只用 t 之前的数据），总敞口上限 3 倍。
"""
import sys

sys.path.insert(0, "/home/user/sterdusts/liqmap")
import numpy as np
import polars as pl

from backtest import criteria, run_backtest, show
from evaluate import HOLDOUT_START, log_trial, n_trials

H = int(sys.argv[1]) if len(sys.argv) > 1 else 24
MODEL = sys.argv[2] if len(sys.argv) > 2 else "full"
MIN_AGE = 90
TARGET_VOL, LOOKBACK_D, MAX_LEV = 0.20, 90, 3.0

p = pl.read_parquet("/home/user/data/panel_base.parquet")
first = pl.read_parquet("/home/user/data/kl1d.parquet").group_by("symbol").agg(
    pl.from_epoch(pl.col("open_time").min(), time_unit="ms").dt.cast_time_unit("us").alias("listed"))
p = p.join(first, on="symbol", how="left").with_columns(
    ((pl.col("t") - pl.col("listed")).dt.total_days()).alias("age_days"))
pr = pl.read_parquet(f"/home/user/data/pred_base_{MODEL}_{H}h_net_z.parquet")
q = p.join(pr, on=["t", "symbol"], how="inner")

# R3a：新币只允许做多 —— 把它们的得分下限截到中位数，使其不会进入空头腿
q_guard = q.with_columns(
    pl.when(pl.col("age_days") < MIN_AGE)
    .then(pl.max_horizontal(pl.col("pred"), pl.col("pred").median().over("t")))
    .otherwise(pl.col("pred")).alias("pred_guard"))

def portable(bt, H):
    """BTC 腿 + alpha 腿，各自按过去 LOOKBACK_D 天波动率缩放到 TARGET_VOL（只用历史）。"""
    ppy = 8760 / H
    n = int(LOOKBACK_D * 24 / H)
    a = bt["net"].to_numpy()
    b = bt["btc"].to_numpy()
    out = np.full(len(a), np.nan)
    for i in range(n, len(a)):
        va = a[i - n:i].std(ddof=1) * np.sqrt(ppy)
        vb = b[i - n:i].std(ddof=1) * np.sqrt(ppy)
        wa = min(TARGET_VOL / max(va, 1e-4), MAX_LEV)
        wb = min(TARGET_VOL / max(vb, 1e-4), MAX_LEV)
        out[i] = wa * a[i] + wb * b[i]
    ok = ~np.isnan(out)
    return bt.filter(pl.Series(ok)).with_columns(pl.Series("net", out[ok]))



DD_CUT, DD_RESUME = 0.10, 0.05


def breaker(pb):
    r = pb["net"].to_numpy()
    out = np.empty(len(r)); eq = 1.0; peak = 1.0; scale = 1.0
    for i, x in enumerate(r):
        out[i] = scale * x                    # 本期用上期末决定的仓位
        eq *= 1 + out[i]; peak = max(peak, eq)
        dd = 1 - eq / peak
        if scale == 1.0 and dd > DD_CUT:
            scale = 0.5
        elif scale < 1.0 and dd < DD_RESUME:
            scale = 1.0
    return pb.with_columns(pl.Series("net", out))


rec = {}
for fee, fl in ((0.0007, "taker"), (0.0005, "mixed"), (0.0003, "maker")):
    bt = run_backtest(q_guard.with_columns(pl.col("pred").alias("pred_use")), "pred_use", H=H, fee=fee, smooth=0.5,
                      end=HOLDOUT_START, inv_vol=True)
    pb = portable(bt, H)
    for lab, x in (("R3b", pb), ("R3c+熔断", breaker(pb))):
        res = criteria(x, H)
        print(f"\n=== {lab} [{fl} {fee*100:.2f}%]")
        show(res)
        rec[f"{lab}_{fl}"] = {"sharpe": round(res["strategy"]["sharpe"], 2), "btc": round(res["btc"]["sharpe"], 2),
                             "mdd_vm": round(res["vol_matched"]["mdd"], 3), "btc_mdd": round(res["btc"]["mdd"], 3),
                             "pass": bool(res["pass"])}
log_trial(f"r3c_breaker_{MODEL}_{H}h", "关卡2 R3c：R3b + 预定义回撤熔断(10%减半/5%恢复)；含混合费率敏感性", rec)
print("\n累计登记试验数:", n_trials())
