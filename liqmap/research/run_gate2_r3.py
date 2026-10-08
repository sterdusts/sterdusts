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

results = {}
for lab, score, data in (("R3a_防轧空", "pred_guard", q_guard),):
    for fee, fl in ((0.0007, "taker"), (0.0003, "maker")):
        bt = run_backtest(data, score, H=H, fee=fee, smooth=0.5, end=HOLDOUT_START, inv_vol=True)
        res = criteria(bt, H)
        print(f"\n=== {lab} [{fl}] 毛收益 {bt['gross'].sum()*100:+.1f}% 成本 {bt['cost'].sum()*100:.1f}% 资金费 {bt['funding'].sum()*100:+.1f}%")
        show(res)
        results[f"{lab}_{fl}"] = (bt, res)
    log_trial(f"r3a_guard_{MODEL}_{H}h", f"关卡2 R3a：{MODEL} {H}h net_z 逆波动，空头排除上线<{MIN_AGE}天",
              {k: {"sharpe": round(v[1]["strategy"]["sharpe"], 2), "pass": bool(v[1]["pass"])}
               for k, v in results.items()})


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


for fl in ("taker", "maker"):
    bt, _ = results[f"R3a_防轧空_{fl}"]
    pb = portable(bt, H)
    res = criteria(pb, H)
    corr = np.corrcoef(bt["net"].to_numpy(), bt["btc"].to_numpy())[0, 1]
    print(f"\n=== R3b BTC+alpha 组合 [{fl}]（alpha 与 BTC 相关系数 {corr:+.2f}）")
    show(res)
    log_trial(f"r3b_portable_{MODEL}_{H}h_{fl}",
              f"关卡2 R3b：BTC 腿 + R3a alpha 腿，50/50 风险预算，{LOOKBACK_D}天波动率缩放至 {TARGET_VOL:.0%}",
              {"sharpe": round(res["strategy"]["sharpe"], 2), "btc_sharpe": round(res["btc"]["sharpe"], 2),
               "corr": round(corr, 2), "criteria": {k: bool(v) for k, v in res["criteria"].items()},
               "pass": bool(res["pass"])})
print("\n累计登记试验数:", n_trials())
