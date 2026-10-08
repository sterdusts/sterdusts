"""关卡 2：LightGBM 组合模型（base vs full），样本外预测 → 扣费回测 → 按成功标准对比 BTC 持有。

开发期样本外：2023-01 ~ 2026-03（锁定验证集 2026-04 之后不碰）。
"""
import sys

sys.path.insert(0, "/home/user/sterdusts/liqmap")
import polars as pl

from backtest import criteria, run_backtest, show
from evaluate import HOLDOUT_START, ic_series, log_trial, make_factors, n_trials
from ml import BASE_FEATS, LIQ_FEATS, walk_forward

tag = sys.argv[1] if len(sys.argv) > 1 else "base"
H = int(sys.argv[2]) if len(sys.argv) > 2 else 8
LABEL = sys.argv[3] if len(sys.argv) > 3 else "rank"
INV_VOL = len(sys.argv) > 4 and sys.argv[4] == "invvol"
p = make_factors(pl.read_parquet(f"/home/user/data/panel_{tag}.parquet"))

for name, feats in (("base", BASE_FEATS), ("full", BASE_FEATS + LIQ_FEATS)):
    pr = walk_forward(p, feats, H, label=LABEL)
    pr.write_parquet(f"/home/user/data/pred_{tag}_{name}_{H}h_{LABEL}.parquet")
    q = p.join(pr, on=["t", "symbol"], how="inner")
    ic = ic_series(q, "pred", f"res_{H}h")
    nov = ic.filter(pl.col("t").dt.hour() % H == 0)
    t_ = nov["ic"].mean() / nov["ic"].std() * nov.height ** 0.5
    yr = ic.group_by(pl.col("t").dt.year().alias("y")).agg(pl.col("ic").mean().round(4)).sort("y").rows()
    print(f"\n=== 模型 {name}（{len(feats)} 个特征，{H}h）样本外 RankIC={ic['ic'].mean():+.4f} t={t_:.2f} 分年={yr}")
    out = {"ic": round(ic["ic"].mean(), 4), "t": round(t_, 2), "by_year": dict(yr)}
    scen = ((0.0007, 0.5, "taker+滑点 0.07%，信号平滑"), (0.0003, 0.5, "maker 0.03%，信号平滑"))
    if LABEL == "rank" and not INV_VOL:
        scen = ((0.0007, 0.0, "taker+滑点 0.07%"),) + scen
    for fee, smooth, lab in scen:
        bt = run_backtest(q, "pred", H=H, fee=fee, smooth=smooth, end=HOLDOUT_START, inv_vol=INV_VOL)
        res = criteria(bt, H)
        print(f"--- 回测 [{lab}]  平均换手 {bt['turnover'].mean():.2f}/期，"
              f"毛收益 {bt['gross'].sum()*100:+.1f}%，成本 {bt['cost'].sum()*100:.1f}%，资金费 {bt['funding'].sum()*100:+.1f}%")
        show(res)
        out[lab] = {"sharpe": round(res["strategy"]["sharpe"], 2), "btc_sharpe": round(res["btc"]["sharpe"], 2),
                    "mdd": round(res["strategy"]["mdd"], 3), "pass": bool(res["pass"])}
    log_trial(f"ml_{name}_{H}h_{LABEL}{'_invvol' if INV_VOL else ''}",
              f"[{tag}] 关卡2 LightGBM {name} 特征，{H}h，标签={LABEL}，逆波动加权={INV_VOL}，滚动前推 18 个月", out)
print("\n累计登记试验数:", n_trials())
