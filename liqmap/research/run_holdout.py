"""关卡 3：锁定验证集（2026-04-01 ~ 2026-09-30）。最多使用 3 次，每次登记到 holdout_log.jsonl。

冻结配置（第 1 次使用前写定，不得根据结果修改）：
- 模型：LightGBM，full 特征（公开因子 + 代理清算地图），24h，标签 net_z，训练窗 18 个月，每月滚动重训
- 组合：多空各 20%，腿内逆波动加权，BTC 贝塔对冲，信号 EMA 平滑 0.5
- BTC + alpha：50/50 风险预算，各自 90 天已实现波动率缩放到 20%，单腿杠杆上限 3
- 熔断：组合回撤 > 10% 减半，< 5% 恢复
- 成本：主口径 0.05%（maker/taker 混合），另报 0.03% / 0.07%
"""
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, "/home/user/sterdusts/liqmap")
import polars as pl

from backtest import criteria, run_backtest, show
from evaluate import HOLDOUT_START, make_factors
from ml import BASE_FEATS, LIQ_FEATS, walk_forward

LOG = "/home/user/sterdusts/liqmap/research/holdout_log.jsonl"
HOLDOUT_END = datetime(2026, 10, 1)
uses = sum(1 for _ in open(LOG)) if os.path.exists(LOG) else 0
assert uses < 3, "锁定验证集 3 次机会已用完"

exec(open("/home/user/sterdusts/liqmap/research/run_gate2_r3c.py").read().split("results = {}")[0]
     .split("p = pl.read_parquet")[0])  # 取常量定义
src = open("/home/user/sterdusts/liqmap/research/run_gate2_r3c.py").read()
exec("def portable(bt, H):" + src.split("def portable(bt, H):")[1].split("rec = {}")[0])

p = make_factors(pl.read_parquet("/home/user/data/panel_base.parquet"))
dev = pl.read_parquet("/home/user/data/pred_base_full_24h_net_z.parquet")
ho = walk_forward(p, BASE_FEATS + LIQ_FEATS, 24, start=HOLDOUT_START, end=HOLDOUT_END, label="net_z")
ho.write_parquet("/home/user/data/pred_holdout_full_24h_net_z.parquet")
q = p.join(pl.concat([dev, ho]), on=["t", "symbol"], how="inner")

rec = {"use": uses + 1, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "config": "full_24h_netz_invvol_portable_breaker"}
for fee, fl in ((0.0005, "mixed"), (0.0003, "maker"), (0.0007, "taker")):
    bt = run_backtest(q, "pred", H=24, fee=fee, smooth=0.5, end=HOLDOUT_END, inv_vol=True)
    full = breaker(portable(bt, 24))          # 连续运行，保证波动率估计和熔断状态延续
    seg = full.filter(pl.col("t") >= HOLDOUT_START)
    alpha_seg = bt.filter(pl.col("t") >= HOLDOUT_START)
    res = criteria(seg, 24)
    ra = criteria(alpha_seg, 24)
    print(f"\n=== 锁定验证集 2026-04 ~ 2026-09 [{fl} {fee*100:.2f}%]  （第 {uses+1} 次使用）")
    show(res)
    print(f"  仅 alpha 腿：夏普 {ra['strategy']['sharpe']:.2f}  累计 {ra['strategy']['cum']*100:+.1f}%  最大回撤 {ra['strategy']['mdd']*100:.1f}%")
    rec[fl] = {"sharpe": round(res["strategy"]["sharpe"], 2), "btc_sharpe": round(res["btc"]["sharpe"], 2),
               "cum": round(res["strategy"]["cum"], 3), "btc_cum": round(res["btc"]["cum"], 3),
               "mdd": round(res["strategy"]["mdd"], 3), "alpha_sharpe": round(ra["strategy"]["sharpe"], 2),
               "criteria": {k: bool(v) for k, v in res["criteria"].items()}, "pass": bool(res["pass"])}
with open(LOG, "a") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
