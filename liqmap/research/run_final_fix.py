"""修复月末缺失收益后，用冻结配置（commit 873b37d）重跑：开发期 + 锁定验证集（第 2 次使用，原因：数据 bug 修复，配置不变）。"""
import json
import sys
import time
from datetime import datetime

sys.path.insert(0, "/home/user/sterdusts/liqmap")
import numpy as np
import polars as pl

from backtest import criteria, run_backtest, show, weights_at
from evaluate import HOLDOUT_START, make_factors
from ml import BASE_FEATS, LIQ_FEATS, walk_forward
from research.dsr import dsr, trial_sharpes

src = open("/home/user/sterdusts/liqmap/research/run_gate2_r3c.py").read()
exec(src.split("p = pl.read_parquet")[0].split("from backtest")[0])
exec("TARGET_VOL, LOOKBACK_D, MAX_LEV = 0.20, 90, 3.0\n")
exec("def portable(bt, H):" + src.split("def portable(bt, H):")[1].split("rec = {}")[0])

LOG = "/home/user/sterdusts/liqmap/research/holdout_log.jsonl"
END = datetime(2026, 10, 1)
uses = sum(1 for _ in open(LOG))
assert uses < 3

p = make_factors(pl.read_parquet("/home/user/data/panel_base.parquet"))
feats = BASE_FEATS + LIQ_FEATS
dev = walk_forward(p, feats, 24, end=HOLDOUT_START, label="net_z")
ho = walk_forward(p, feats, 24, start=HOLDOUT_START, end=END, label="net_z")
dev.write_parquet("/home/user/data/pred_base_full_24h_net_z.parquet")
ho.write_parquet("/home/user/data/pred_holdout_full_24h_net_z.parquet")
q = p.join(pl.concat([dev, ho]), on=["t", "symbol"], how="inner")

# 持仓中仍缺失未来收益的情况（真实下架/改名）
d = q.filter(pl.col("t").dt.hour() == 0).sort("symbol", "t").with_columns(
    pl.col("pred").ewm_mean(alpha=0.5, ignore_nulls=True).over("symbol")).sort("t")
miss, tot = [], 0.0
for (t,), g in d.group_by("t", maintain_order=True):
    w = weights_at(g, "pred", 0.2, True, True)
    r = dict(zip(g["symbol"], g["fwd_24h"]))
    for s, wi in w.items():
        tot += abs(wi)
        if r.get(s) is None or not np.isfinite(r.get(s)):
            miss.append((str(t)[:10], s, round(wi, 3)))
print(f"修复后：持仓中未来收益缺失 {len(miss)} 条，占总权重 {sum(abs(x[2]) for x in miss)/tot*100:.3f}%  {miss[:10]}")

rec = {"use": uses + 1, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "config": "full_24h_netz_invvol_portable_breaker",
       "reason": "修复月末缺失未来收益的数据 bug；配置与 873b37d 完全相同"}
srs = trial_sharpes("/home/user/sterdusts/liqmap/research/trials.jsonl")
for fee, fl in ((0.0005, "mixed"), (0.0003, "maker"), (0.0007, "taker")):
    bt = run_backtest(q, "pred", H=24, fee=fee, smooth=0.5, end=END, inv_vol=True)
    full = breaker(portable(bt, 24))
    for seg_name, seg, aseg in (("开发期 2023-04~2026-03", full.filter(pl.col("t") < HOLDOUT_START),
                                 bt.filter(pl.col("t") < HOLDOUT_START)),
                                ("锁定验证集 2026-04~09", full.filter(pl.col("t") >= HOLDOUT_START),
                                 bt.filter(pl.col("t") >= HOLDOUT_START))):
        res = criteria(seg, 24)
        ra = criteria(aseg, 24)
        print(f"\n=== [{fl} {fee*100:.2f}%] {seg_name}")
        show(res)
        print(f"  仅 alpha 腿：夏普 {ra['strategy']['sharpe']:.2f}  累计 {ra['strategy']['cum']*100:+.1f}%  最大回撤 {ra['strategy']['mdd']*100:.1f}%")
        if seg_name.startswith("开发期") and fl == "mixed":
            x = dsr(seg["net"].to_numpy(), 25, srs, 365, 1.5 * res["btc"]["sharpe"])
            y = dsr(seg["net"].to_numpy(), 25, srs, 365)
            print(f"  紧缩夏普：真实夏普 > 选择偏差门槛 的概率 {y['prob']:.3f}；≥ 1.5×BTC 的概率 {x['prob']:.3f}")
        if seg_name.startswith("锁定"):
            rec[fl] = {"sharpe": round(res["strategy"]["sharpe"], 2), "btc_sharpe": round(res["btc"]["sharpe"], 2),
                       "cum": round(res["strategy"]["cum"], 3), "btc_cum": round(res["btc"]["cum"], 3),
                       "alpha_sharpe": round(ra["strategy"]["sharpe"], 2),
                       "criteria": {k: bool(v) for k, v in res["criteria"].items()}, "pass": bool(res["pass"])}
with open(LOG, "a") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
