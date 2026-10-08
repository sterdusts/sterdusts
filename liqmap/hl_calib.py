"""用 Hyperliquid 大户真实持仓校准代理清算地图的形状。

对比内容：
1. 真实仓位的清算距离分布（名义价值加权），以及杠杆设置分布。
2. 在 10% 范围内，清算质量落在 1/2/3/5% 以内的比例：真实 vs 代理模型（BTC 最新时刻 / 全样本中位数）。
"""
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "/home/user/sterdusts/liqmap")

import os
snap = pl.read_parquet(os.environ.get("SNAP", "/home/user/data/hl_snapshot.parquet")).filter(pl.col("mid").is_not_null())
snap = snap.with_columns(
    pl.when(pl.col("szi") > 0).then(pl.lit("long")).otherwise(pl.lit("short")).alias("side"),
    ((pl.col("liq_px") - pl.col("mid")).abs() / pl.col("mid")).alias("dist"),
)
print(f"仓位数 {snap.height}，名义价值合计 ${snap['ntl'].sum()/1e9:.2f}B，币种数 {snap['coin'].n_unique()}")
print("有清算价的名义占比:", round(snap.filter(pl.col("liq_px").is_not_null())["ntl"].sum() / snap["ntl"].sum(), 3))

# 1) 清算距离分布（名义加权）
bins = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]
s = snap.filter(pl.col("dist").is_not_null())
tot = snap["ntl"].sum()
print("\n清算距离分布（占全部名义价值，含无清算价的仓位在分母里）:")
for a, b in zip(bins[:-1], bins[1:]):
    w = s.filter((pl.col("dist") >= a) & (pl.col("dist") < b))["ntl"].sum() / tot
    print(f"  {a*100:>5.0f}% – {b*100:>5.0f}%: {w*100:5.1f}%")

print("\n杠杆设置分布（名义加权）:")
lv = snap.group_by(pl.col("lev").clip(1, 100).cut([2, 3, 5, 10, 20, 25, 40, 50]).alias("lev_bucket")).agg(
    (pl.col("ntl").sum() / tot * 100).round(1).alias("pct")).sort("lev_bucket")
print(lv.to_pandas().to_string(index=False))
print("\n逐仓/全仓:", snap.group_by("lev_type").agg((pl.col("ntl").sum() / tot * 100).round(1)).rows())


# 2) 10% 范围内的形状对比
def shape_real(df):
    w10 = df.filter(pl.col("dist") < 0.10)["ntl"].sum()
    return {d: df.filter(pl.col("dist") < d / 100)["ntl"].sum() / w10 for d in (1, 2, 3, 5)} if w10 > 0 else None


print("\n10% 以内清算质量中，落在 d% 以内的比例（真实 Hyperliquid）:")
for c in ["BTC", "ETH", "SOL", None]:
    sub = s if c is None else s.filter(pl.col("coin") == c)
    r = shape_real(sub)
    if r:
        print(f"  {c or '全部':>4}: " + "  ".join(f"{d}%:{v*100:5.1f}%" for d, v in r.items()))

if len(sys.argv) > 1:
    panel = pl.read_parquet(sys.argv[1])
    print("\n代理模型（同一口径，全样本中位数；以及最新时刻）:")
    for c in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
        g = panel.filter(pl.col("symbol") == c).drop_nulls(["up10", "dn10"])
        if g.height == 0:
            continue
        tot10 = pl.col("up10") + pl.col("dn10")
        med = g.select(*[((pl.col(f"up{d}") + pl.col(f"dn{d}")) / tot10).median().alias(str(d)) for d in (1, 2, 3, 5)]).row(0)
        last = g.tail(1).select(*[((pl.col(f"up{d}") + pl.col(f"dn{d}")) / tot10).alias(str(d)) for d in (1, 2, 3, 5)]).row(0)
        print(f"  {c:>8} 中位: " + "  ".join(f"{d}%:{v*100:5.1f}%" for d, v in zip((1, 2, 3, 5), med)))
        print(f"  {c:>8} 最新: " + "  ".join(f"{d}%:{v*100:5.1f}%" for d, v in zip((1, 2, 3, 5), last)))
