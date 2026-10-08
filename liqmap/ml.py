"""关卡 2：滚动前推 LightGBM 组合模型。

对比两组特征，衡量清算类特征的增量价值：
- base：公开基准因子（动量/反转/波动/成交额/资金费/OI 变化/主动买入/贝塔/多空比）
- full：base + 代理清算地图特征
标签：未来 H 小时去 BTC 贝塔残差收益的截面排名。每月重训，训练窗 18 个月，训练与预测之间留 2 天间隔。
"""
import sys
from datetime import datetime

import lightgbm as lgb
import numpy as np
import polars as pl
from dateutil.relativedelta import relativedelta

from evaluate import HOLDOUT_START, make_factors

BASE_FEATS = ["ret_1h", "ret_4h", "ret_24h", "ret_7d", "vol_24h", "lqv", "funding_rate", "doi_4h", "doi_24h",
              "buy_4h", "beta", "dtopls_24h", "ls_acct_log"]
LIQ_FEATS = ([f"up{d}" for d in (1, 2, 3, 5, 10)] + [f"dn{d}" for d in (1, 2, 3, 5, 10)]
             + ["gup", "gdn", "h1_imb2", "h1_imb5", "h1_imb10", "h1_gimb", "h1_diff5",
                "h2_net1h", "h2_net4h", "h2_net24h", "h2_net4h_rel", "h2_casc4h", "fragility5",
                "liqL_4h", "liqS_4h", "liqL_24h", "liqS_24h"])
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_data_in_leaf=1000,
              feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0, verbose=-1,
              num_threads=4, seed=7)
ROUNDS = 400


def prep(p, feats, H, label="rank"):
    """label="rank"：残差收益的截面排名；
    label="net_z"：与盈亏对齐 —— (残差收益 - 资金费) 截面去极值(2.5%/97.5%)后标准化，保留右尾信息。"""
    tgt = f"res_{H}h"
    if label == "net_z":
        x = pl.col(f"res_{H}h") - pl.col(f"fund_{H}h")
        lo, hi = x.quantile(0.025).over("t"), x.quantile(0.975).over("t")
        xc = x.clip(lo, hi)
        p = p.with_columns(((xc - xc.mean().over("t")) / xc.std().over("t")).alias("net_z"))
        tgt = "net_z"
    p = p.filter(pl.col(tgt).is_not_null() | (pl.col("t") >= HOLDOUT_START))
    # 特征与标签都做截面排名（-0.5~0.5），缺失填 0（中位）
    cols = feats + ([tgt] if label == "rank" else [])
    p = p.with_columns([((pl.col(c).rank("average").over("t") - 0.5) / pl.col(c).count().over("t") - 0.5)
                        .fill_null(0.0).fill_nan(0.0).alias(f"r_{c}") for c in cols])
    ycol = f"r_{tgt}" if label == "rank" else tgt
    return p.with_columns(pl.when(pl.col(tgt).is_null()).then(None).otherwise(pl.col(ycol)).alias("y"))


def walk_forward(p, feats, H, start=datetime(2023, 1, 1), end=HOLDOUT_START, train_months=18, seed=7,
                 rounds=ROUNDS, label="rank"):
    d = prep(p, feats, H, label)
    X_cols = [f"r_{c}" for c in feats]
    preds = []
    m = start
    while m < end:
        nxt = m + relativedelta(months=1)
        tr = d.filter((pl.col("t") >= m - relativedelta(months=train_months)) &
                      (pl.col("t") < m - relativedelta(days=2)) & pl.col("y").is_not_null())
        te = d.filter((pl.col("t") >= m) & (pl.col("t") < nxt))
        if tr.height < 10000 or te.height == 0:
            m = nxt
            continue
        params = dict(PARAMS, seed=seed)
        model = lgb.train(params, lgb.Dataset(tr.select(X_cols).to_numpy(), tr["y"].to_numpy()),
                          num_boost_round=rounds)
        preds.append(te.select("t", "symbol").with_columns(
            pl.Series("pred", model.predict(te.select(X_cols).to_numpy()))))
        m = nxt
    return pl.concat(preds)


def importance(p, feats, H, at=datetime(2026, 3, 1), train_months=18):
    d = prep(p, feats, H)
    X_cols = [f"r_{c}" for c in feats]
    tr = d.filter((pl.col("t") >= at - relativedelta(months=train_months)) & (pl.col("t") < at)
                  & pl.col("y").is_not_null())
    model = lgb.train(PARAMS, lgb.Dataset(tr.select(X_cols).to_numpy(), tr["y"].to_numpy()), ROUNDS)
    gain = model.feature_importance("gain")
    return sorted(zip(feats, gain / gain.sum()), key=lambda x: -x[1])


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "base"
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    p = make_factors(pl.read_parquet(f"/home/user/data/panel_{tag}.parquet"))
    for name, feats in (("base", BASE_FEATS), ("full", BASE_FEATS + LIQ_FEATS)):
        pr = walk_forward(p, feats, H)
        pr.write_parquet(f"/home/user/data/pred_{tag}_{name}_{H}h.parquet")
        print(name, pr.shape)
