import json
from datetime import date
import numpy as np
import polars as pl
from scipy import stats
D0 = date(2022, 1, 1)

def ic_daily(df, ycol, resid="e_A", minn=30):
    out = []
    for (d,), g in df.filter(pl.col(ycol).is_not_null()).group_by(["date"]):
        if g.height < minn:
            continue
        r = stats.spearmanr(g[resid].to_numpy(), g[ycol].to_numpy()).statistic
        out.append((d, r, g.height))
    return pl.DataFrame(out, schema=["date", "ic", "n"], orient="row").sort("date")

def nw_se(x, lag=2):
    x = np.asarray(x, float); T = len(x); u = x - x.mean()
    s = u @ u / T
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (u[l:] @ u[:-l]) / T
    return np.sqrt(s / T)

def summarize(icd, label_):
    w = (icd.with_columns(pl.col("date").dt.truncate("1w").alias("wk"))
         .group_by("wk").agg(pl.col("ic").mean(), pl.len().alias("nd")).filter(pl.col("nd") >= 4).sort("wk"))
    W = w["ic"].to_numpy()
    m = W.mean(); se = nw_se(W); t = m / se
    ph = icd.with_columns(((pl.col("date") - pl.lit(D0)).dt.total_days() % 7).alias("ph")).group_by("ph").agg(
        pl.col("ic").mean().alias("m"), (pl.col("ic").mean() / (pl.col("ic").std() / pl.len().sqrt())).alias("t")).sort("ph")
    yr = w.group_by(pl.col("wk").dt.year().alias("y")).agg(pl.col("ic").mean()).sort("y")
    r = dict(IC=float(m), se=float(se), t=float(t), n_weeks=len(W), n_days=icd.height, mean_n=float(icd["n"].mean()),
             phases_mean=[round(x, 5) for x in ph["m"].to_list()], phases_t=[round(x, 2) for x in ph["t"].to_list()],
             phases_pos=int((ph["m"] > 0).sum()), yearly=dict(zip(yr["y"].to_list(), [round(x, 5) for x in yr["ic"].to_list()])))
    print(label_, json.dumps(r))
    return r

