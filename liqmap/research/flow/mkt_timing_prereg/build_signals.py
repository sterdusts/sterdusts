"""市场级资金流择时检测器 MT1/MT2/MT3：预登记信号与控制变量构建（只建信号，不含任何未来收益）。

输出 /home/user/data/flow/mkt_timing_prereg/mkt_timing_signals.parquet，键 date（UTC 日 d），
所有列在 d+1 00:00 UTC 可知（MT1 的稳定币值 value(D=d) 在 d 06:00 即可知）。
参数在 prereg.json 中冻结；本脚本不读取 fwd_*/res_*/fund_* 标签列。
运行：python -I build_signals.py
"""
import json
import hashlib
import pathlib

import numpy as np
import polars as pl

FLOW = pathlib.Path("/home/user/data/flow")
OUT = FLOW / "mkt_timing_prereg" / "mkt_timing_signals.parquet"
STABLE_OR_EXCL = ["BTC", "USDC", "FDUSD", "BUSD", "TUSD", "U", "USD1", "USTC", "PAXG", "XAUT"]


def mt1() -> pl.DataFrame:
    s = pl.read_parquet(FLOW / "stablecoins_daily.parquet").sort("date")
    cols = ["usdt_total", "usdc_total", "busd_total", "fdusd_total", "tusd_total"]
    s = s.with_columns(pl.sum_horizontal([pl.col(c).fill_null(0) for c in cols]).alias("mt1_F"))
    s = s.with_columns((pl.col("mt1_F").log() - pl.col("mt1_F").log().shift(1)).alias("dl"))
    s = s.with_columns(
        pl.when(pl.col("date") == pl.date(2022, 1, 2)).then(0.0).otherwise(pl.col("dl"))
        .clip(-0.01, 0.01).fill_null(0.0).alias("mt1_dlc"))
    s = s.with_columns(pl.col("mt1_dlc").cum_sum().alias("L"))
    s = s.with_columns((pl.col("L") - pl.col("L").shift(14)).alias("mt1_g14"),
                       (pl.col("L").shift(14) - pl.col("L").shift(180)).alias("mt1_g166"))
    s = s.with_columns((pl.col("mt1_g14") - 14 / 166 * pl.col("mt1_g166")).alias("mt1_I"))
    s = s.with_columns(pl.when(pl.col("mt1_I").is_null()).then(None)
                       .when(pl.col("mt1_I") > 0).then(1).otherwise(-1).cast(pl.Int8).alias("mt1_state"))
    return s.select("date", "mt1_F", "mt1_dlc", "mt1_g14", "mt1_g166", "mt1_I", "mt1_state",
                    pl.col("known_at").alias("mt1_known_at"))


def mt2() -> pl.DataFrame:
    c = pl.read_parquet(FLOW / "coinbase_premium_daily.parquet").filter(pl.col("asset") == "BTC").sort("date")
    c = c.with_columns(pl.when(pl.col("n_hours_prem_adj") >= 20).then(pl.col("prem_adj_median")).alias("mt2_p"))
    c = c.with_columns(pl.col("mt2_p").rolling_mean(7, min_samples=5).alias("mt2_P7"),
                       pl.col("mt2_p").shift(7).rolling_median(91, min_samples=60).alias("mt2_B"))
    c = c.with_columns((pl.col("mt2_P7") - pl.col("mt2_B")).alias("mt2_S"))
    c = c.with_columns(pl.when(pl.col("mt2_S").is_null()).then(None)
                       .when(pl.col("mt2_S") > 0).then(1).otherwise(-1).cast(pl.Int8).alias("mt2_state"))
    return c.select("date", "mt2_p", "mt2_P7", "mt2_B", "mt2_S", "mt2_state")


def mt3() -> pl.DataFrame:
    k = pl.read_parquet(FLOW / "kimchi.parquet").sort("date")
    k = k.with_columns(pl.col("kimchi_fx_ecb").alias("mt3_k"))
    k = k.with_columns(pl.col("mt3_k").rolling_mean(7, min_samples=7).alias("mt3_K7"),
                       pl.col("mt3_k").shift(7).rolling_median(91, min_samples=60).alias("mt3_KB"))
    k = k.with_columns((pl.col("mt3_K7") - pl.col("mt3_KB")).alias("mt3_S"))
    # 反向：韩国溢价低于自身 90 日常态 = 有利状态（+1）；高于常态 = 过热（-1）
    k = k.with_columns(pl.when(pl.col("mt3_S").is_null()).then(None)
                       .when(pl.col("mt3_S") <= 0).then(1).otherwise(-1).cast(pl.Int8).alias("mt3_state"))
    return k.select("date", "mt3_k", "mt3_K7", "mt3_KB", "mt3_S", "mt3_state", "fx_age_days")


def controls() -> pl.DataFrame:
    p = (pl.read_parquet(FLOW / "perp_daily.parquet",
                         columns=["date", "symbol", "bar_status", "close", "ret_1d", "funding_1d", "oi"])
         .filter(pl.col("symbol") == "BTCUSDT").sort("date"))
    assert p["date"].is_unique().all() and (p["bar_status"] == "trade").all()
    p = p.with_columns(
        (pl.col("close").log() - pl.col("close").log().shift(7)).alias("c_mom7"),
        (pl.col("close").log() - pl.col("close").log().shift(30)).alias("c_mom30"),
        pl.col("ret_1d").rolling_std(30, min_samples=25).alias("c_rv30"),
        pl.col("funding_1d").rolling_sum(7, min_samples=7).alias("c_fund7"),
        (pl.col("oi").log() - pl.col("oi").log().shift(7)).alias("c_doi7"),
    )
    b = (pl.read_parquet(FLOW / "spot_daily_base.parquet", columns=["date", "base", "qv_usdt_USDT", "tbqv_usdt_USDT"])
         .filter(~pl.col("base").is_in(STABLE_OR_EXCL))
         .group_by("date").agg(pl.col("qv_usdt_USDT").sum().alias("qv"), pl.col("tbqv_usdt_USDT").sum().alias("tb"))
         .sort("date"))
    b = b.with_columns((pl.col("tb").rolling_sum(7, min_samples=7) / pl.col("qv").rolling_sum(7, min_samples=7) - 0.5)
                       .alias("c_tkr7"))
    return p.select("date", "c_mom7", "c_mom30", "c_rv30", "c_fund7", "c_doi7").join(
        b.select("date", "c_tkr7"), on="date", how="left")


def main():
    cal = pl.DataFrame({"date": pl.date_range(pl.date(2021, 1, 1), pl.date(2026, 9, 30), "1d", eager=True)})
    df = (cal.join(mt1(), on="date", how="left").join(mt2(), on="date", how="left")
          .join(mt3(), on="date", how="left").join(controls(), on="date", how="left"))
    df = df.with_columns(
        pl.col("date").cast(pl.Datetime("us", "UTC")).dt.offset_by("1d").alias("decision_at"),
        (pl.col("date") >= pl.date(2026, 4, 1)).alias("in_holdout"),
        pl.col("date").is_between(pl.date(2022, 1, 1), pl.date(2026, 3, 31)).alias("in_dev"),
    )
    assert df["date"].is_unique().all()
    assert not OUT.exists(), f"{OUT} 已存在，拒绝覆盖"
    df.write_parquet(OUT)
    src = pathlib.Path(__file__).read_bytes()
    print(json.dumps({"rows": df.height, "cols": df.width, "script_sha256": hashlib.sha256(src).hexdigest()}))
    d = df.filter(pl.col("in_dev"))
    print(d.select([pl.col(c).null_count().alias(c) for c in d.columns if c.startswith(("mt", "c_"))]))
    for st in ("mt1_state", "mt2_state", "mt3_state"):
        x = d[st].to_numpy().astype(float)
        print(st, "favorable_frac", round(float(np.nanmean(x > 0)), 3), "switches", int(np.nansum(x[1:] != x[:-1])))


if __name__ == "__main__":
    main()
