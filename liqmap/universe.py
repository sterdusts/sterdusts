"""动态选币池：每月初按过去 30 天成交额取前 N，剔除稳定币/指数/传统金融永续，要求上线满 60 天。"""
import polars as pl

DATA = "/home/user/data"
EXCL = {"USDCUSDT", "BTCDOMUSDT", "FDUSDUSDT", "DEFIUSDT", "BLUEBIRDUSDT", "FOOTBALLUSDT", "TUSDUSDT",
        "USDPUSDT", "BTCSTUSDT", "XAUUSDT", "XAGUSDT", "PAXGUSDT", "XAUTUSDT", "CLUSDT", "BZUSDT",
        "XPTUSDT", "XPDUSDT", "NATGASUSDT", "COPPERUSDT"}


def build(top_n=40, start=(2022, 1, 1), min_age=60):
    df = pl.read_parquet(f"{DATA}/kl1d.parquet").with_columns(
        pl.from_epoch("open_time", time_unit="ms").dt.date().alias("d"))
    # 传统金融永续：周末成交额远低于工作日
    wk = (df.with_columns((pl.col("d").dt.weekday() >= 6).alias("we"))
            .group_by("symbol").agg((pl.col("quote_volume").filter(pl.col("we")).mean()
                                     / pl.col("quote_volume").filter(~pl.col("we")).mean()).alias("we_ratio")))
    tradfi = set(wk.filter(pl.col("we_ratio").is_null() | (pl.col("we_ratio") < 0.35))["symbol"])
    df = df.filter(~pl.col("symbol").is_in(list(EXCL | tradfi)))
    df = df.sort("symbol", "d").with_columns(
        pl.col("quote_volume").rolling_mean(30).over("symbol").alias("qv30"),
        pl.col("d").cum_count().over("symbol").alias("age"))
    ms = df.filter(pl.col("d").dt.day() == 1, pl.col("d") >= pl.date(*start), pl.col("age") >= min_age)
    ms = ms.with_columns(pl.col("qv30").rank("ordinal", descending=True).over("d").alias("rk"))
    return ms.filter(pl.col("rk") <= top_n).sort("d", "rk").select("d", "symbol", "qv30"), tradfi


if __name__ == "__main__":
    u, tradfi = build()
    print("tradfi excluded:", len(tradfi))
    u.write_parquet(f"{DATA}/universe.parquet")
    syms = sorted(u["symbol"].unique().to_list())
    open(f"{DATA}/union.txt", "w").write(",".join(syms))
    print("union symbols:", len(syms))
    print(u.filter(pl.col("d") == u["d"].max())["symbol"].to_list())
