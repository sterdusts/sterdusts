"""Pool (universe.parquet) spot coverage by year. usage: python -I coverage.py <outdir> [write]"""
import sys
import polars as pl
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_cols(30)
O = sys.argv[1]
u = pl.read_parquet("/home/user/data/universe.parquet").with_columns(pl.col("d").dt.strftime("%Y-%m").alias("ym"))
p = pl.read_parquet(f"{O}/spot_daily_perp.parquet").with_columns(pl.col("date").dt.strftime("%Y-%m").alias("ym"))
kl = (pl.read_parquet("/home/user/data/kl1d.parquet", columns=["open_time", "symbol", "quote_volume"])
        .with_columns(pl.from_epoch("open_time", time_unit="ms").dt.date().alias("date"))
        .rename({"symbol": "perp_symbol", "quote_volume": "perp_qv"}).drop("open_time"))
x = (p.join(u.select("ym", pl.col("symbol").alias("perp_symbol")), on=["ym", "perp_symbol"], how="inner")
       .join(kl, on=["date", "perp_symbol"], how="left")
       .with_columns(pl.when(pl.col("date") >= pl.date(2026, 4, 1)).then(pl.lit("2026-04+(locked)"))
                       .when(pl.col("date").dt.year() == 2026).then(pl.lit("2026-01..03"))
                       .otherwise(pl.col("date").dt.year().cast(pl.String)).alias("period")))
cov = (x.group_by("period").agg(
        pl.len().alias("pool_sym_days"),
        pl.col("perp_symbol").n_unique().alias("pool_syms"),
        pl.col("has_spot").mean().round(4).alias("cov_days"),
        ((pl.col("perp_qv") * pl.col("has_spot")).sum() / pl.col("perp_qv").sum()).round(4).alias("cov_perp_qv_wt"),
        pl.col("perp_symbol").filter(pl.col("has_spot")).n_unique().alias("syms_with_spot"),
        (pl.col("has_spot") & (pl.col("qv_usdt_USDT") > 0)).mean().round(4).alias("cov_USDT_pair"),
        (pl.col("has_spot") & (pl.col("qv_usdt_FDUSD") > 0)).mean().round(4).alias("cov_FDUSD"),
        (pl.col("has_spot") & (pl.col("qv_usdt_USDC") > 0)).mean().round(4).alias("cov_USDC"),
        (pl.col("has_spot") & (pl.col("qv_usdt_BUSD") > 0)).mean().round(4).alias("cov_BUSD"),
        (pl.col("has_spot") & (pl.col("qv_usdt_TUSD") > 0)).mean().round(4).alias("cov_TUSD"),
        (pl.col("qv_usdt") / pl.col("perp_qv")).filter(pl.col("has_spot")).median().round(4).alias("med_spot_perp_qv_ratio"),
        (pl.col("qv_usdt").filter(pl.col("has_spot")).sum() / pl.col("perp_qv").filter(pl.col("has_spot")).sum()).round(4).alias("agg_spot_perp_qv_ratio"),
     ).sort("period"))
print(cov)
nos = (x.filter(~pl.col("has_spot")).group_by("period").agg(pl.col("perp_symbol").unique().sort().alias("syms_without_spot"))
         .sort("period"))
for per, syms in nos.iter_rows():
    print(per, len(syms), ",".join(syms))
if len(sys.argv) > 2 and sys.argv[2] == "write":
    cov.write_parquet(f"{O}/spot_coverage_by_year.parquet")
