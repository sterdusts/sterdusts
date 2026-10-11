"""Build hourly + daily Coinbase premium (BTC/ETH/SOL) from downloaded Coinbase 1h candles and Binance spot 1h archive.
ts = hour OPEN time (UTC); a row with ts=h describes [h, h+1h) and is knowable at h+1h (bar close).
prem_raw = cb_close(USD) / bn_close(USDT) - 1   (contains the USDT/USD basis)
prem_adj = cb_close / (bn_close * usdt_usd_cb) - 1, usdt_usd_cb = Coinbase USDT-USD close of the same hour
           (forward-filled at most 6h from earlier hours if that hour had no USDT-USD trade)
prem_adj_usdc = cb_close * usdcusdt_bn / bn_close - 1 (alt: Binance USDCUSDT, assumes USDC == 1 USD)."""
import sys, datetime as dt
import polars as pl

SCR, OUT_H, OUT_D, OUT_RAW = sys.argv[1:5]
START = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
END = dt.datetime(2026, 9, 30, 23, tzinfo=dt.timezone.utc)
ASSETS = ["BTC", "ETH", "SOL"]
FFILL_MAX_H = 6
HOLDOUT = dt.date(2026, 4, 1)

grid = pl.DataFrame({"ts": pl.datetime_range(START, END, "1h", eager=True, time_zone="UTC")}).with_columns(
    pl.col("ts").dt.cast_time_unit("us"))


def cb(pid):
    return pl.read_parquet(f"{SCR}/cb1h_{pid}.parquet").with_columns(pl.col("ts").dt.cast_time_unit("us"))


def bn(sym):
    d = pl.read_parquet(f"{SCR}/bn1h_{sym}.parquet").with_columns(pl.col("ts").dt.cast_time_unit("us"))
    # zero-volume archive rows are placeholders (exchange halted): no valid price
    return d.with_columns([pl.when(pl.col("volume") > 0).then(pl.col(c)).otherwise(None).alias(c)
                           for c in ("open", "high", "low", "close")])


# ---- stablecoin conversion series on the hourly grid ----
usdt = grid.join(cb("USDT-USD").select("ts", pl.col("close").alias("usdt_usd_cb_raw")), on="ts", how="left")
usdt = usdt.with_columns(pl.when(pl.col("usdt_usd_cb_raw").is_not_null()).then(pl.col("ts")).alias("_last"))
usdt = usdt.with_columns(pl.col("_last").forward_fill())
usdt = usdt.with_columns(((pl.col("ts") - pl.col("_last")).dt.total_hours()).alias("usdt_usd_cb_age_h"))
usdt = usdt.with_columns(pl.col("usdt_usd_cb_raw").forward_fill().alias("_ff"))
usdt = usdt.with_columns(
    pl.when(pl.col("usdt_usd_cb_age_h") <= FFILL_MAX_H).then(pl.col("_ff")).otherwise(None).alias("usdt_usd_cb"),
    pl.when(pl.col("usdt_usd_cb_age_h") <= FFILL_MAX_H).then(pl.col("usdt_usd_cb_age_h")).otherwise(None).alias("usdt_usd_cb_age_h"),
).drop("_last", "_ff")
usdc = bn("USDCUSDT").select("ts", pl.col("close").alias("usdcusdt_bn"))

# ---- per-asset hourly panel ----
parts, raw_parts = [], []
for a in ASSETS:
    c = cb(f"{a}-USD")
    b = bn(f"{a}USDT")
    raw_parts.append(c.select(pl.lit(f"{a}-USD").alias("product"), "ts", "open", "high", "low", "close", "volume"))
    h = (grid.join(c.select("ts", *[pl.col(x).alias(f"cb_{x}") for x in ("open", "high", "low", "close")],
                           pl.col("volume").alias("cb_volume_base")), on="ts", how="left")
             .join(b.select("ts", pl.col("close").alias("bn_close"), pl.col("volume").alias("bn_volume_base"),
                            pl.col("quote_volume").alias("bn_quote_volume"),
                            pl.col("taker_buy_quote_volume").alias("bn_taker_buy_quote_volume")), on="ts", how="left")
             .join(usdt, on="ts", how="left").join(usdc, on="ts", how="left")
             .with_columns(pl.lit(a).alias("asset")))
    parts.append(h)
raw_parts.append(cb("USDT-USD").select(pl.lit("USDT-USD").alias("product"), "ts", "open", "high", "low", "close", "volume"))

hr = pl.concat(parts).with_columns(
    (pl.col("ts") + pl.duration(hours=1)).alias("known_at"),
    (pl.col("ts").dt.date() >= HOLDOUT).alias("in_holdout"),
    (pl.col("cb_close") / pl.col("bn_close") - 1).alias("prem_raw"),
    (pl.col("cb_close") / (pl.col("bn_close") * pl.col("usdt_usd_cb")) - 1).alias("prem_adj"),
    (pl.col("cb_close") * pl.col("usdcusdt_bn") / pl.col("bn_close") - 1).alias("prem_adj_usdc"),
    (pl.col("cb_volume_base") * (pl.col("cb_high") + pl.col("cb_low") + pl.col("cb_close")) / 3).alias("cb_volume_usd_approx"),
)
hr = hr.with_columns((pl.col("cb_volume_usd_approx") /
                      (pl.col("cb_volume_usd_approx") + pl.col("bn_quote_volume"))).alias("cb_vol_share"))
hcols = ["ts", "asset", "known_at", "in_holdout", "prem_raw", "prem_adj", "prem_adj_usdc",
         "cb_open", "cb_high", "cb_low", "cb_close", "cb_volume_base", "cb_volume_usd_approx",
         "bn_close", "bn_volume_base", "bn_quote_volume", "bn_taker_buy_quote_volume", "cb_vol_share",
         "usdt_usd_cb", "usdt_usd_cb_raw", "usdt_usd_cb_age_h", "usdcusdt_bn"]
hr = hr.select(hcols).sort(["asset", "ts"])
hr.write_parquet(OUT_H, compression="zstd")

# ---- daily: UTC day d = hours [d 00:00, d+1 00:00), knowable at d+1 00:00 ----
last = hr.filter(pl.col("ts").dt.hour() == 23).select(
    pl.col("ts").dt.date().alias("date"), "asset",
    pl.col("prem_raw").alias("prem_raw_close"), pl.col("prem_adj").alias("prem_adj_close"),
    pl.col("prem_adj_usdc").alias("prem_adj_usdc_close"),
    pl.col("cb_close").alias("cb_close_d"), pl.col("bn_close").alias("bn_close_d"),
    pl.col("usdt_usd_cb").alias("usdt_usd_cb_close"))
dy = (hr.group_by(pl.col("ts").dt.date().alias("date"), "asset")
        .agg(pl.len().alias("n_hours"),
             pl.col("prem_raw").is_not_null().sum().alias("n_hours_prem_raw"),
             pl.col("prem_adj").is_not_null().sum().alias("n_hours_prem_adj"),
             pl.col("prem_raw").mean().alias("prem_raw_mean"),
             pl.col("prem_adj").mean().alias("prem_adj_mean"),
             pl.col("prem_adj").median().alias("prem_adj_median"),
             pl.col("prem_adj_usdc").mean().alias("prem_adj_usdc_mean"),
             pl.col("usdt_usd_cb_raw").mean().alias("usdt_usd_cb_mean"),
             pl.col("cb_volume_usd_approx").sum().alias("cb_volume_usd_approx"),
             pl.col("cb_volume_usd_approx").is_not_null().sum().alias("n_hours_cb"),
             pl.col("bn_quote_volume").sum().alias("bn_quote_volume"),
             pl.col("bn_quote_volume").is_not_null().sum().alias("n_hours_bn"),
             pl.col("bn_taker_buy_quote_volume").sum().alias("bn_taker_buy_quote_volume"))
        .join(last, on=["date", "asset"], how="left"))
dy = dy.with_columns(
    pl.when(pl.col("n_hours_cb") > 0).then(pl.col("cb_volume_usd_approx")).otherwise(None).alias("cb_volume_usd_approx"),
    pl.when(pl.col("n_hours_bn") > 0).then(pl.col("bn_quote_volume")).otherwise(None).alias("bn_quote_volume"),
    pl.when(pl.col("n_hours_bn") > 0).then(pl.col("bn_taker_buy_quote_volume")).otherwise(None).alias("bn_taker_buy_quote_volume"),
)
dy = dy.with_columns(
    (pl.col("cb_volume_usd_approx") / (pl.col("cb_volume_usd_approx") + pl.col("bn_quote_volume"))).alias("cb_vol_share"),
    ((pl.col("date").cast(pl.Datetime("us")) + pl.duration(days=1)).dt.replace_time_zone("UTC")).alias("known_at"),
    (pl.col("date") >= HOLDOUT).alias("in_holdout"),
)
dcols = ["date", "asset", "known_at", "in_holdout", "prem_raw_close", "prem_adj_close", "prem_adj_usdc_close",
         "prem_raw_mean", "prem_adj_mean", "prem_adj_median", "prem_adj_usdc_mean",
         "n_hours", "n_hours_prem_raw", "n_hours_prem_adj", "n_hours_cb", "n_hours_bn",
         "cb_close_d", "bn_close_d", "usdt_usd_cb_close", "usdt_usd_cb_mean",
         "cb_volume_usd_approx", "bn_quote_volume", "bn_taker_buy_quote_volume", "cb_vol_share"]
dy = dy.select(dcols).sort(["asset", "date"])
dy.write_parquet(OUT_D, compression="zstd")

# ---- raw Coinbase candles (aux) ----
pl.concat(raw_parts).sort(["product", "ts"]).write_parquet(OUT_RAW, compression="zstd")
print(hr.shape, dy.shape)
