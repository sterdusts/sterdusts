"""Build /home/user/data/flow/perp_daily.parquet  — key (date, symbol), Binance USDT-M perps, daily, 2021-01-01..2026-09-30.

Time convention (all UTC):
  row `date`=d describes the UTC day [d 00:00, d+1 00:00). Every non-forward column is known at d+1 00:00 UTC
  (= the moment the signal is assumed formed). Entry price for forward returns = close_d (the d+1 00:00 price).
  fwd_*/res_*/fund_* are LABELS (future info) — never features.
"""
import glob, json, os, time, datetime as dt
import polars as pl
import pyarrow.parquet as pq

DATA = "/home/user/data"
OUT = "/home/user/data/flow"
DST = f"{OUT}/perp_daily.parquet"
H = [1, 3, 7, 14]
GLOBAL_END = dt.date(2026, 9, 30)
ARCHIVE_START = dt.date(2020, 1, 1)
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.1f}s]", *a, flush=True)


# ---------------------------------------------------------------- 1. daily klines (+ archive gap fill)
kcols = ["symbol", "open_time", "open", "high", "low", "close", "quote_volume", "taker_buy_quote_volume", "count"]
k = pl.concat([
    pl.read_parquet(f"{DATA}/kl1d.parquet").select(kcols).with_columns(pl.lit(False).alias("gapfill")),
    pl.read_parquet(f"{OUT}/perp_daily_aux_kl1d_gapfill.parquet").select(kcols).with_columns(pl.lit(True).alias("gapfill")),
]).with_columns(pl.from_epoch("open_time", time_unit="ms").dt.date().alias("date")).drop("open_time")
assert k.select("symbol", "date").is_duplicated().sum() == 0
log("klines", k.shape)

span = k.group_by("symbol").agg(pl.col("date").min().alias("a"), pl.col("date").max().alias("b"))
grid = span.with_columns(pl.date_ranges("a", "b").alias("date")).explode("date").select("symbol", "date")
p = grid.join(k, on=["symbol", "date"], how="left")
p = p.with_columns(
    pl.when(pl.col("quote_volume").is_null()).then(pl.lit("missing"))
      .when(pl.col("quote_volume") == 0).then(pl.lit("placeholder"))
      .otherwise(pl.lit("trade")).alias("bar_status"),
    pl.col("gapfill").fill_null(False))
trade = pl.col("bar_status") == "trade"
# placeholder rows: Binance archive writes flat OHLC = stale/settlement price with zero volume -> price set null
p = p.with_columns([pl.when(trade).then(pl.col(c)).otherwise(None).alias(c) for c in ["open", "high", "low", "close"]])
p = p.sort("symbol", "date")
lc = pl.col("close").log()
p = p.with_columns((lc - lc.shift(1)).over("symbol").alias("ret_1d"))  # null unless d-1 and d both 'trade'
log("grid", p.shape)

# ---------------------------------------------------------------- 2. age
fd = pl.read_parquet(f"{OUT}/perp_daily_aux_first_date.parquet").select("symbol", "archive_first_date")
first = span.select("symbol", pl.col("a").alias("kl1d_first")).join(fd, on="symbol", how="left").with_columns(
    pl.min_horizontal("kl1d_first", "archive_first_date").alias("first_kline_date"))
p = p.join(first.select("symbol", "first_kline_date"), on="symbol", how="left").with_columns(
    (pl.col("date") - pl.col("first_kline_date")).dt.total_days().cast(pl.Int32).alias("age_days"),
    (pl.col("first_kline_date") <= ARCHIVE_START).alias("age_censored"))

# ---------------------------------------------------------------- 3. universe membership
u = pl.read_parquet(f"{DATA}/universe.parquet").select(pl.col("d").alias("month"), "symbol").with_columns(
    pl.lit(True).alias("_u"))
umin = u["month"].min()
p = p.with_columns(pl.col("date").dt.truncate("1mo").alias("month")).join(u, on=["month", "symbol"], how="left")
p = p.with_columns(pl.when(pl.col("month") < umin).then(None).otherwise(pl.col("_u").fill_null(False))
                   .alias("in_universe")).drop("_u", "month")

# ---------------------------------------------------------------- 3b. universe.py exclusion rule (static class flag), holdout flag
EXCL = {"USDCUSDT", "BTCDOMUSDT", "FDUSDUSDT", "DEFIUSDT", "BLUEBIRDUSDT", "FOOTBALLUSDT", "TUSDUSDT",
        "USDPUSDT", "BTCSTUSDT", "XAUUSDT", "XAGUSDT", "PAXGUSDT", "XAUTUSDT", "CLUSDT", "BZUSDT",
        "XPTUSDT", "XPDUSDT", "NATGASUSDT", "COPPERUSDT"}  # copied from liqmap/universe.py
_k = pl.read_parquet(f"{DATA}/kl1d.parquet").with_columns(pl.from_epoch("open_time", time_unit="ms").dt.date().alias("d"))
_wk = (_k.with_columns((pl.col("d").dt.weekday() >= 6).alias("we")).group_by("symbol")
         .agg((pl.col("quote_volume").filter(pl.col("we")).mean()
               / pl.col("quote_volume").filter(~pl.col("we")).mean()).alias("we_ratio")))
tradfi = set(_wk.filter(pl.col("we_ratio").is_null() | (pl.col("we_ratio") < 0.35))["symbol"])
p = p.with_columns(pl.col("symbol").is_in(sorted(EXCL | tradfi)).alias("univ_excluded"),
                   (pl.col("date") >= dt.date(2026, 4, 1)).alias("in_holdout"))

# ---------------------------------------------------------------- 4. beta_60d vs BTCUSDT (window = 60 calendar days ending d)
btc = p.filter(pl.col("symbol") == "BTCUSDT").sort("date").select(
    "date", pl.col("ret_1d").alias("btc_ret"),
    *[(pl.col("close").log().shift(-h) - pl.col("close").log()).alias(f"_btcfwd_{h}") for h in H])
assert btc.height == btc["date"].n_unique() and btc["date"].diff().drop_nulls().dt.total_days().max() == 1
p = p.join(btc, on="date", how="left").sort("symbol", "date")
m = pl.col("ret_1d").is_not_null() & pl.col("btc_ret").is_not_null()
x = pl.when(m).then(pl.col("ret_1d")).otherwise(0.0)
y = pl.when(m).then(pl.col("btc_ret")).otherwise(0.0)
W, MINN = 60, 30
p = p.with_columns(
    m.cast(pl.Int32).rolling_sum(W, min_samples=1).over("symbol").alias("beta_n"),
    x.rolling_sum(W, min_samples=1).over("symbol").alias("_sx"),
    y.rolling_sum(W, min_samples=1).over("symbol").alias("_sy"),
    (x * y).rolling_sum(W, min_samples=1).over("symbol").alias("_sxy"),
    (y * y).rolling_sum(W, min_samples=1).over("symbol").alias("_syy"))
n = pl.col("beta_n").cast(pl.Float64)
cov = pl.col("_sxy") / n - pl.col("_sx") * pl.col("_sy") / (n * n)
var = pl.col("_syy") / n - (pl.col("_sy") / n) ** 2
p = p.with_columns(pl.when((pl.col("beta_n") >= MINN) & (var > 1e-12)).then(cov / var).otherwise(None).alias("beta_60d"))
p = p.drop("_sx", "_sy", "_sxy", "_syy")
log("beta done")

# ---------------------------------------------------------------- 5. metrics (5m snapshots) -> daily
mfiles = sorted(glob.glob(f"{DATA}/metrics/*.parquet"))
pos = lambda c: pl.when(pl.col(c) > 0).then(pl.col(c)).otherwise(None).alias(c)  # <=0 / NaN -> null (data glitches)
mt = (pl.scan_parquet(mfiles)
      .with_columns(pl.col("ts").dt.replace_time_zone(None), pl.col("ts").dt.date().alias("date"))
      .with_columns(*[pos(c) for c in ["oi", "oi_value", "top_ls_pos", "top_ls_acct", "ls_acct", "taker_ls"]])
      .with_columns(pl.col(c).fill_nan(None) for c in ["oi", "oi_value", "top_ls_pos", "top_ls_acct", "ls_acct", "taker_ls"])
      .sort("symbol", "ts")
      .group_by("symbol", "date")
      .agg(pl.len().cast(pl.Int32).alias("metrics_n"),
           pl.col("ts").filter(pl.col("oi").is_not_null()).max().alias("metrics_last_ts"),
           *[pl.col(c).drop_nulls().last().alias(c) for c in ["oi", "oi_value", "top_ls_pos", "top_ls_acct", "ls_acct"]],
           pl.col("taker_ls").mean().alias("taker_ls_mean"))
      .collect())
log("metrics daily", mt.shape)
p = p.join(mt, on=["symbol", "date"], how="left")

# ---------------------------------------------------------------- 6. funding
ff = sorted(glob.glob(f"{DATA}/funding_*.parquet")) + [f"{OUT}/perp_daily_aux_funding_backfill.parquet"]
f = pl.concat([pl.read_parquet(x).select("symbol", "calc_time", "funding_interval_hours", "funding_rate") for x in ff])
f = f.with_columns((pl.col("calc_time") // 1000 * 1000).alias("ct")).unique(["symbol", "calc_time"], keep="first")  # exact dups only; 18 tradfi stock-perp cases have 2 distinct settlements in the same second -> both kept
f = f.with_columns(pl.from_epoch("ct", time_unit="ms").alias("fts")).with_columns(
    pl.col("fts").dt.date().alias("date"), pl.col("fts").dt.truncate("1mo").dt.date().alias("month"))
cover = f.select("symbol", "month").unique().with_columns(pl.lit(True).alias("_fcov"))
fd_ = f.group_by("symbol", "date").agg(pl.col("funding_rate").sum().alias("funding_1d"),
                                       pl.len().cast(pl.Int32).alias("funding_n"))
p = (p.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
      .join(cover, on=["symbol", "month"], how="left").join(fd_, on=["symbol", "date"], how="left"))
# month file present -> day with no settlement counts as 0; month file absent -> null (unknown)
p = p.with_columns(
    pl.when(pl.col("_fcov")).then(pl.col("funding_1d").fill_null(0.0)).otherwise(None).alias("funding_1d"),
    pl.when(pl.col("_fcov")).then(pl.col("funding_n").fill_null(0)).otherwise(None).alias("funding_n"),
).drop("_fcov", "month")
log("funding", f.shape)

# ---------------------------------------------------------------- 7. forward labels (built only, never analysed here)
p = p.sort("symbol", "date")
lc = pl.col("close").log()
lc_ff = lc.forward_fill().over("symbol")
td_ff = pl.when(trade).then(pl.col("date")).otherwise(None).forward_fill().over("symbol")
p = p.with_columns(lc_ff.alias("_lcff"), td_ff.alias("_tdff"),
                   lc_ff.last().over("symbol").alias("_lc_last"), td_ff.last().over("symbol").alias("_td_last"))
labels = []
for h in H:
    labels += [
        (lc.shift(-h) - lc).over("symbol").alias(f"fwd_{h}d"),
        pl.col("funding_1d").rolling_sum(h, min_samples=h).shift(-h).over("symbol").alias(f"fund_{h}d"),
        pl.col("_lcff").shift(-h).over("symbol").alias(f"_xlc_{h}"),
        pl.col("_tdff").shift(-h).over("symbol").alias(f"_xtd_{h}"),
    ]
p = p.with_columns(labels)
for h in H:
    inside = (pl.col("date") + pl.duration(days=h)) <= GLOBAL_END
    # beyond the symbol's archive end but before global end (archive stopped after delisting) -> last tradable close
    xlc = pl.when(pl.col(f"_xtd_{h}").is_null() & inside).then(pl.col("_lc_last")).otherwise(pl.col(f"_xlc_{h}"))
    xtd = pl.when(pl.col(f"_xtd_{h}").is_null() & inside).then(pl.col("_td_last")).otherwise(pl.col(f"_xtd_{h}"))
    p = p.with_columns(
        (pl.col(f"fwd_{h}d") - pl.col("beta_60d") * pl.col(f"_btcfwd_{h}")).alias(f"res_{h}d"),
        pl.when(trade & (xtd > pl.col("date"))).then(xlc - lc).otherwise(None).alias(f"fwd_{h}d_lv"))
p = p.drop([c for c in p.columns if c.startswith("_")] + ["btc_ret"])

# ---------------------------------------------------------------- 8. order / types / write
cols = ["date", "symbol", "bar_status", "gapfill", "open", "high", "low", "close", "ret_1d",
        "quote_volume", "taker_buy_quote_volume", "count", "in_universe", "univ_excluded", "in_holdout", "first_kline_date", "age_days", "age_censored",
        "beta_60d", "beta_n",
        "oi", "oi_value", "top_ls_pos", "top_ls_acct", "ls_acct", "taker_ls_mean", "metrics_n", "metrics_last_ts",
        "funding_1d", "funding_n"] \
    + [f"fwd_{h}d" for h in H] + [f"res_{h}d" for h in H] + [f"fund_{h}d" for h in H] + [f"fwd_{h}d_lv" for h in H]
p = p.select(cols).with_columns(pl.col("bar_status").cast(pl.Categorical), pl.col("count").cast(pl.Int64)).sort("date", "symbol")
assert p.select("date", "symbol").is_duplicated().sum() == 0

doc = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "columns.json")))
assert set(doc["columns"]) == set(cols), set(cols) ^ set(doc["columns"])
tbl = p.to_arrow()
tbl = tbl.replace_schema_metadata({**(tbl.schema.metadata or {}),
                                   b"perp_daily_doc": json.dumps(doc, ensure_ascii=False).encode()})
assert not os.path.exists(DST), "refusing to overwrite"
pq.write_table(tbl, DST, compression="zstd")
log("written", DST, p.shape)
