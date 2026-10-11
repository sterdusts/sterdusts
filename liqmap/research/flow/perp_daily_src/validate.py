"""Read back perp_daily.parquet and check structure / coverage / spot-checks (no signal-vs-return statistics)."""
import json, datetime as dt
import numpy as np
import polars as pl
import pyarrow.parquet as pq

pl.Config.set_tbl_cols(20); pl.Config.set_tbl_width_chars(230); pl.Config.set_tbl_rows(60)
F = "/home/user/data/flow/perp_daily.parquet"
p = pl.read_parquet(F)
print("shape", p.shape, " symbols", p["symbol"].n_unique(), " dates", p["date"].min(), p["date"].max())
print("dup keys", p.select("date", "symbol").is_duplicated().sum())
print("sorted by (date,symbol):", p.select("date", "symbol").equals(p.select("date", "symbol").sort("date", "symbol")))
print(p["bar_status"].value_counts().sort("count", descending=True))
print("gapfill rows", p["gapfill"].sum())
print("missing rows by symbol", p.filter(pl.col("bar_status") == "missing").group_by("symbol").agg(pl.len(), pl.col("date").min().alias("a"), pl.col("date").max().alias("b")))
meta = json.loads(pq.read_schema(F).metadata[b"perp_daily_doc"])
print("doc columns == file columns:", set(meta["columns"]) == set(p.columns))

# null rates overall and on trade rows
tr = p.filter(pl.col("bar_status") == "trade")
nr = pl.DataFrame({"col": p.columns,
                   "null_all": [p[c].null_count() / p.height for c in p.columns],
                   "null_trade": [tr[c].null_count() / tr.height for c in p.columns]})
print(nr.with_columns(pl.col("null_all", "null_trade").round(4)))

# universe consistency
u = pl.read_parquet("/home/user/data/universe.parquet")
pu = p.filter(pl.col("in_universe")).select(pl.col("date").dt.truncate("1mo").alias("d"), "symbol").unique()
print("universe pairs", u.height, " pairs found in panel", pu.height,
      " universe pairs not in panel", u.select("d", "symbol").join(pu, on=["d", "symbol"], how="anti").height)
print("in_universe null dates:", p.filter(pl.col("in_universe").is_null())["date"].max())
print("in_universe trade-row counts/day (2022+):", tr.filter(pl.col("in_universe")).group_by("date").len()["len"].describe())

# beta checks
print("BTC beta range", p.filter(pl.col("symbol") == "BTCUSDT")["beta_60d"].drop_nulls().min(), p.filter(pl.col("symbol") == "BTCUSDT")["beta_60d"].drop_nulls().max())
btc = p.filter(pl.col("symbol") == "BTCUSDT").select("date", pl.col("ret_1d").alias("b"))
for sym, d in [("ETHUSDT", dt.date(2023, 6, 30)), ("SOLUSDT", dt.date(2022, 4, 10)), ("DOGEUSDT", dt.date(2025, 1, 15))]:
    w = p.filter(pl.col("symbol") == sym, pl.col("date").is_between(d - dt.timedelta(days=59), d)).join(btc, on="date").drop_nulls(["ret_1d", "b"])
    x, y = w["ret_1d"].to_numpy(), w["b"].to_numpy()
    ref = np.cov(x, y, ddof=0)[0, 1] / np.var(y)
    got = p.filter(pl.col("symbol") == sym, pl.col("date") == d)["beta_60d"][0]
    print(f"beta check {sym} {d}: panel={got:.6f} numpy={ref:.6f} n={len(x)}")
print("beta_60d quantiles (trade rows):", tr["beta_60d"].quantile(0.001), tr["beta_60d"].quantile(0.5), tr["beta_60d"].quantile(0.999))

# fwd / res / fund spot checks
s = p.filter(pl.col("symbol") == "ETHUSDT").sort("date")
i = s["date"].to_list().index(dt.date(2024, 2, 1))
c = s["close"].to_numpy()
print("fwd_7d check ETH 2024-02-01:", s["fwd_7d"][i], np.log(c[i + 7] / c[i]))
f1 = s["funding_1d"].to_numpy()
print("fund_7d check:", s["fund_7d"][i], f1[i + 1:i + 8].sum())
bs = p.filter(pl.col("symbol") == "BTCUSDT").sort("date")
bi = bs["date"].to_list().index(dt.date(2024, 2, 1))
bc = bs["close"].to_numpy()
print("res_7d check:", s["res_7d"][i], np.log(c[i + 7] / c[i]) - s["beta_60d"][i] * np.log(bc[bi + 7] / bc[bi]))

# funding raw check
fr = pl.concat([pl.read_parquet(x) for x in
                ["/home/user/data/funding_0.parquet", "/home/user/data/funding_25.parquet", "/home/user/data/funding_50.parquet",
                 "/home/user/data/funding_75.parquet", "/home/user/data/funding_100.parquet", "/home/user/data/funding_125.parquet",
                 "/home/user/data/funding_150.parquet", "/home/user/data/funding_175.parquet", "/home/user/data/funding_200.parquet",
                 "/home/user/data/funding_225.parquet", "/home/user/data/funding_250.parquet", "/home/user/data/funding_275.parquet",
                 "/home/user/data/funding_300.parquet", "/home/user/data/funding_extra.parquet",
                 "/home/user/data/flow/perp_daily_aux_funding_backfill.parquet"]])
fr = fr.filter(pl.col("symbol") == "BTCUSDT").with_columns(pl.from_epoch(pl.col("calc_time") // 1000, time_unit="s").alias("t"))
day = fr.filter(pl.col("t").dt.date() == dt.date(2023, 5, 10))
print("BTC 2023-05-10 settlements", day.select("t", "funding_rate").to_dicts(), " sum", day["funding_rate"].sum(),
      " panel", p.filter(pl.col("symbol") == "BTCUSDT", pl.col("date") == dt.date(2023, 5, 10))["funding_1d"][0])

# funding coverage on trade rows
print("trade rows funding null:", tr["funding_1d"].null_count(), " funding_n==0:", tr.filter(pl.col("funding_n") == 0).height)
fz = tr.filter(pl.col("funding_n") == 0).with_columns(
    (pl.col("age_days") == 0).alias("listing_day"))
print("  of which listing day:", fz["listing_day"].sum())
print(fz.filter(~pl.col("listing_day")).group_by(pl.col("date").dt.strftime("%Y-%m").alias("m")).len().sort("len", descending=True).head(10))
print("funding_n distribution (trade rows):", tr["funding_n"].value_counts().sort("count", descending=True).head(10))
print("funding null trade rows by year:", tr.filter(pl.col("funding_1d").is_null()).group_by(pl.col("date").dt.year()).len().sort("date"))

# metrics coverage
mc = tr.filter(pl.col("metrics_n").is_not_null())
print("metrics rows", mc.height, " symbols", mc["symbol"].n_unique(), " metrics_n<280:", mc.filter(pl.col("metrics_n") < 280).height,
      " in_universe & metrics missing:", tr.filter(pl.col("in_universe") & pl.col("oi").is_null()).height)
print("in_universe & oi null by month:", tr.filter(pl.col("in_universe") & pl.col("oi").is_null()).group_by(pl.col("date").dt.strftime("%Y-%m").alias("m")).len().sort("len", descending=True).head(8))
print("metrics_last_ts hour dist:", mc["metrics_last_ts"].dt.hour().value_counts().sort("count", descending=True).head(5))

# label availability
for h in [1, 3, 7, 14]:
    print(f"h={h}: fwd non-null {tr[f'fwd_{h}d'].is_not_null().sum()}, lv non-null {tr[f'fwd_{h}d_lv'].is_not_null().sum()}, "
          f"strict-null-but-lv {tr.filter(pl.col(f'fwd_{h}d').is_null() & pl.col(f'fwd_{h}d_lv').is_not_null()).height}, "
          f"last fwd date {tr.filter(pl.col(f'fwd_{h}d').is_not_null())['date'].max()}")
print("age: censored symbols", p.filter(pl.col("age_censored"))["symbol"].unique().to_list(),
      " age<0:", p.filter(pl.col("age_days") < 0).height)
print("ret_1d extreme |r|>1:", tr.filter(pl.col("ret_1d").abs() > 1).select("date", "symbol", "ret_1d", "quote_volume").sort("date"))
print("holdout rows (date>=2026-04-01):", p.filter(pl.col("date") >= dt.date(2026, 4, 1)).height)
