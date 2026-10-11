"""Read back Upbit flow outputs with polars and print row counts, null counts, time ranges, key checks.
Usage: python3 -I validate_upbit.py <dir> [spot_daily.parquet]"""
import sys, polars as pl
D = sys.argv[1]
pl.Config.set_tbl_cols(60); pl.Config.set_tbl_width_chars(250); pl.Config.set_tbl_rows(80)
files = ["upbit_daily", "kimchi", "upbit_listings", "upbit_perp_map", "upbit_perp_daily"]
for f in files:
    df = pl.read_parquet(f"{D}/{f}.parquet")
    nc = {c: n for c, n in zip(df.columns, df.null_count().row(0)) if n}
    rng = {c: (str(df[c].min()), str(df[c].max())) for c in df.columns if c in ("date", "first_bar_date")}
    print(f"== {f}: rows={df.height} cols={df.width} ranges={rng}")
    print("   nulls:", nc)

d = pl.read_parquet(f"{D}/upbit_daily.parquet")
print("daily: markets", d["market"].n_unique(), "dup(date,market)", d.group_by("date", "market").len()["len"].max())
print("daily: candle cut check (all candle_date_time_utc at 00:00 UTC):", (d["candle_date_time_utc"].dt.hour() == 0).all(),
      " kst 09:00:", (d["candle_date_time_kst"].dt.hour() == 9).all())
print("daily: available_at == date+1d 00:00 UTC:", d.select((pl.col("available_at") == pl.col("candle_date_time_utc") + pl.duration(days=1)).all()).item())
print("daily: close_staleness_min min/median/p99/max:", d.select(pl.col("close_staleness_min").min(), pl.col("close_staleness_min").median().alias("med"),
      pl.col("close_staleness_min").quantile(0.99).alias("p99"), pl.col("close_staleness_min").max().alias("mx")).row(0))
print("daily: rows with last trade after window end:", d.filter(pl.col("close_staleness_min") < 0).height,
      " max overshoot sec:", -60 * d["close_staleness_min"].min())
print("daily: rows per year / markets per year")
print(d.group_by(pl.col("date").dt.year().alias("y")).agg(pl.len(), pl.col("market").n_unique().alias("mkts"),
      (pl.col("acc_trade_price_krw").sum() / 1e12).alias("turnover_tKRW"), (pl.col("acc_trade_price_krw") == 0).sum().alias("zero_turnover")).sort("y"))
print("daily: gaps (days_since_prev_bar>1) since 2021:", d.filter((pl.col("days_since_prev_bar") > 1) & (pl.col("date") >= pl.date(2021, 1, 1))).height)
print("daily: high>=max(open,close)&low<=min:", d.filter((pl.col("high_krw") < pl.max_horizontal("open_krw", "close_krw") - 1e-9) |
      (pl.col("low_krw") > pl.min_horizontal("open_krw", "close_krw") + 1e-9)).height, "violations")
print("daily: prev_close_api == previous row close (consecutive days, since 2021):",
      d.sort("market", "date").with_columns(pl.col("close_krw").shift(1).over("market").alias("pc")).filter(
          (pl.col("days_since_prev_bar") == 1) & (pl.col("date") >= pl.date(2021, 1, 1))).select(
          ((pl.col("pc") - pl.col("prev_close_krw_api")).abs() / pl.col("pc") < 1e-9).mean()).item())

k = pl.read_parquet(f"{D}/kimchi.parquet")
print("kimchi: days", k.height, k["date"].min(), k["date"].max(), "contiguous:", k["date"].diff().dt.total_days().drop_nulls().max() == 1)
print("kimchi: bn bar open hour all 23:", (k["bn_bar_open_utc"].dt.hour() == 23).all(), " bn bar date == date:", (k["bn_bar_open_utc"].dt.date() == k["date"]).all())
print("kimchi: first KRW-USDT day:", k.filter(pl.col("krw_usdt_close").is_not_null())["date"].min())
print(k.select([pl.col(c).quantile(q).round(5).alias(f"{c}@{q}") for c in ["kimchi_fx_ecb", "prem_btc_vs_krwusdt", "usdt_prem_fx_ecb", "kimchi_fx_fred"] for q in [0.01, 0.5, 0.99]]).transpose(include_header=True))
print("kimchi: identity (1+kimchi_fx)=(1+usdt_prem)(1+prem_vs_usdt) max abs err:",
      k.select(((1 + pl.col("kimchi_fx_ecb")) - (1 + pl.col("usdt_prem_fx_ecb")) * (1 + pl.col("prem_btc_vs_krwusdt"))).abs().max()).item())
if len(sys.argv) > 2:
    s = pl.scan_parquet(sys.argv[2]).filter((pl.col("base") == "BTC") & (pl.col("quote") == "USDT")).select("date", pl.col("close").alias("spot_close")).collect()
    j = k.join(s, on="date", how="inner")
    print("kimchi: Binance 1h(23:00 bar) close == spot_daily BTCUSDT 1d close:", (j["btcusdt_bn_close"] == j["spot_close"]).mean(), "on", j.height, "days")

li = pl.read_parquet(f"{D}/upbit_listings.parquet")
print("listings: n", li.height, "first_bar_date range", li["first_bar_date"].min(), li["first_bar_date"].max(), "left-censored", li["listing_left_censored"].sum())
print(li.group_by(pl.col("first_bar_date").dt.year().alias("y")).agg(pl.len(), pl.col("is_stable").sum().alias("stables")).sort("y"))
print("listings: first_hour within first_bar_date:", (li["first_hour_utc"].dt.date() == li["first_bar_date"]).all())
print("listings: with predecessor alias:", li.filter(pl.col("predecessor_perp_ok").is_not_null()).select("market", "first_bar_date", "predecessor_perp_ok").rows())

m = pl.read_parquet(f"{D}/upbit_perp_map.parquet")
print(m.group_by("status").agg(pl.len(), pl.col("market").n_unique().alias("markets")).sort("status"))
ok = m.filter(pl.col("status") == "ok")
print("map: Upbit markets with >=1 ok perp:", ok["market"].n_unique(), "/", li.height, "; perps used:", ok["perp_symbol"].n_unique())
print("map: mult_effective != mult_prefix:", m.filter(pl.col("mult_effective") != pl.col("mult_prefix")).select("market", "perp_symbol", "mult_prefix", "mult_effective", "status").rows())
print("map: perps mapped to >1 market:", ok.group_by("perp_symbol").len().filter(pl.col("len") > 1).rows())
p = pl.read_parquet(f"{D}/upbit_perp_daily.parquet")
print("perp_daily: (date,market) with >1 perp:", p.group_by("date", "market").len().filter(pl.col("len") > 1).height,
      "; (date,perp) with >1 market:", p.group_by("date", "perp_symbol").len().filter(pl.col("len") > 1).height)
print("perp_daily: px_match_10pct share:", p["px_match_10pct"].mean())
print("perp_daily rows per year:", p.group_by(pl.col("date").dt.year().alias("y")).agg(pl.len(), pl.col("market").n_unique().alias("mkts")).sort("y").rows())
