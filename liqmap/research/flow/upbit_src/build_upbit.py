"""Build Upbit (Korea) flow datasets.
Usage: python3 -I build_upbit.py <raw_dir> <listing_hour_json> <kl1d.parquet> <out_dir>
Outputs (never overwrites an existing file):
  upbit_daily.parquet, kimchi.parquet, upbit_listings.parquet, upbit_perp_map.parquet, upbit_perp_daily.parquet
"""
import glob, io, json, math, os, re, sys, zipfile, datetime as dt
import polars as pl

RAW, LH, KL1D, OUT = sys.argv[1:5]
END_DATE = dt.date(2026, 9, 30)          # last UTC day kept in the daily outputs
UTC = "UTC"
STABLE = {"USDT", "USDC", "USDE", "USDS", "USD1", "PYUSD", "RLUSD", "USDG", "EURC", "JPYC", "XAUT", "DAI", "TUSD", "BUSD", "FDUSD"}

# ------------------------------------------------------------------ 1. Upbit daily candles
recs, meta = [], {}
for fn in sorted(glob.glob(os.path.join(RAW, "days", "*.json"))):
    j = json.load(open(fn))
    meta[j["market"]] = {"status": j["status"], "fetched_at_utc": j["fetched_at_utc"]}
    recs.extend(j["rows"])
raw = pl.DataFrame(recs, infer_schema_length=None)
assert raw.select(pl.col("candle_date_time_utc").str.slice(10).unique()).to_series().to_list() == ["T00:00:00"], "daily cut not 00:00 UTC"
assert raw.select(pl.col("candle_date_time_kst").str.slice(10).unique()).to_series().to_list() == ["T09:00:00"], "daily cut not 09:00 KST"
f64 = pl.Float64
up = raw.select(
    pl.col("market"),
    pl.col("market").str.split("-").list.get(0).alias("quote"),
    pl.col("market").str.split("-").list.get(1).alias("base"),
    pl.col("candle_date_time_utc").str.to_datetime("%Y-%m-%dT%H:%M:%S", time_unit="ms").dt.replace_time_zone(UTC).alias("candle_date_time_utc"),
    pl.col("candle_date_time_kst").str.to_datetime("%Y-%m-%dT%H:%M:%S", time_unit="ms").alias("candle_date_time_kst"),
    pl.col("opening_price").cast(f64).alias("open"),
    pl.col("high_price").cast(f64).alias("high"),
    pl.col("low_price").cast(f64).alias("low"),
    pl.col("trade_price").cast(f64).alias("close"),
    pl.col("prev_closing_price").cast(f64).alias("prev_close_api"),
    pl.col("candle_acc_trade_price").cast(f64).alias("acc_trade_price"),
    pl.col("candle_acc_trade_volume").cast(f64).alias("acc_trade_volume"),
    pl.from_epoch(pl.col("timestamp").cast(pl.Int64), time_unit="ms").dt.replace_time_zone(UTC).alias("last_trade_ts_utc"),
).with_columns(pl.col("candle_date_time_utc").dt.date().alias("date"))
dup = up.group_by("market", "date").len().filter(pl.col("len") > 1).height
assert dup == 0, f"duplicate rows {dup}"
up = up.sort("market", "date")
# sanity: last trade must fall inside the candle window
bad_ts = up.filter((pl.col("last_trade_ts_utc") < pl.col("candle_date_time_utc")) |
                   (pl.col("last_trade_ts_utc") >= pl.col("candle_date_time_utc") + pl.duration(days=1))).height
print("last_trade_ts outside candle window:", bad_ts)

# first-bar / gap info computed on the full history (incl. Oct-2026 tail) before trimming
hist = up.filter(pl.col("quote") == "KRW").with_columns(
    (pl.col("date") - pl.col("date").shift(1).over("market")).dt.total_days().alias("days_since_prev_bar"),
    (pl.col("close") / pl.col("close").shift(1).over("market")).log().alias("_lr"),
)

# ------------------------------------------------------------------ 2. FX (ECB reference, FRED H.10)
with zipfile.ZipFile(os.path.join(RAW, "eurofxref-hist.zip")) as z:
    ecb = pl.read_csv(io.BytesIO(z.read("eurofxref-hist.csv")), null_values=["N/A"], infer_schema_length=0)
ecb = ecb.select(pl.col("Date").str.to_date().alias("ecb_date"),
                 (pl.col("KRW").cast(f64) / pl.col("USD").cast(f64)).alias("usdkrw_ecb")).drop_nulls().sort("ecb_date")
fred = pl.read_csv(os.path.join(RAW, "DEXKOUS.csv"), infer_schema_length=0).select(
    pl.col("observation_date").str.to_date().alias("fred_date"),
    pl.col("DEXKOUS").cast(f64, strict=False).alias("usdkrw_fred")).drop_nulls().sort("fred_date")
cal = pl.DataFrame({"date": pl.date_range(dt.date(2017, 9, 1), dt.date(2026, 10, 9), "1d", eager=True)})
fx = (cal.join_asof(ecb, left_on="date", right_on="ecb_date", strategy="backward")
         .join_asof(fred, left_on="date", right_on="fred_date", strategy="backward")
         .with_columns((pl.col("date") - pl.col("ecb_date")).dt.total_days().alias("fx_age_days")))

# ------------------------------------------------------------------ 3. upbit_daily.parquet (KRW markets)
daily = (hist.filter(pl.col("date") <= END_DATE)
         .join(fx.select("date", "usdkrw_ecb", "ecb_date", "fx_age_days"), on="date", how="left")
         .with_columns(
             (pl.col("candle_date_time_utc") + pl.duration(days=1)).alias("available_at"),
             ((pl.col("candle_date_time_utc") + pl.duration(days=1) - pl.col("last_trade_ts_utc")).dt.total_milliseconds() / 60000.0).alias("close_staleness_min"),
             pl.col("base").is_in(list(STABLE)).alias("is_stable"),
             pl.col("date").min().over("market").alias("first_bar_date"),
         )
         .with_columns(
             (pl.col("date") - pl.col("first_bar_date")).dt.total_days().alias("age_days"),
             (pl.col("acc_trade_price") / pl.col("usdkrw_ecb")).alias("acc_trade_price_usd_ecb"),
             (pl.col("close") / pl.col("usdkrw_ecb")).alias("close_usd_ecb"),
         ))
# KRW-USDT close for USDT conversion (null before KRW-USDT existed)
usdt_px = daily.filter(pl.col("market") == "KRW-USDT").select("date", pl.col("close").alias("krw_usdt_close"))
daily = daily.join(usdt_px, on="date", how="left").with_columns(
    (pl.col("acc_trade_price") / pl.col("krw_usdt_close")).alias("acc_trade_price_usdt_upbit"))
daily = daily.select(
    "date", "market", "base", "candle_date_time_utc", "candle_date_time_kst", "available_at",
    pl.col("open").alias("open_krw"), pl.col("high").alias("high_krw"), pl.col("low").alias("low_krw"),
    pl.col("close").alias("close_krw"), pl.col("prev_close_api").alias("prev_close_krw_api"),
    pl.col("acc_trade_price").alias("acc_trade_price_krw"), "acc_trade_volume",
    "last_trade_ts_utc", "close_staleness_min", "days_since_prev_bar", "first_bar_date", "age_days", "is_stable",
    "usdkrw_ecb", "ecb_date", "fx_age_days", "close_usd_ecb", "acc_trade_price_usd_ecb",
    "krw_usdt_close", "acc_trade_price_usdt_upbit",
).sort("date", "market")

# ------------------------------------------------------------------ 4. Binance spot BTCUSDT 1h -> price at UTC 00:00
frames = []
for fn in sorted(glob.glob(os.path.join(RAW, "bn1h", "*.zip"))):
    with zipfile.ZipFile(fn) as z:
        for nm in z.namelist():
            cols = ["ot", "o", "h", "l", "c", "v", "ct", "qv", "n", "tbv", "tbqv", "ig"]
            frames.append(pl.read_csv(io.BytesIO(z.read(nm)), has_header=False, new_columns=cols, infer_schema_length=0).select(
                pl.col("ot").cast(pl.Int64), pl.col("c").cast(f64), pl.col("ct").cast(pl.Int64), pl.col("qv").cast(f64)))
bn = pl.concat(frames).with_columns(
    pl.when(pl.col("ot") > 10**14).then(pl.col("ot") // 1000).otherwise(pl.col("ot")).alias("ot"))
bn = bn.with_columns(pl.from_epoch("ot", time_unit="ms").dt.replace_time_zone(UTC).alias("open_time")).unique("open_time").sort("open_time")
print("binance 1h bars:", bn.height, bn["open_time"].min(), bn["open_time"].max())
# bar opening at d 23:00 closes at d+1 00:00 = same instant as Upbit daily close of day d
bn24 = bn.filter(pl.col("open_time").dt.hour() == 23).select(
    pl.col("open_time").dt.date().alias("date"), pl.col("c").alias("btcusdt_bn_close"),
    pl.col("open_time").alias("bn_bar_open_utc"))

# ------------------------------------------------------------------ 5. kimchi.parquet (one row per UTC day)
def mk(market, prefix):
    return up.filter(pl.col("market") == market).select(
        "date", pl.col("close").alias(f"{prefix}_close"), pl.col("acc_trade_price").alias(f"{prefix}_turnover"),
        pl.col("last_trade_ts_utc").alias(f"{prefix}_last_trade_ts_utc"))

btc, usdt, ub = mk("KRW-BTC", "krw_btc"), mk("KRW-USDT", "krw_usdt"), mk("USDT-BTC", "usdt_btc_upbit")
agg = daily.group_by("date").agg(
    pl.col("acc_trade_price_krw").sum().alias("upbit_krw_turnover_all"),
    pl.col("acc_trade_price_krw").filter(~pl.col("is_stable")).sum().alias("upbit_krw_turnover_nonstable"),
    pl.col("acc_trade_price_krw").filter(~pl.col("is_stable") & ~pl.col("base").is_in(["BTC", "ETH"])).sum().alias("upbit_krw_turnover_alts"),
    pl.col("acc_trade_price_krw").filter(pl.col("is_stable")).sum().alias("upbit_krw_turnover_stables"),
    (pl.col("acc_trade_price_krw") > 0).sum().alias("n_krw_markets_traded"),
)
k = (pl.DataFrame({"date": pl.date_range(dt.date(2021, 1, 1), END_DATE, "1d", eager=True)})
     .join(btc, on="date", how="left").join(usdt, on="date", how="left").join(ub, on="date", how="left")
     .join(bn24, on="date", how="left")
     .join(fx.select("date", "usdkrw_ecb", "ecb_date", "fx_age_days", "usdkrw_fred", "fred_date"), on="date", how="left")
     .join(agg, on="date", how="left")
     .with_columns(
         (pl.col("date").cast(pl.Datetime("ms")).dt.replace_time_zone(UTC) + pl.duration(days=1)).alias("snapshot_utc"),
         (pl.col("krw_btc_close") / (pl.col("krw_usdt_close") * pl.col("btcusdt_bn_close")) - 1).alias("prem_btc_vs_krwusdt"),
         (pl.col("krw_btc_close") / (pl.col("usdkrw_ecb") * pl.col("btcusdt_bn_close")) - 1).alias("kimchi_fx_ecb"),
         (pl.col("krw_btc_close") / (pl.col("usdkrw_fred") * pl.col("btcusdt_bn_close")) - 1).alias("kimchi_fx_fred"),
         (pl.col("krw_usdt_close") / pl.col("usdkrw_ecb") - 1).alias("usdt_prem_fx_ecb"),
         (pl.col("krw_btc_close") / pl.col("usdt_btc_upbit_close")).alias("usdtkrw_implied_upbit_cross"),
         (pl.col("usdt_btc_upbit_close") / pl.col("btcusdt_bn_close") - 1).alias("prem_upbit_usdtmkt_vs_bn"),
         ((pl.col("date").cast(pl.Datetime("ms")).dt.replace_time_zone(UTC) + pl.duration(days=1) - pl.col("krw_btc_last_trade_ts_utc")).dt.total_milliseconds() / 60000.0).alias("krw_btc_staleness_min"),
         ((pl.col("date").cast(pl.Datetime("ms")).dt.replace_time_zone(UTC) + pl.duration(days=1) - pl.col("krw_usdt_last_trade_ts_utc")).dt.total_milliseconds() / 60000.0).alias("krw_usdt_staleness_min"),
         ((pl.col("date").cast(pl.Datetime("ms")).dt.replace_time_zone(UTC) + pl.duration(days=1) - pl.col("usdt_btc_upbit_last_trade_ts_utc")).dt.total_milliseconds() / 60000.0).alias("usdt_btc_upbit_staleness_min"),
     )
     .with_columns((pl.col("upbit_krw_turnover_all") / pl.col("usdkrw_ecb")).alias("upbit_turnover_all_usd_ecb"),
                   (pl.col("upbit_krw_turnover_alts") / pl.col("usdkrw_ecb")).alias("upbit_turnover_alts_usd_ecb"),
                   pl.col("snapshot_utc").alias("available_at"))
     .drop("krw_btc_last_trade_ts_utc", "krw_usdt_last_trade_ts_utc", "usdt_btc_upbit_last_trade_ts_utc"))
k = k.select("date", "snapshot_utc", "available_at", "krw_btc_close", "krw_usdt_close", "btcusdt_bn_close", "bn_bar_open_utc",
             "usdkrw_ecb", "ecb_date", "fx_age_days", "usdkrw_fred", "fred_date",
             "prem_btc_vs_krwusdt", "kimchi_fx_ecb", "usdt_prem_fx_ecb", "kimchi_fx_fred",
             "usdt_btc_upbit_close", "usdtkrw_implied_upbit_cross", "prem_upbit_usdtmkt_vs_bn",
             "krw_btc_staleness_min", "krw_usdt_staleness_min", "usdt_btc_upbit_staleness_min",
             "krw_btc_turnover", "krw_usdt_turnover", "usdt_btc_upbit_turnover",
             "upbit_krw_turnover_all", "upbit_krw_turnover_nonstable", "upbit_krw_turnover_alts", "upbit_krw_turnover_stables",
             "upbit_turnover_all_usd_ecb", "upbit_turnover_alts_usd_ecb", "n_krw_markets_traded")

# ------------------------------------------------------------------ 6. upbit_listings.parquet
ma = pl.DataFrame([{"market": x["market"], "korean_name": x.get("korean_name"), "english_name": x.get("english_name"),
                    "snap_warning": (x.get("market_event") or {}).get("warning"),
                    **{"snap_caution_" + c.lower(): v for c, v in ((x.get("market_event") or {}).get("caution") or {}).items()}}
                   for x in json.load(open(os.path.join(RAW, "market_all.json"))) if x["market"].startswith("KRW-")])
snap_ts = dt.datetime.fromtimestamp(os.path.getmtime(os.path.join(RAW, "market_all.json")), dt.timezone.utc).replace(microsecond=0)
lh = json.load(open(LH))
lhd = pl.DataFrame([{"market": m, "first_hour_utc": (v["hours"][0] if v["hours"] else None), "n_traded_hours_first_day": len(v["hours"])}
                    for m, v in lh["listing"].items()]).with_columns(
    pl.col("first_hour_utc").str.to_datetime("%Y-%m-%dT%H:%M:%S", time_unit="ms").dt.replace_time_zone(UTC))
UPBIT_FIRST = hist["date"].min()
gaps = (hist.filter(pl.col("days_since_prev_bar") > 1)
        .group_by("market").agg(pl.len().alias("n_gaps"), (pl.col("days_since_prev_bar") - 1).sum().alias("n_gap_days"),
                                (pl.col("days_since_prev_bar") - 1).max().alias("max_gap_days"),
                                pl.format("{}(+{}d)", pl.col("date"), pl.col("days_since_prev_bar") - 1).filter(pl.col("days_since_prev_bar") > 7).str.join(";").alias("gaps_gt7d_resume_dates")))
jumps = (hist.filter(pl.col("_lr").abs() > math.log(5))
         .group_by("market").agg(pl.format("{}({})", pl.col("date"), pl.col("_lr").exp().round(4)).str.join(";").alias("price_jumps_gt5x")))
li = (hist.group_by("market").agg(
          pl.col("base").first(), pl.col("date").min().alias("first_bar_date"), pl.col("date").max().alias("last_bar_date"),
          pl.len().alias("n_bars"), pl.col("acc_trade_price").sort_by("date").first().alias("first_day_turnover_krw"),
          pl.col("close").sort_by("date").first().alias("first_day_close_krw"),
          pl.col("acc_trade_price").sort_by("date").slice(1, 7).mean().alias("days2to8_mean_turnover_krw"))
      .join(ma, on="market", how="left").join(lhd, on="market", how="left")
      .join(gaps, on="market", how="left").join(jumps, on="market", how="left")
      .with_columns(
          (pl.col("first_bar_date") <= UPBIT_FIRST + dt.timedelta(days=0)).alias("listing_left_censored"),
          pl.col("base").is_in(list(STABLE)).alias("is_stable"),
          pl.lit(snap_ts).dt.replace_time_zone(UTC).alias("market_all_snapshot_utc"),
          pl.lit(True).alias("survivor_only"),
          pl.col("n_gaps").fill_null(0), pl.col("n_gap_days").fill_null(0), pl.col("max_gap_days").fill_null(0),
      ).sort("first_bar_date", "market"))

# ------------------------------------------------------------------ 7. perp mapping (name match + known renames), validated by price
kl = (pl.scan_parquet(KL1D).select("open_time", "symbol", "close", "quote_volume").collect()
      .with_columns(pl.from_epoch("open_time", time_unit="ms").dt.date().alias("date"))
      .filter(pl.col("quote_volume") > 0).select("date", "symbol", pl.col("close").alias("perp_close"), pl.col("quote_volume").alias("perp_quote_volume")))
syms = kl["symbol"].unique().sort().to_list()
def strip(sym):
    b = sym[:-4] if sym.endswith("USDT") else sym
    m = re.match(r"^(1000000|100000|10000|1000|1M)(.+)$", b)
    if m and not b.startswith("1INCH"):
        return b, m.group(2), (1e6 if m.group(1) == "1M" else float(m.group(1)))
    return b, b, 1.0
perp_tbl = pl.DataFrame([dict(zip(["perp_symbol", "perp_base", "perp_base_stripped", "mult_prefix"], (s, *strip(s)))) for s in syms])
# known renames / Upbit "<X>2" collision suffix (candidates only; each is price-validated below)
ALIAS = {"BTT": ["BTTC"], "POL": ["MATIC"], "A": ["EOS"], "RENDER": ["RNDR"], "KAIA": ["KLAY"], "MANTRA": ["OM"],
         "SKY": ["MKR"], "TOKAMAK": ["TON"]}
cands = []
for b in li["base"].to_list():
    names = [(b, "name")] + [(a, "alias_rename") for a in ALIAS.get(b, [])]
    if re.match(r"^[A-Z]+2$", b):
        names.append((b[:-1], "alias_suffix2"))
    for nm, how in names:
        for r in perp_tbl.filter(pl.col("perp_base_stripped") == nm).iter_rows(named=True):
            cands.append({"base": b, "market": "KRW-" + b, "perp_symbol": r["perp_symbol"], "mult_prefix": r["mult_prefix"], "method": how})
cand = pl.DataFrame(cands).unique(["market", "perp_symbol"], keep="first", maintain_order=True)
btcref = daily.filter(pl.col("market") == "KRW-BTC").select("date", pl.col("close_krw").alias("btc_krw")).join(
    kl.filter(pl.col("symbol") == "BTCUSDT").select("date", pl.col("perp_close").alias("btc_perp")), on="date")
pairs = (cand.join(daily.select("date", "market", "close_krw", "usdkrw_ecb", "acc_trade_price_usd_ecb", "close_staleness_min"), on="market")
         .join(kl.rename({"symbol": "perp_symbol"}), on=["date", "perp_symbol"])
         .join(btcref, on="date", how="left")
         .with_columns((pl.col("close_krw") / (pl.col("usdkrw_ecb") * pl.col("perp_close") / pl.col("mult_prefix"))).log().alias("lr_fx"),
                       (pl.col("btc_krw") / (pl.col("usdkrw_ecb") * pl.col("btc_perp"))).log().alias("lr_btc"))
         .with_columns((pl.col("lr_fx") - pl.col("lr_btc")).alias("lr_rel")))
# detect residual power-of-10 unit differences (e.g. BTT redenomination back-adjustment)
stat = pairs.group_by("market", "perp_symbol").agg(pl.col("lr_rel").median().alias("med_lr_rel_raw"))
stat = stat.with_columns((pl.col("med_lr_rel_raw") / math.log(10)).round(0).alias("pow10"))
stat = stat.with_columns(pl.when((pl.col("med_lr_rel_raw") - pl.col("pow10") * math.log(10)).abs() < 0.15)
                         .then(pl.col("pow10")).otherwise(0.0).alias("pow10"))
pairs = pairs.join(stat.select("market", "perp_symbol", "pow10"), on=["market", "perp_symbol"]).with_columns(
    (pl.col("lr_rel") - pl.col("pow10") * math.log(10)).alias("lr_rel"), (pl.col("lr_fx") - pl.col("pow10") * math.log(10)).alias("lr_fx"))
pstat = pairs.group_by("market", "perp_symbol").agg(
    pl.len().alias("n_overlap_days"), pl.col("date").min().alias("overlap_first"), pl.col("date").max().alias("overlap_last"),
    pl.col("lr_rel").median().alias("med_logprem_rel_btc"), (pl.col("lr_rel").abs() < 0.10).mean().alias("frac_days_rel_within_10pct"),
    pl.col("lr_fx").median().alias("med_logprem_fx"), pl.col("pow10").first())
pmap = (cand.join(pstat, on=["market", "perp_symbol"], how="left")
        .with_columns((pl.col("mult_prefix") * (10.0 ** (-pl.col("pow10").fill_null(0)))).alias("mult_effective"),
                      pl.when(pl.col("n_overlap_days").is_null()).then(pl.lit("no_overlap"))
                      .when((pl.col("n_overlap_days") >= 10) & (pl.col("med_logprem_rel_btc").abs() < 0.10) & (pl.col("frac_days_rel_within_10pct") >= 0.8)).then(pl.lit("ok"))
                      .when(pl.col("n_overlap_days") < 10).then(pl.lit("too_short"))
                      .when(pl.col("frac_days_rel_within_10pct") >= 0.3).then(pl.lit("partial"))
                      .otherwise(pl.lit("price_mismatch")).alias("status"))
        .join(li.select("market", "english_name", "first_bar_date"), on="market", how="left")
        .sort("market", "perp_symbol"))
# listing-level flag: an alias (renamed predecessor) perp price-matches the Upbit history -> first_bar_date may be the predecessor's listing
pred = pmap.filter((pl.col("method") == "alias_rename") & (pl.col("status") == "ok")).group_by("market").agg(pl.col("perp_symbol").str.join(",").alias("predecessor_perp_ok"))
li = li.join(pred, on="market", how="left")
# add unmapped Upbit markets as explicit rows
unm = li.filter(~pl.col("market").is_in(cand["market"].unique().to_list())).select("market", "base", "english_name", "first_bar_date").with_columns(pl.lit("no_perp_name_match").alias("status"))
pmap = pl.concat([pmap, unm], how="diagonal_relaxed").sort("market", "perp_symbol", nulls_last=True)

good = pmap.filter(pl.col("status").is_in(["ok", "partial"])).select("market", "perp_symbol", "status", "mult_effective")
pdaily = (pairs.join(good, on=["market", "perp_symbol"])
          .select("date", "market", "perp_symbol", pl.col("status").alias("map_status"), "mult_effective", "close_krw", "close_staleness_min", "usdkrw_ecb",
                  pl.col("perp_close"), (pl.col("lr_fx").exp() - 1).alias("prem_fx_ecb"), (pl.col("lr_rel").exp() - 1).alias("prem_rel_btc"),
                  (pl.col("lr_rel").abs() < 0.10).alias("px_match_10pct"),
                  pl.col("acc_trade_price_usd_ecb").alias("upbit_turnover_usd_ecb"), "perp_quote_volume")
          .with_columns((pl.col("upbit_turnover_usd_ecb") / pl.col("perp_quote_volume")).alias("upbit_to_perp_turnover"),
                        (pl.col("date").cast(pl.Datetime("ms")).dt.replace_time_zone(UTC) + pl.duration(days=1)).alias("available_at"))
          .sort("date", "market", "perp_symbol"))
assert pdaily.group_by("date", "market", "perp_symbol").len()["len"].max() == 1
assert daily.group_by("date", "market").len()["len"].max() == 1

# ------------------------------------------------------------------ write (refuse to overwrite)
outs = {"upbit_daily.parquet": daily, "kimchi.parquet": k, "upbit_listings.parquet": li,
        "upbit_perp_map.parquet": pmap, "upbit_perp_daily.parquet": pdaily}
for nm, df in outs.items():
    p = os.path.join(OUT, nm)
    assert not os.path.exists(p), f"refusing to overwrite {p}"
for nm, df in outs.items():
    df.write_parquet(os.path.join(OUT, nm), compression="zstd")
    print("wrote", nm, df.shape)
