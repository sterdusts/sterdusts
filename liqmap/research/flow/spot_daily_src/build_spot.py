"""Stage 5: raw spot bars -> /home/user/data/flow/{spot_daily, spot_daily_xq, spot_daily_base, spot_daily_perp, spot_map}.parquet

usage: python -I build_spot.py <workdir> <outdir>
No signal/forward-return statistics are computed here.
"""
import sys, os
import polars as pl

W, OUT = sys.argv[1], sys.argv[2]
Q5 = ["USDT", "FDUSD", "USDC", "TUSD", "BUSD"]
QX = ["USD1", "U"]
CONV = {"USDC": "USDCUSDT", "FDUSD": "FDUSDUSDT", "TUSD": "TUSDUSDT", "BUSD": "BUSDUSDT",
        "USD1": "USD1USDT", "U": "UUSDT"}
PX_TOL = 0.10  # |log(perp_close / (spot_close_usdt*mult))| tolerance for a day to count as the same asset

for f in ["spot_daily", "spot_daily_xq", "spot_daily_base", "spot_daily_perp", "spot_map"]:
    assert not os.path.exists(f"{OUT}/{f}.parquet"), f"refuse to overwrite {f}"

# ---------------- long table ----------------
r = pl.read_parquet(f"{W}/spot_raw.parquet")
us = lambda c: pl.when(pl.col(c) > 10**14).then(pl.col(c) // 1000).otherwise(pl.col(c))  # 2025-01+ files are microseconds
r = (r.with_columns(us("open_time").alias("ot"), us("close_time").alias("ct"))
       .unique(["spot_symbol", "ot"], keep="first")              # 6 exact duplicate rows in the archive
       .with_columns(pl.from_epoch("ot", time_unit="ms").dt.date().alias("date"),
                     ((pl.col("ct") - pl.col("ot") + 1) / 3.6e6).round(3).alias("bar_hours")))
assert r.filter(pl.from_epoch("ot", time_unit="ms").dt.time() != pl.time(0)).height == 0

conv = (r.filter(pl.col("spot_symbol").is_in(list(CONV.values())))
          .select("date", pl.col("spot_symbol").replace_strict({v: k for k, v in CONV.items()}).alias("quote"),
                  pl.col("close").alias("quote_usdt_raw")))

r = r.join(conv, on=["date", "quote"], how="left")
# fill: USDT quote = 1; otherwise same-day close of quote/USDT; if missing, last known within 7 days; else 1.0 flagged
r = r.sort("spot_symbol", "date").with_columns(
    pl.when(pl.col("quote") == "USDT").then(pl.lit(1.0)).otherwise(pl.col("quote_usdt_raw")).alias("q1"))
fill = (conv.sort("quote", "date").rename({"date": "d_conv", "quote_usdt_raw": "q_ffill"}))
r = r.sort("date").join_asof(fill.sort("d_conv"), left_on="date", right_on="d_conv", by="quote", strategy="backward")
r = r.with_columns(
    pl.when(pl.col("q1").is_not_null()).then(pl.lit("same_day"))
      .when(pl.col("q_ffill").is_not_null() & ((pl.col("date") - pl.col("d_conv")).dt.total_days() <= 7)).then(pl.lit("ffill"))
      .otherwise(pl.lit("assumed_1")).alias("quote_usdt_src"),
    pl.coalesce(pl.col("q1"),
                pl.when((pl.col("date") - pl.col("d_conv")).dt.total_days() <= 7).then(pl.col("q_ffill")),
                pl.lit(1.0)).alias("quote_usdt"))
first_bar = r.group_by("spot_symbol").agg(pl.col("date").min().alias("first_date"))
r = r.join(first_bar, on="spot_symbol").with_columns(
    ((pl.col("date") == pl.col("first_date")) & (pl.col("date") > pl.date(2021, 1, 1))).alias("is_listing_day"))

long_cols = ["date", "base", "quote", "spot_symbol", "open", "high", "low", "close", "volume",
             "quote_volume", "taker_buy_volume", "taker_buy_quote_volume", "count", "bar_hours",
             "is_listing_day", "quote_usdt", "quote_usdt_src", "quote_volume_usdt",
             "taker_buy_quote_volume_usdt", "available_at"]
r = r.with_columns(
    (pl.col("quote_volume") * pl.col("quote_usdt")).alias("quote_volume_usdt"),
    (pl.col("taker_buy_quote_volume") * pl.col("quote_usdt")).alias("taker_buy_quote_volume_usdt"),
    (pl.col("date").cast(pl.Datetime("ms", "UTC")) + pl.duration(days=1)).alias("available_at"),
).select(long_cols).sort("base", "quote", "date")
spot = r.filter(pl.col("quote").is_in(Q5))
xq = r.filter(pl.col("quote").is_in(QX))
spot.write_parquet(f"{OUT}/spot_daily.parquet")
xq.write_parquet(f"{OUT}/spot_daily_xq.parquet")
print("long", spot.height, "xq", xq.height)

# ---------------- base aggregate (5 quotes) ----------------
per_q = (spot.group_by("date", "base").agg(
    [pl.col("quote_volume_usdt").filter(pl.col("quote") == q).sum().alias(f"qv_usdt_{q}") for q in Q5]
    + [pl.col("taker_buy_quote_volume_usdt").filter(pl.col("quote") == q).sum().alias(f"tbqv_usdt_{q}") for q in Q5]
    + [pl.col("quote_volume_usdt").sum().alias("qv_usdt"),
       pl.col("taker_buy_quote_volume_usdt").sum().alias("tbqv_usdt"),
       pl.col("count").sum().alias("trades"),
       (pl.col("quote_volume") > 0).sum().cast(pl.Int8).alias("n_quotes_traded"),
       pl.col("close").filter(pl.col("quote") == "USDT").first().alias("close_usdt_pair"),
       (pl.col("close") * pl.col("quote_usdt")).sort_by("quote_volume_usdt", descending=True).first().alias("close_maxvol"),
       pl.col("quote").sort_by("quote_volume_usdt", descending=True).first().alias("maxvol_quote"),
       pl.col("bar_hours").min().alias("bar_hours_min"),
       pl.col("is_listing_day").any().alias("any_pair_listing_day"),
       (pl.col("quote_usdt_src") == "assumed_1").any().alias("any_quote_assumed_1"),
       ]))
xq_agg = xq.group_by("date", "base").agg(pl.col("quote_volume_usdt").sum().alias("xq_qv_usdt"),
                                         pl.col("taker_buy_quote_volume_usdt").sum().alias("xq_tbqv_usdt"))
base = (per_q.join(xq_agg, on=["date", "base"], how="left")
        .with_columns(
            pl.coalesce("close_usdt_pair", "close_maxvol").alias("close_usdt"),
            pl.when(pl.col("close_usdt_pair").is_not_null()).then(pl.lit("USDT")).otherwise(pl.col("maxvol_quote")).alias("close_src"),
            pl.when(pl.col("qv_usdt") > 0).then(pl.col("tbqv_usdt") / pl.col("qv_usdt")).alias("tb_ratio"),
            (pl.col("date").cast(pl.Datetime("ms", "UTC")) + pl.duration(days=1)).alias("available_at"))
        .sort("base", "date")
        .with_columns(pl.col("date").diff().over("base").dt.total_days().alias("days_since_prev"))
        .drop("close_usdt_pair", "close_maxvol", "maxvol_quote"))
base = base.select(["date", "base", "qv_usdt", "tbqv_usdt", "tb_ratio", "trades", "n_quotes_traded", "close_usdt",
                    "close_src"] + [f"qv_usdt_{q}" for q in Q5] + [f"tbqv_usdt_{q}" for q in Q5]
                   + ["xq_qv_usdt", "xq_tbqv_usdt", "bar_hours_min", "any_pair_listing_day", "any_quote_assumed_1",
                      "days_since_prev", "available_at"])
base.write_parquet(f"{OUT}/spot_daily_base.parquet")
print("base", base.height)

# ---------------- perp-keyed panel ----------------
cands = pl.read_parquet(f"{W}/cands.parquet")
kl = (pl.read_parquet("/home/user/data/kl1d.parquet", columns=["open_time", "symbol", "close", "quote_volume"])
        .with_columns(pl.from_epoch("open_time", time_unit="ms").dt.date().alias("date"))
        .rename({"symbol": "perp_symbol", "close": "perp_close", "quote_volume": "perp_qv"}))
live = kl.filter(pl.col("perp_qv") > 0).select("date", "perp_symbol", "perp_close")  # kl1d keeps flat qv=0 rows after delisting
traded = base.filter(pl.col("qv_usdt") > 0).select("date", "base")
opts = (live.join(cands, on="perp_symbol", how="inner")
            .join(traded.rename({"base": "spot_base"}).with_columns(pl.lit(True).alias("_t")), on=["date", "spot_base"], how="left")
            .filter(pl.col("_t").fill_null(False))
            .sort("perp_symbol", "date", "priority")
            .unique(["perp_symbol", "date"], keep="first").drop("_t"))
perp = (live.join(opts.drop("perp_close"), on=["perp_symbol", "date"], how="left")
            .join(base.rename({"base": "spot_base"}).drop("available_at"), on=["date", "spot_base"], how="left")
            .with_columns((pl.col("close_usdt") * pl.col("mult")).alias("spot_close_perp_units"))
            .with_columns((pl.col("perp_close") / pl.col("spot_close_perp_units")).log().alias("px_logdiff"))
            .with_columns((pl.col("px_logdiff").abs() < PX_TOL).fill_null(False).alias("px_ok"))
            .with_columns((pl.col("date").cast(pl.Datetime("ms", "UTC")) + pl.duration(days=1)).alias("available_at")))
pcols = ["date", "perp_symbol", "spot_base", "method", "mult", "has_spot", "px_ok", "px_logdiff", "perp_close",
         "spot_close_perp_units", "qv_usdt", "tbqv_usdt", "tb_ratio", "trades", "n_quotes_traded"] \
        + [f"qv_usdt_{q}" for q in Q5] + ["xq_qv_usdt", "xq_tbqv_usdt", "bar_hours_min", "any_pair_listing_day",
                                          "available_at"]

# ---------------- map ----------------
NOTES = {
    "LUNAUSDT": "old Terra LUNA; spot LUNA ticker = old LUNA until 2022-05-26 (LUNAUSDT bars end 2022-05-13), then reused by Terra 2.0 from 2022-05-31",
    "LUNA2USDT": "Terra 2.0 = spot LUNA from 2022-05-31 (ticker reused, see LUNAUSDT)",
    "1000LUNCUSDT": "LUNC spot pairs start 2022-05-30 (BUSD) / 2022-09-09 (USDT); old-LUNA history stays under spot LUNA",
    "1000SATSUSDT": "spot itself is 1000SATS", "1000CATUSDT": "spot itself is 1000CAT",
    "1000CHEEMSUSDT": "spot itself is 1000CHEEMS", "1MBABYDOGEUSDT": "spot itself is 1MBABYDOGE",
    "MKRUSDT": "rename MKR->SKY at 1:24000, only used on perp-live days without MKR spot bar",
    "GALUSDT": "rename GAL->G at 1:60 (Galxe->Gravity), only used on perp-live days without GAL spot bar",
    "BTTUSDT": "old BTT; BTT->BTTC redenomination 1:1000",
    "BOTUSDT": "spot BOT bars (2021-01..02) are an unrelated delisted token; no overlap with perp life, unused",
    "DODOXUSDT": "assumed DODO; price-checked", "RAYSOLUSDT": "assumed Raydium RAY; price-checked",
}
listing = set(open(f"{W}/spot_syms.txt").read().split())
sp_rng = base.filter(pl.col("qv_usdt") > 0).group_by("base").agg(pl.col("date").min().alias("spot_first_date"),
                                                                  pl.col("date").max().alias("spot_last_date"))
used = perp.filter(pl.col("spot_base").is_not_null()).group_by("perp_symbol", "spot_base").agg(
    pl.len().alias("days_used"), pl.col("px_ok").sum().alias("days_px_ok"),
    pl.col("px_logdiff").abs().median().alias("px_absdiff_med"))
lv = live.group_by("perp_symbol").agg(pl.col("date").min().alias("perp_first_live"),
                                      pl.col("date").max().alias("perp_last_live"), pl.len().alias("perp_live_days"))
allp = kl.select("perp_symbol").unique()
m = (allp.join(cands, on="perp_symbol", how="left")
         .with_columns(pl.col("perp_symbol").str.slice(0, pl.col("perp_symbol").str.len_chars() - 4).alias("perp_base"),
                       pl.col("method").fill_null("none"))
         .with_columns(pl.col("spot_base").map_elements(lambda b: ",".join(q for q in Q5 if b and b + q in listing),
                                                        return_dtype=pl.String).alias("spot_quotes"))
         .join(sp_rng.rename({"base": "spot_base"}), on="spot_base", how="left")
         .join(lv, on="perp_symbol", how="left")
         .join(used, on=["perp_symbol", "spot_base"], how="left")
         .with_columns(pl.col("days_used").fill_null(0), pl.col("days_px_ok").fill_null(0), pl.col("perp_live_days").fill_null(0))
         .with_columns(
             pl.when(pl.col("spot_base").is_null()).then(pl.lit("no_spot"))
               .when(pl.col("days_used") == 0).then(pl.lit("unused"))
               .when(pl.col("days_px_ok") >= 0.95 * pl.col("days_used")).then(pl.lit("ok"))
               .when(pl.col("days_px_ok") >= 0.5 * pl.col("days_used")).then(pl.lit("partial_px"))
               .otherwise(pl.lit("px_mismatch")).alias("status"),
             pl.col("perp_symbol").replace_strict(NOTES, default=None).alias("note"),
             pl.col("priority").fill_null(0).cast(pl.Int8))
         .select("perp_symbol", "perp_base", "spot_base", "mult", "method", "priority", "spot_quotes",
                 "spot_first_date", "spot_last_date", "perp_first_live", "perp_last_live", "perp_live_days",
                 "days_used", "days_px_ok", "px_absdiff_med", "status", "note")
         .sort("perp_symbol", "priority"))
m.write_parquet(f"{OUT}/spot_map.parquet")
print("map", m.height)

# has_spot: a mapped base with traded bars, and the (perp, spot_base) mapping passed the pair-level price check.
# Day-level px_ok is only a diagnostic: big same-asset dislocations (delisting days, squeezes) are kept.
ok = m.filter(pl.col("status") == "ok").select("perp_symbol", "spot_base", pl.lit(True).alias("_ok"))
perp = (perp.join(ok, on=["perp_symbol", "spot_base"], how="left")
            .with_columns((pl.col("spot_base").is_not_null() & pl.col("_ok").fill_null(False)).alias("has_spot")))
perp = perp.select(pcols).sort("perp_symbol", "date")
perp.write_parquet(f"{OUT}/spot_daily_perp.parquet")
print("perp", perp.height)
