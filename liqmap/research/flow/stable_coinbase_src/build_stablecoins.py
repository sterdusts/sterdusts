"""Build /home/user/data/flow/stablecoins_daily.parquet from DefiLlama stablecoins API JSON (downloaded to RAW dir).
Inputs (RAW): charts_all.json (stablecoincharts/all), chart_all_sc<id>.json (stablecoincharts/all?stablecoin=<id>),
              stablecoin_1.json / stablecoin_2.json (stablecoin/<id>, for per-chain balances).
Timing (verified against on-chain USDC totalSupply via archive RPC, see llama_timing_probe*.py):
  DefiLlama value labelled date D (timestamp D 00:00 UTC) = snapshot taken at the first run after D 00:00,
  empirically between D-1 23:55 and D 02:45 UTC (15 probe dates) -> state at ~START of day D; column known_at = D 06:00 (conservative)."""
import json, sys, datetime as dt
import polars as pl

RAW = sys.argv[1]
OUT = sys.argv[2]
LAST_DATE = dt.date.fromisoformat(sys.argv[3])      # last DefiLlama date label to keep
FIRST_DATE = dt.date(2020, 1, 1)
CHAINS = {"Ethereum": "eth", "Tron": "tron", "Solana": "sol", "BSC": "bsc", "Arbitrum": "arb"}
EXTRA = {4: "busd", 119: "fdusd", 7: "tusd", 5: "dai", 209: "usds", 146: "usde", 3: "ustc"}


def d_of(t):
    return dt.datetime.fromtimestamp(int(t), dt.timezone.utc).date()


def check_midnight(ts):
    bad = [t for t in ts if int(t) % 86400 != 0]
    assert not bad, f"non-midnight timestamps: {bad[:5]}"


# ---- total (all USD-pegged, and all peg types) ----
allc = json.load(open(f"{RAW}/charts_all.json"))
check_midnight([x["date"] for x in allc])
tot = pl.DataFrame({
    "date": [d_of(x["date"]) for x in allc],
    "total_usd": [float(x["totalCirculatingUSD"].get("peggedUSD") or 0) for x in allc],
    "total_usd_units": [float(x["totalCirculating"].get("peggedUSD") or 0) for x in allc],
    "total_allpeg_usd": [float(sum(v or 0 for v in x["totalCirculatingUSD"].values())) for x in allc],
})

# ---- per-asset totals ----
frames = [tot]
for i, name in [(1, "usdt"), (2, "usdc")] + list(EXTRA.items()):
    a = json.load(open(f"{RAW}/chart_all_sc{i}.json"))
    check_midnight([x["date"] for x in a])
    cols = {"date": [d_of(x["date"]) for x in a],
            f"{name}_total": [float(x["totalCirculating"].get("peggedUSD")) if x["totalCirculating"].get("peggedUSD") is not None else None for x in a]}
    if name in ("usdt", "usdc"):
        cols[f"{name}_total_usd"] = [float(x["totalCirculatingUSD"].get("peggedUSD")) if x["totalCirculatingUSD"].get("peggedUSD") is not None else None for x in a]
    frames.append(pl.DataFrame(cols))

# ---- per-chain USDT / USDC + chain-entry diagnostics ----
entry_rows = []
for i, name in [(1, "usdt"), (2, "usdc")]:
    s = json.load(open(f"{RAW}/stablecoin_{i}.json"))
    for chain, short in CHAINS.items():
        toks = s["chainBalances"].get(chain, {}).get("tokens", [])
        check_midnight([x["date"] for x in toks])
        frames.append(pl.DataFrame({
            "date": [d_of(x["date"]) for x in toks],
            f"{name}_{short}": [float((x.get("circulating") or {}).get("peggedUSD")) if (x.get("circulating") or {}).get("peggedUSD") is not None else None for x in toks],
        }, schema={"date": pl.Date, f"{name}_{short}": pl.Float64}))
    # first tracked date (with circulating > 0) of every chain of this asset -> entry size on that date
    for chain, v in s["chainBalances"].items():
        pts = [x for x in v.get("tokens", []) if ((x.get("circulating") or {}).get("peggedUSD") or 0) > 0]
        if pts:
            entry_rows.append({"asset": name, "date": d_of(pts[0]["date"]), "chain": chain,
                               "entry_usd": float(pts[0]["circulating"]["peggedUSD"])})
entry = (pl.DataFrame(entry_rows).group_by(["asset", "date"])
         .agg(pl.col("entry_usd").sum(), pl.col("chain").sort_by("entry_usd", descending=True).str.join(",").alias("chains"))
         .pivot(on="asset", index="date", values=["entry_usd", "chains"]))
entry = entry.rename({c: (c.replace("entry_usd_", "") + "_chain_entry_usd") if c.startswith("entry_usd_") else
                      (c.replace("chains_", "") + "_chain_entry_names") if c.startswith("chains_") else c for c in entry.columns})

# ---- assemble on a full daily calendar ----
cal = pl.DataFrame({"date": pl.date_range(FIRST_DATE, LAST_DATE, "1d", eager=True)})
df = cal
for f in frames + [entry]:
    f = f.unique("date", keep="last")
    df = df.join(f, on="date", how="left")
for a in ("usdt", "usdc"):
    for c in (f"{a}_chain_entry_usd",):
        df = df.with_columns(pl.col(c).fill_null(0.0))

# ---- simple same-day-computable changes (need D and D-1 only) ----
for c in ("total_usd", "usdt_total", "usdc_total"):
    df = df.with_columns((pl.col(c) - pl.col(c).shift(1)).alias(f"{c}_d1"))

# ---- QA flags ----
# qa_vshape_total_LOOKAHEAD: one-day dip/spike in total_usd (>1%) that reverts on D+1.
#   USES D+1 -> for data diagnosis only, NOT usable as a real-time filter.
r0 = pl.col("total_usd") / pl.col("total_usd").shift(1) - 1
r1 = pl.col("total_usd").shift(-1) / pl.col("total_usd") - 1
df = df.with_columns(((r0.abs() > 0.01) & (r1.abs() > 0.01) & ((r0 * r1) < 0) &
                      ((r0 + r1).abs() < 0.5 * r0.abs())).fill_null(False).alias("qa_vshape_total_LOOKAHEAD"))

# ---- timing / regime columns ----
df = df.with_columns(
    pl.col("date").cast(pl.Datetime("us")).dt.replace_time_zone("UTC").alias("snapshot_ts_nominal"),
    (pl.col("date").cast(pl.Datetime("us")).dt.replace_time_zone("UTC") + pl.duration(hours=6)).alias("known_at"),
    (pl.col("date") < dt.date(2022, 5, 12)).alias("era_pre_20220512"),
    (pl.col("date") >= dt.date(2026, 4, 1)).alias("in_holdout"),
)
lead = ["date", "snapshot_ts_nominal", "known_at", "in_holdout", "era_pre_20220512"]
df = df.select(lead + [c for c in df.columns if c not in lead]).sort("date")
df.write_parquet(OUT, compression="zstd")
print(df.shape)
print(df.select(pl.all().null_count()).transpose(include_header=True).filter(pl.col("column_0") > 0))
