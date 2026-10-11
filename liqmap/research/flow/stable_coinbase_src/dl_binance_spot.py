"""Download Binance SPOT monthly 1h klines from data.binance.vision (zips parsed in memory, never extracted).
Note: spot archive timestamps switch from milliseconds to MICROseconds starting 2025-01-01 files; normalised here.
Output: <SCR>/bn1h_<SYM>.parquet; ts = UTC open time."""
import asyncio, io, zipfile, sys
import aiohttp
import polars as pl

SCR = "/tmp/claude-0/-home-user-sterdusts/3ad56404-3e60-5e0b-ae02-0725979310e7/scratchpad/flow/stable-coinbase"
BASE = "https://data.binance.vision/data/spot/monthly/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
MONTHS = [f"{y}-{m:02d}" for y in range(2021, 2027) for m in range(1, 13) if (y, m) <= (2026, 9)]


async def fetch(s, sem, url):
    async with sem:
        for i in range(6):
            try:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                    if r.status == 404:
                        return 404, None
                    if r.status == 200:
                        return 200, await r.read()
                    await asyncio.sleep(3 * (i + 1))
            except Exception:
                await asyncio.sleep(2 ** i)
    return "fail", None


def parse(b):
    z = zipfile.ZipFile(io.BytesIO(b))
    txt = z.read(z.namelist()[0]).decode()
    lines = [l for l in txt.splitlines() if l and l[0].isdigit()]   # drop header if any
    df = pl.read_csv(io.StringIO("\n".join(lines)), has_header=False, new_columns=COLS,
                     schema_overrides={c: pl.Float64 for c in COLS if c not in ("open_time", "close_time", "count", "ignore")} | {"open_time": pl.Int64, "close_time": pl.Int64, "count": pl.Int64, "ignore": pl.Utf8})
    # ms -> us normalisation: anything > 1e14 is microseconds
    ot = pl.when(pl.col("open_time") > 10**14).then(pl.col("open_time")).otherwise(pl.col("open_time") * 1000)
    return (df.with_columns(pl.from_epoch(ot, time_unit="us").dt.replace_time_zone("UTC").alias("ts"))
              .drop(["open_time", "close_time", "ignore"]))


async def main(syms):
    sem = asyncio.Semaphore(8)
    async with aiohttp.ClientSession() as s:
        for sym in syms:
            urls = [f"{BASE}/{sym}/1h/{sym}-1h-{m}.zip" for m in MONTHS]
            res = await asyncio.gather(*[fetch(s, sem, u) for u in urls])
            dfs, miss = [], []
            for m, (st, b) in zip(MONTHS, res):
                if st == 200:
                    dfs.append(parse(b))
                else:
                    miss.append((m, st))
            df = pl.concat(dfs).unique("ts").sort("ts") if dfs else None
            if df is not None:
                df.write_parquet(f"{SCR}/bn1h_{sym}.parquet")
            print(sym, "months_ok", len(dfs), "missing", miss, "rows", None if df is None else df.height,
                  None if df is None else (df["ts"].min(), df["ts"].max()), flush=True)

asyncio.run(main(sys.argv[1:]))
