"""Download Binance UM perp 1h klines for the D1' 01:00-entry robustness check.

Spec (D1P-spot-lead-flow formula (5)): futures/um/monthly/klines/{SYM}/1h/{SYM}-1h-{YYYY-MM}.zip,
SYM = every code that entered C_d in the dev period + BTCUSDT; months 2021-12..2026-03 ONLY (no 2026-04+).
Stored as /home/user/data/flow/kl1h_um/{SYM}.parquet (new files only). open_time normalised to ms
(files from 2025-01 use microseconds). Months without any perp trade day for the symbol are skipped (would 404).
"""
import asyncio, io, os, sys, zipfile, json
import aiohttp
import polars as pl

BASE = "https://data.binance.vision/data/futures/um/monthly/klines"
OUT = "/home/user/data/flow/kl1h_um"
RUN = "/home/user/data/flow/D1P-spot-lead-flow_run"
KL_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
CONC = 12
M_LO, M_HI = "2021-12", "2026-03"

os.makedirs(OUT, exist_ok=True)
f = pl.read_parquet(f"{RUN}/d1_features.parquet", columns=["symbol", "in_C"])
syms = sorted(set(f.filter("in_C")["symbol"].unique().to_list()) | {"BTCUSDT"})
pm = (pl.scan_parquet("/home/user/data/flow/perp_daily.parquet").select("date", "symbol", "bar_status")
      .filter((pl.col("date") >= pl.date(2021, 12, 1)) & (pl.col("date") <= pl.date(2026, 3, 31)) &
              pl.col("symbol").is_in(syms) & (pl.col("bar_status") == "trade"))
      .select("symbol", pl.col("date").dt.strftime("%Y-%m").alias("m")).unique().collect())
jobs = {}
for s, m in pm.iter_rows():
    assert M_LO <= m <= M_HI
    jobs.setdefault(s, []).append(m)
print(len(syms), "symbols", sum(len(v) for v in jobs.values()), "symbol-months", flush=True)


async def fetch(session, sem, url, retries=6):
    async with sem:
        for i in range(retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=90)) as r:
                    if r.status == 404:
                        return 404, None
                    if r.status == 200:
                        return 200, await r.read()
                    await asyncio.sleep(3 * (i + 1))
                    continue
            except Exception:
                pass
            await asyncio.sleep(2 ** i)
    return "fail", None


def parse(blob, sym):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:200].decode(errors="ignore").split("\n")[0]
    has_header = not first[:1].isdigit()
    df = pl.read_csv(raw, has_header=has_header, new_columns=None if has_header else KL_COLS, infer_schema_length=10000)
    if has_header and len(df.columns) == len(KL_COLS):
        df.columns = KL_COLS
    df = df.drop("ignore", "close_time").with_columns(
        [pl.col(c).cast(pl.Float64) for c in ["open", "high", "low", "close", "volume", "quote_volume",
                                                "taker_buy_volume", "taker_buy_quote_volume"]]
        + [pl.col("open_time").cast(pl.Int64), pl.col("count").cast(pl.Int64)])
    # microseconds (>=2025-01 files) -> ms by magnitude
    df = df.with_columns(pl.when(pl.col("open_time") > 10**14).then(pl.col("open_time") // 1000)
                         .otherwise(pl.col("open_time")).alias("open_time"))
    return df.with_columns(pl.lit(sym).alias("symbol"))


async def main():
    sem = asyncio.Semaphore(CONC)
    status = {}
    async with aiohttp.ClientSession() as session:
        for k, s in enumerate(sorted(jobs)):
            fn = f"{OUT}/{s}.parquet"
            if os.path.exists(fn):
                status[s] = "exists_skip"
                continue
            urls = [(m, f"{BASE}/{s}/1h/{s}-1h-{m}.zip") for m in sorted(jobs[s])]
            res = await asyncio.gather(*[fetch(session, sem, u) for _, u in urls])
            dfs, st = [], {}
            for (m, _), (code, b) in zip(urls, res):
                st[m] = code
                if code == 200:
                    dfs.append(parse(b, s))
            if dfs:
                d = pl.concat(dfs).unique("open_time").sort("open_time")
                # hard guard: nothing at/after 2026-04-01 00:00 UTC
                d = d.filter(pl.col("open_time") < 1775001600000)
                d.write_parquet(fn)
            status[s] = {"n_ok": sum(v == 200 for v in st.values()), "n_404": sum(v == 404 for v in st.values()),
                         "fail": [m for m, v in st.items() if v == "fail"]}
            if k % 25 == 0:
                print(k, s, status[s], flush=True)
    json.dump(status, open(f"{RUN}/dl_kl1h_status.json", "w"), indent=0)


asyncio.run(main())
print("done", flush=True)
