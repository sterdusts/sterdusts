"""Downloads for perp_daily (data.binance.vision public archive). Parses zips in memory, never extracts to disk.

modes:
  gapfill   : daily 1d klines for calendar days missing inside each symbol's kl1d span
  firstdate : earliest archived 1d kline for symbols whose kl1d series starts 2021-01-01 (left-censored)
  funding   : monthly fundingRate files for trading symbol-months not present in /home/user/data/funding_*.parquet
Outputs go to /home/user/data/flow/ (new files only).
"""
import asyncio, io, sys, zipfile, os, re, glob, datetime as dt
import aiohttp
import polars as pl

BASE = "https://data.binance.vision/data/futures/um"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DATA = "/home/user/data"
OUT = "/home/user/data/flow"
SCR = "/tmp/claude-0/-home-user-sterdusts/3ad56404-3e60-5e0b-ae02-0725979310e7/scratchpad/flow/perp-daily"
KL_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
CONC = 16


async def fetch(session, sem, url, retries=5):
    """returns (status, bytes): status 200 / 404 / 'fail'"""
    async with sem:
        for i in range(retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                    if r.status == 404:
                        return 404, None
                    if r.status == 200:
                        return 200, await r.read()
                    if r.status in (403, 429, 503):
                        await asyncio.sleep(5 * (i + 1))
                        continue
            except Exception:
                pass
            await asyncio.sleep(2 ** i)
    return "fail", None


def read_zip_csv(blob, cols=None):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:200].decode(errors="ignore").split("\n")[0]
    has_header = not first[:1].isdigit()
    df = pl.read_csv(raw, has_header=has_header, new_columns=None if has_header else cols,
                     infer_schema_length=10000)
    if cols and has_header and len(df.columns) == len(cols):
        df.columns = cols
    return df


def parse_kl(b, sym):
    df = read_zip_csv(b, KL_COLS).drop("ignore")
    return df.with_columns(pl.lit(sym).alias("symbol")).with_columns(
        [pl.col(c).cast(pl.Float64) for c in ["open", "high", "low", "close", "volume", "quote_volume",
                                                "taker_buy_volume", "taker_buy_quote_volume"]]
        + [pl.col("open_time").cast(pl.Int64), pl.col("count").cast(pl.Int64)]).drop("close_time")


def parse_funding(b, sym):
    df = read_zip_csv(b, ["calc_time", "funding_interval_hours", "last_funding_rate"])
    return df.select(pl.col("calc_time").cast(pl.Int64), pl.lit(sym).alias("symbol"),
                     pl.col("funding_interval_hours").cast(pl.Int64),
                     pl.col("last_funding_rate").cast(pl.Float64, strict=False).alias("funding_rate"))


async def run(jobs, parse, conc=CONC):
    """jobs: list of (url, meta). returns (frames, status_rows)"""
    sem = asyncio.Semaphore(conc)
    frames, stat = [], []
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=conc)) as s:
        async def one(url, meta):
            st, b = await fetch(s, sem, url)
            nrow = 0
            if b is not None:
                try:
                    df = parse(b, meta)
                    if df is not None and df.height:
                        frames.append(df)
                        nrow = df.height
                except Exception as e:
                    st = f"parse_fail:{e}"[:80]
            stat.append((url, str(meta), str(st), nrow))
        tasks = [one(u, m) for u, m in jobs]
        for i in range(0, len(tasks), 1000):
            await asyncio.gather(*tasks[i:i + 1000])
            print(f"  {min(i + 1000, len(tasks))}/{len(tasks)}", flush=True)
    return frames, pl.DataFrame(stat, schema=["url", "meta", "status", "rows"], orient="row")


def kl1d():
    return pl.read_parquet(f"{DATA}/kl1d.parquet").with_columns(
        pl.from_epoch("open_time", time_unit="ms").dt.date().alias("d"))


def gapfill():
    df = kl1d().sort("symbol", "d")
    sp = df.group_by("symbol").agg(pl.col("d").min().alias("a"), pl.col("d").max().alias("b"))
    grid = sp.with_columns(pl.date_ranges("a", "b").alias("d")).explode("d").select("symbol", "d")
    miss = grid.join(df.select("symbol", "d"), on=["symbol", "d"], how="anti").sort("symbol", "d")
    print("missing symbol-days:", miss.height)
    jobs = [(f"{BASE}/daily/klines/{s}/1d/{s}-1d-{d}.zip", s) for s, d in miss.iter_rows()]
    frames, st = asyncio.run(run(jobs, parse_kl))
    print(st.group_by("status").len())
    out = pl.concat(frames).sort("symbol", "open_time") if frames else None
    if out is not None:
        out = out.unique(["symbol", "open_time"], keep="last").sort("symbol", "open_time")
        # keep only rows that are truly missing from kl1d (archive daily file holds 1 row/day)
        out = out.with_columns(pl.from_epoch("open_time", time_unit="ms").dt.date().alias("d")).join(
            miss, on=["symbol", "d"], how="semi").drop("d")
        out.write_parquet(f"{OUT}/perp_daily_aux_kl1d_gapfill.parquet")
        print("gapfill rows written:", out.height)
    st.write_parquet(f"{SCR}/dl_status_gapfill.parquet")


async def s3_list(session, sem, prefix):
    keys, marker = [], ""
    while True:
        url = f"{S3}?delimiter=/&prefix={prefix}" + (f"&marker={marker}" if marker else "")
        st, b = await fetch(session, sem, url)
        if b is None:
            return None
        txt = b.decode()
        ks = re.findall(r"<Key>([^<]+)</Key>", txt)
        keys += ks
        if "<IsTruncated>true</IsTruncated>" in txt and ks:
            marker = ks[-1]
        else:
            return keys


def firstdate():
    df = kl1d()
    first = df.group_by("symbol").agg(pl.col("d").min().alias("a"))
    syms = sorted(first.filter(pl.col("a") == dt.date(2021, 1, 1))["symbol"].to_list())
    print("left-censored symbols:", len(syms))

    async def main():
        sem = asyncio.Semaphore(8)
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=8)) as s:
            async def one(sym):
                keys = await s3_list(s, sem, f"data/futures/um/monthly/klines/{sym}/1d/")
                if not keys:
                    return sym, None, "list_fail_or_empty"
                ms = sorted(k for k in keys if k.endswith(".zip"))
                st, b = await fetch(s, sem, f"https://data.binance.vision/{ms[0]}")
                if b is None:
                    return sym, None, f"zip_{st}"
                k = parse_kl(b, sym).sort("open_time")
                k = k.filter(pl.col("quote_volume") > 0) if k.filter(pl.col("quote_volume") > 0).height else k
                d0 = dt.datetime.fromtimestamp(k["open_time"][0] / 1000, dt.UTC).date()
                return sym, d0, ms[0].split("/")[-1]
            return await asyncio.gather(*[one(x) for x in syms])
    res = asyncio.run(main())
    out = pl.DataFrame(res, schema=[("symbol", pl.String), ("archive_first_date", pl.Date), ("src", pl.String)],
                       orient="row")
    print(out.sort("archive_first_date"))
    out.write_parquet(f"{OUT}/perp_daily_aux_first_date.parquet")


def funding():
    df = kl1d()
    gf = f"{OUT}/perp_daily_aux_kl1d_gapfill.parquet"
    if os.path.exists(gf):
        g = pl.read_parquet(gf).with_columns(pl.from_epoch("open_time", time_unit="ms").dt.date().alias("d"))
        df = pl.concat([df, g.select(df.columns)])
    sm = df.filter(pl.col("quote_volume") > 0).select("symbol", pl.col("d").dt.strftime("%Y-%m").alias("m")).unique()
    f = pl.concat([pl.read_parquet(x) for x in glob.glob(f"{DATA}/funding_*.parquet")])
    fm = f.select("symbol", pl.from_epoch("calc_time", time_unit="ms").dt.strftime("%Y-%m").alias("m")).unique()
    need = sm.join(fm, on=["symbol", "m"], how="anti").sort("symbol", "m")
    print("need symbol-months:", need.height)
    jobs = [(f"{BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip", s) for s, m in need.iter_rows()]
    frames, st = asyncio.run(run(jobs, parse_funding))
    print(st.group_by("status").len())
    st.write_parquet(f"{SCR}/dl_status_funding.parquet")
    if frames:
        out = pl.concat(frames).sort("symbol", "calc_time")
        out.write_parquet(f"{OUT}/perp_daily_aux_funding_backfill.parquet")
        print("funding rows written:", out.height)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"gapfill": gapfill, "firstdate": firstdate, "funding": funding}[sys.argv[1]]()
