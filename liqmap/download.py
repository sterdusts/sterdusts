"""币安 U 本位永续归档数据下载 → Parquet。

数据源: https://data.binance.vision （公开归档，含已下架币种）
用法:
  python download.py daily1d   # 全币种 1d K线 (月文件)，用于动态选币池
  python download.py kl5m SYMS  # 5m K线 (月文件)
  python download.py metrics SYMS START END  # 5m 持仓量/多空比 (日文件)
  python download.py funding SYMS  # 资金费率 (月文件)
"""
import asyncio, io, sys, zipfile, os
from datetime import date, timedelta
import aiohttp
import polars as pl

BASE = "https://data.binance.vision/data/futures/um"
OUT = os.environ.get("DATA_DIR", "/home/user/data")
KL_COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
           "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]


def months(start=(2021, 1), end=(2026, 9)):
    y, m = start
    while (y, m) <= end:
        yield f"{y:04d}-{m:02d}"
        m += 1
        if m > 12:
            y, m = y + 1, 1


async def fetch(session, sem, url, retries=4):
    async with sem:
        for i in range(retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                    if r.status == 404:
                        return None
                    if r.status == 200:
                        return await r.read()
            except Exception:
                pass
            await asyncio.sleep(2 ** i)
    return None


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


async def run(urls_meta, parse, conc=48):
    sem = asyncio.Semaphore(conc)
    conn = aiohttp.TCPConnector(limit=conc)
    out = []
    async with aiohttp.ClientSession(connector=conn) as s:
        async def one(url, meta):
            b = await fetch(s, sem, url)
            if b is None:
                return
            try:
                df = parse(b, meta)
                if df is not None and df.height:
                    out.append(df)
            except Exception as e:
                print("parse fail", url, e, file=sys.stderr)
        tasks = [one(u, m) for u, m in urls_meta]
        for i in range(0, len(tasks), 2000):
            await asyncio.gather(*tasks[i:i + 2000])
            print(f"  {min(i + 2000, len(tasks))}/{len(tasks)} done, got {len(out)}", flush=True)
    return out


def parse_kl(b, sym):
    df = read_zip_csv(b, KL_COLS).drop("ignore")
    return df.with_columns(pl.lit(sym).alias("symbol")).with_columns(
        [pl.col(c).cast(pl.Float64) for c in ["open", "high", "low", "close", "volume", "quote_volume",
                                                "taker_buy_volume", "taker_buy_quote_volume"]]
        + [pl.col("open_time").cast(pl.Int64), pl.col("count").cast(pl.Int64)]).drop("close_time")


def parse_metrics(b, sym):
    df = read_zip_csv(b)
    return df.select(
        pl.col("create_time").cast(pl.Utf8).str.to_datetime(strict=False).alias("ts"),
        pl.lit(sym).alias("symbol"),
        pl.col("sum_open_interest").cast(pl.Float64, strict=False).alias("oi"),
        pl.col("sum_open_interest_value").cast(pl.Float64, strict=False).alias("oi_value"),
        pl.col("count_toptrader_long_short_ratio").cast(pl.Float64, strict=False).alias("top_ls_acct"),
        pl.col("sum_toptrader_long_short_ratio").cast(pl.Float64, strict=False).alias("top_ls_pos"),
        pl.col("count_long_short_ratio").cast(pl.Float64, strict=False).alias("ls_acct"),
        pl.col("sum_taker_long_short_vol_ratio").cast(pl.Float64, strict=False).alias("taker_ls"),
    )


def parse_funding(b, sym):
    df = read_zip_csv(b, ["calc_time", "funding_interval_hours", "last_funding_rate"])
    return df.select(pl.col("calc_time").cast(pl.Int64), pl.lit(sym).alias("symbol"),
                     pl.col("funding_interval_hours").cast(pl.Int64),
                     pl.col("last_funding_rate").cast(pl.Float64, strict=False).alias("funding_rate"))


def main():
    mode = sys.argv[1]
    if mode == "daily1d":
        syms = open("/tmp/claude-0/probe/um_syms.txt").read().split()
        um = [(f"{BASE}/monthly/klines/{s}/1d/{s}-1d-{m}.zip", s) for s in syms for m in months()]
        dfs = asyncio.run(run(um, parse_kl, conc=96))
        pl.concat(dfs).write_parquet(f"{OUT}/kl1d.parquet")
    elif mode == "kl5m":
        syms = sys.argv[2].split(",")
        os.makedirs(f"{OUT}/kl5m", exist_ok=True)
        um = [(f"{BASE}/monthly/klines/{s}/5m/{s}-5m-{m}.zip", s) for s in syms for m in months()]
        dfs = asyncio.run(run(um, parse_kl))
        df = pl.concat(dfs)
        for (s,), g in df.group_by("symbol"):
            g.sort("open_time").unique("open_time", keep="last").sort("open_time").write_parquet(f"{OUT}/kl5m/{s}.parquet")
    elif mode == "metrics":
        syms = sys.argv[2].split(",")
        d0, d1 = date.fromisoformat(sys.argv[3]), date.fromisoformat(sys.argv[4])
        os.makedirs(f"{OUT}/metrics", exist_ok=True)
        days = [d0 + timedelta(i) for i in range((d1 - d0).days + 1)]
        um = [(f"{BASE}/daily/metrics/{s}/{s}-metrics-{d}.zip", s) for s in syms for d in days]
        dfs = asyncio.run(run(um, parse_metrics, conc=64))
        df = pl.concat(dfs)
        for (s,), g in df.group_by("symbol"):
            g.sort("ts").unique("ts", keep="last").sort("ts").write_parquet(f"{OUT}/metrics/{s}.parquet")
    elif mode == "funding":
        syms = sys.argv[2].split(",")
        um = [(f"{BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip", s) for s in syms for m in months()]
        dfs = asyncio.run(run(um, parse_funding))
        pl.concat(dfs).sort("symbol", "calc_time").write_parquet(f"{OUT}/funding.parquet")


if __name__ == "__main__" and sys.argv[1] != "universe":
    main()


def needed_months(warm=2):
    """每个币在池内的月份 + 之前 warm 个月预热。"""
    u = pl.read_parquet(f"{OUT}/universe.parquet")
    need = {}
    for d, s in u.select("d", "symbol").iter_rows():
        y, m = d.year, d.month
        for k in range(warm + 1):
            mm = m - k; yy = y
            while mm <= 0:
                mm += 12; yy -= 1
            need.setdefault(s, set()).add(f"{yy:04d}-{mm:02d}")
    return need


def universe_mode(batch=25):
    import calendar
    need = needed_months()
    syms = sorted(need)
    for p in ("kl5m", "metrics"):
        os.makedirs(f"{OUT}/{p}", exist_ok=True)
    fund = []
    for i in range(0, len(syms), batch):
        bs = [s for s in syms[i:i + batch] if not os.path.exists(f"{OUT}/metrics/{s}.parquet")]
        if not bs:
            continue
        print(f"batch {i}: {len(bs)} symbols", flush=True)
        kl = [(f"{BASE}/monthly/klines/{s}/5m/{s}-5m-{m}.zip", s) for s in bs for m in sorted(need[s])]
        fr = [(f"{BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip", s) for s in bs for m in sorted(need[s])]
        mt = []
        for s in bs:
            for m in sorted(need[s]):
                y, mo = map(int, m.split("-"))
                for dd in range(1, calendar.monthrange(y, mo)[1] + 1):
                    mt.append((f"{BASE}/daily/metrics/{s}/{s}-metrics-{y:04d}-{mo:02d}-{dd:02d}.zip", s))
        k = asyncio.run(run(kl, parse_kl))
        f = asyncio.run(run(fr, parse_funding))
        mtd = asyncio.run(run(mt, parse_metrics, conc=64))
        if k:
            for (s,), g in pl.concat(k).group_by("symbol"):
                g.unique("open_time", keep="last").sort("open_time").write_parquet(f"{OUT}/kl5m/{s}.parquet")
        if f:
            pl.concat(f).write_parquet(f"{OUT}/funding_{i}.parquet")
        if mtd:
            for (s,), g in pl.concat(mtd).group_by("symbol"):
                g.unique("ts", keep="last").sort("ts").write_parquet(f"{OUT}/metrics/{s}.parquet")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "universe":
    universe_mode()
