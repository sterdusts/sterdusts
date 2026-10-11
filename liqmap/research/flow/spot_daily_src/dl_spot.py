"""Stage 1-4: candidate spot pairs for every perp in kl1d -> S3 listing -> download monthly 1d zips -> raw parquet.

usage: python -I dl_spot.py <workdir>
writes <workdir>/cands.parquet, <workdir>/pair_files.parquet, <workdir>/spot_raw.parquet
"""
import asyncio, io, re, sys, zipfile, urllib.parse, os, time
import aiohttp
import polars as pl

W = sys.argv[1]
KL1D = "/home/user/data/kl1d.parquet"
Q5 = ["USDT", "FDUSD", "USDC", "TUSD", "BUSD"]
QX = ["USD1", "U"]  # newer stable quotes, stored separately (not part of spec'd 5)
CONV = ["USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "BUSDUSDT", "USD1USDT", "UUSDT"]
M0, M1 = "2021-01", "2026-09"
S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DL = "https://data.binance.vision/data/spot/monthly/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
        "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]

# old perp base -> (successor spot base, perp_price / successor_spot_price). Used only on days
# where the perp's own-name spot has no bar. Verified later by a daily price check.
RENAME = {
    "LUNA2": ("LUNA", 1.0),      # Terra 2.0 trades as LUNA on spot from 2022-05-28
    "RNDR": ("RENDER", 1.0),
    "MATIC": ("POL", 1.0),
    "FTM": ("S", 1.0),
    "EOS": ("A", 1.0),           # Vaulta
    "KLAY": ("KAIA", 1.0),
    "TOMO": ("VIC", 1.0),
    "DAR": ("D", 1.0),
    "MKR": ("SKY", 24000.0),
    "GAL": ("G", 60.0),
    "BTT": ("BTTC", 1000.0),
    "DODOX": ("DODO", 1.0),      # guess, price-checked
    "RAYSOL": ("RAY", 1.0),      # guess, price-checked
}
PREFIX = [("1000000", 1e6), ("1000", 1e3), ("1M", 1e6)]


def build_cands(listing):
    perps = pl.read_parquet(KL1D, columns=["symbol"])["symbol"].unique().sort().to_list()
    rows = []
    for p in perps:
        b = p[:-4]
        has = lambda x: any(x + q in listing for q in Q5)
        if has(b):
            rows.append((p, b, 1.0, "exact", 0))
        else:
            for pre, mult in PREFIX:
                if b.startswith(pre) and len(b) > len(pre):
                    st = b[len(pre):]
                    if has(st):
                        rows.append((p, st, mult, "strip_prefix", 0))
                    break
        if b in RENAME:
            succ, mult = RENAME[b]
            if has(succ):
                rows.append((p, succ, mult, "rename", 0 if not has(b) else 1))
    return perps, pl.DataFrame(rows, schema=["perp_symbol", "spot_base", "mult", "method", "priority"], orient="row")


async def list_pair(s, sem, pair):
    out = []
    marker = ""
    async with sem:
        while True:
            q = urllib.parse.urlencode({"prefix": f"data/spot/monthly/klines/{pair}/1d/", "marker": marker})
            for i in range(5):
                try:
                    async with s.get(f"{S3}?{q}", timeout=aiohttp.ClientTimeout(total=60)) as r:
                        x = await r.text()
                        if r.status == 200:
                            break
                except Exception:
                    pass
                await asyncio.sleep(2 ** i)
            else:
                raise RuntimeError(f"list failed {pair}")
            keys = re.findall(r"<Key>([^<]+)</Key>", x)
            out += [k for k in keys if k.endswith(".zip")]
            if "<IsTruncated>true</IsTruncated>" not in x or not keys:
                break
            marker = keys[-1]
    return pair, out


async def fetch(s, sem, url):
    async with sem:
        for i in range(5):
            try:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=120)) as r:
                    if r.status == 404:
                        return None, 404
                    if r.status == 200:
                        return await r.read(), 200
            except Exception:
                pass
            await asyncio.sleep(2 ** i)
    return None, -1


def parse(blob, pair, month):
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        raw = z.read(z.namelist()[0])
    first = raw[:1].decode(errors="ignore")
    has_header = not first.isdigit()
    df = pl.read_csv(raw, has_header=has_header, new_columns=None if has_header else COLS,
                     infer_schema_length=0)
    if has_header:
        df.columns = COLS[: len(df.columns)]
    df = df.select(
        pl.col("open_time").cast(pl.Int64), pl.col("close_time").cast(pl.Int64),
        *[pl.col(c).cast(pl.Float64) for c in ["open", "high", "low", "close", "volume", "quote_volume",
                                               "taker_buy_volume", "taker_buy_quote_volume"]],
        pl.col("count").cast(pl.Int64),
    )
    return df.with_columns(pl.lit(pair).alias("spot_symbol"), pl.lit(month).alias("src_month"),
                           pl.lit(has_header).alias("src_header"))


async def main():
    listing = set(open(f"{W}/spot_syms.txt").read().split())
    perps, cands = build_cands(listing)
    cands.write_parquet(f"{W}/cands.parquet")
    pairs = []
    for b in cands["spot_base"].unique().to_list():
        for q in Q5 + QX:
            if b + q in listing:
                pairs.append((b + q, b, q))
    for c in CONV:
        if c in listing:
            pairs.append((c, c[:-4], "USDT"))
    pairs = sorted(set(pairs))
    print(f"perps={len(perps)} cand_rows={cands.height} bases={cands['spot_base'].n_unique()} pairs={len(pairs)}", flush=True)

    t = time.time()
    conn = aiohttp.TCPConnector(limit=32)
    async with aiohttp.ClientSession(connector=conn) as s:
        sem = asyncio.Semaphore(16)
        res = await asyncio.gather(*[list_pair(s, sem, p) for p, _, _ in pairs])
        files = []
        for pair, keys in res:
            for k in keys:
                m = re.search(r"-1d-(\d{4}-\d{2})\.zip$", k)
                if m and M0 <= m.group(1) <= M1:
                    files.append((pair, m.group(1), k))
        meta = {p: (b, q) for p, b, q in pairs}
        pf = pl.DataFrame(files, schema=["spot_symbol", "month", "key"], orient="row")
        pf.write_parquet(f"{W}/pair_files.parquet")
        print(f"listed in {time.time()-t:.0f}s: {len(files)} monthly files over {pf['spot_symbol'].n_unique()} pairs", flush=True)

        sem = asyncio.Semaphore(32)
        out, miss, nbytes = [], [], 0

        async def one(pair, month, key):
            nonlocal nbytes
            url = "https://data.binance.vision/" + urllib.parse.quote(key)
            b, st = await fetch(s, sem, url)
            if b is None:
                miss.append((pair, month, st))
                return
            nbytes += len(b)
            try:
                out.append(parse(b, pair, month))
            except Exception as e:
                miss.append((pair, month, -2))
                print("parse fail", pair, month, e, file=sys.stderr)

        tasks = [one(*f) for f in files]
        for i in range(0, len(tasks), 2000):
            await asyncio.gather(*tasks[i:i + 2000])
            print(f"  {min(i+2000, len(tasks))}/{len(tasks)} files, {nbytes/1e6:.1f}MB, {time.time()-t:.0f}s", flush=True)
    df = pl.concat(out)
    df = df.with_columns(pl.col("spot_symbol").replace_strict({p: m[0] for p, m in meta.items()}).alias("base"),
                         pl.col("spot_symbol").replace_strict({p: m[1] for p, m in meta.items()}).alias("quote"))
    df.write_parquet(f"{W}/spot_raw.parquet")
    pl.DataFrame(miss, schema=["spot_symbol", "month", "status"], orient="row").write_parquet(f"{W}/missing.parquet")
    print(f"done rows={df.height} missing={len(miss)} bytes={nbytes/1e6:.1f}MB", flush=True)


asyncio.run(main())
