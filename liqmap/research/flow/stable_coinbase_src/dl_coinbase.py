"""Download Coinbase Exchange 1h candles (public /products/{id}/candles, max 300 per request, end inclusive).
Rate: public limit is ~10 req/s per IP; we stay at <=3 req/s and back off on 429.
Output: <SCR>/cb1h_<PRODUCT>.parquet with columns ts(UTC open time), low, high, open, close, volume(base)."""
import sys, time, json, datetime as dt
import urllib.request, urllib.error
import polars as pl

SCR = "/tmp/claude-0/-home-user-sterdusts/3ad56404-3e60-5e0b-ae02-0725979310e7/scratchpad/flow/stable-coinbase"
START = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
END = dt.datetime(2026, 9, 30, 23, tzinfo=dt.timezone.utc)   # last hour open time (inclusive)
H = dt.timedelta(hours=1)


def get(url, retries=8):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-dl/1.0", "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 * (i + 1)); continue
            raise
        except Exception:
            time.sleep(2 * (i + 1))
    raise RuntimeError("failed " + url)


def main(pid):
    rows, t, n = [], START, 0
    while t <= END:
        e = min(t + 299 * H, END)
        url = (f"https://api.exchange.coinbase.com/products/{pid}/candles?granularity=3600"
               f"&start={t.strftime('%Y-%m-%dT%H:%M:%SZ')}&end={e.strftime('%Y-%m-%dT%H:%M:%SZ')}")
        d = get(url)
        if isinstance(d, dict):
            raise RuntimeError(f"{pid} {t}: {d}")
        rows.extend(d); n += 1
        t = e + H
        time.sleep(0.35)
    df = (pl.DataFrame(rows, schema=["t", "low", "high", "open", "close", "volume"], orient="row")
          .with_columns(pl.from_epoch("t", time_unit="s").dt.replace_time_zone("UTC").alias("ts"))
          .drop("t").unique("ts").sort("ts")
          .filter((pl.col("ts") >= START) & (pl.col("ts") <= END)))
    df.write_parquet(f"{SCR}/cb1h_{pid}.parquet")
    print(pid, "requests", n, "rows", df.height, df["ts"].min(), df["ts"].max(), flush=True)


if __name__ == "__main__":
    for pid in sys.argv[1:]:
        main(pid)
