"""Download full daily-candle history for every Upbit KRW market (+ USDT-BTC, BTC-USDT? no) .

Usage: python3 -I dl_upbit.py <market_all.json> <out_dir> <end_exclusive_utc>
- pages backward with count=200 & to=<earliest candle utc>Z until an empty / short page
- rate limit: <= 8 req/s (Upbit candles group allows 10/s, 600/min); honours Remaining-Req header, backs off on 429
- resumable: skips markets whose <market>.json already exists
"""
import json, os, sys, time
import requests

MA, OUT, END = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
markets = [x["market"] for x in json.load(open(MA)) if x["market"].startswith("KRW-")]
markets = sorted(markets) + ["USDT-BTC"]
S = requests.Session()
URL = "https://api.upbit.com/v1/candles/days"
MIN_GAP = 1.0 / 8.0
last = [0.0]
nreq = [0]


def get(params):
    for attempt in range(8):
        dt = time.time() - last[0]
        if dt < MIN_GAP:
            time.sleep(MIN_GAP - dt)
        last[0] = time.time()
        try:
            r = S.get(URL, params=params, timeout=30)
        except Exception as e:  # network hiccup
            print("ERR", params, e, flush=True)
            time.sleep(2 + 2 * attempt)
            continue
        nreq[0] += 1
        rr = r.headers.get("Remaining-Req", "")
        if r.status_code == 429:
            time.sleep(1.5 + attempt)
            continue
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            print("HTTP", r.status_code, params, r.text[:200], flush=True)
            time.sleep(2 + 2 * attempt)
            continue
        try:  # adaptive: if per-second budget nearly exhausted, pause
            sec = int(rr.split("sec=")[1].split(";")[0])
            if sec <= 1:
                time.sleep(1.0)
        except Exception:
            pass
        return r.json()
    raise RuntimeError(f"failed {params}")


t0 = time.time()
for i, m in enumerate(markets):
    fn = os.path.join(OUT, m + ".json")
    if os.path.exists(fn):
        continue
    rows, to, status = [], END, "ok"
    while True:
        d = get({"market": m, "to": to, "count": 200})
        if d is None:
            status = "404"
            break
        if not d:
            break
        rows.extend(d)
        earliest = min(x["candle_date_time_utc"] for x in d)
        if len(d) < 200:
            break
        to = earliest + "Z"
    tmp = fn + ".tmp"
    json.dump({"market": m, "status": status, "end_exclusive": END,
               "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows}, open(tmp, "w"))
    os.replace(tmp, fn)
    print(f"{i+1}/{len(markets)} {m} {status} n={len(rows)} "
          f"first={min((x['candle_date_time_utc'] for x in rows), default=None)} req={nreq[0]} t={time.time()-t0:.0f}s", flush=True)
print("DONE", nreq[0], time.time() - t0, flush=True)
