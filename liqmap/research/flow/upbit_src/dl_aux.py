"""Aux downloads: Binance spot BTCUSDT 1h (data.binance.vision monthly zips + daily zips for the tail),
ECB reference FX history, FRED DEXKOUS.  Usage: python3 -I dl_aux.py <out_dir>"""
import os, sys, time, hashlib
import requests

OUT = sys.argv[1]
os.makedirs(os.path.join(OUT, "bn1h"), exist_ok=True)
S = requests.Session()


def fetch(url, fn):
    if os.path.exists(fn):
        return "cached"
    for a in range(5):
        try:
            r = S.get(url, timeout=60)
            if r.status_code == 404:
                return "404"
            r.raise_for_status()
            open(fn + ".tmp", "wb").write(r.content)
            os.replace(fn + ".tmp", fn)
            return "ok"
        except Exception as e:
            print("retry", url, e, flush=True)
            time.sleep(2 + 2 * a)
    return "fail"


base = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/"
months = [f"{y}-{m:02d}" for y in range(2021, 2027) for m in range(1, 13) if f"{y}-{m:02d}" <= "2026-09"]
for ym in months:
    fn = os.path.join(OUT, "bn1h", f"BTCUSDT-1h-{ym}.zip")
    st = fetch(base + os.path.basename(fn), fn)
    # verify checksum
    cs = S.get(base + os.path.basename(fn) + ".CHECKSUM", timeout=30)
    ok = cs.status_code == 200 and cs.text.split()[0] == hashlib.sha256(open(fn, "rb").read()).hexdigest()
    print(ym, st, "sha_ok" if ok else "SHA_MISMATCH", flush=True)
    time.sleep(0.2)

for name, url in [("eurofxref-hist.zip", "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip"),
                  ("DEXKOUS.csv", "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXKOUS")]:
    print(name, fetch(url, os.path.join(OUT, name)), flush=True)
