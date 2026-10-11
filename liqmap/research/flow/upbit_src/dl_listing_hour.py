"""For every downloaded KRW market, fetch the 60-min candles of its first daily-candle UTC day to get the
first traded hour (approx. listing time).  Also fetch KRW-BTC hourly candles on a few historical dates to
verify the daily-candle cut (UTC 00:00).  Usage: python3 -I dl_listing_hour.py <days_dir> <out_json>"""
import glob, json, os, sys, time, datetime as dt
import requests

DAYS, OUT = sys.argv[1], sys.argv[2]
S = requests.Session()
URL = "https://api.upbit.com/v1/candles/minutes/60"
last = [0.0]


def get(params):
    for a in range(8):
        w = 1 / 7.0 - (time.time() - last[0])
        if w > 0:
            time.sleep(w)
        last[0] = time.time()
        try:
            r = S.get(URL, params=params, timeout=30)
        except Exception as e:
            print("ERR", e, flush=True); time.sleep(2 + a); continue
        if r.status_code == 429:
            time.sleep(1.5 + a); continue
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            print("HTTP", r.status_code, r.text[:200], flush=True); time.sleep(2 + a); continue
        rr = r.headers.get("Remaining-Req", "")
        try:
            if int(rr.split("sec=")[1].split(";")[0]) <= 1:
                time.sleep(1.0)
        except Exception:
            pass
        return r.json()
    raise RuntimeError(params)


res = json.load(open(OUT)) if os.path.exists(OUT) else {"listing": {}, "cut_check": {}}
for fn in sorted(glob.glob(os.path.join(DAYS, "KRW-*.json"))):
    j = json.load(open(fn))
    m = j["market"]
    if m in res["listing"] or not j["rows"]:
        continue
    first = min(x["candle_date_time_utc"] for x in j["rows"])[:10]
    to = (dt.date.fromisoformat(first) + dt.timedelta(days=1)).isoformat() + "T00:00:00Z"
    d = get({"market": m, "to": to, "count": 24})
    hrs = sorted(x["candle_date_time_utc"] for x in (d or []) if x["candle_date_time_utc"][:10] == first)
    res["listing"][m] = {"first_day": first, "hours": hrs,
                         "first_hour_turnover": [x["candle_acc_trade_price"] for x in sorted(d or [], key=lambda z: z["candle_date_time_utc"])][:3]}
    print(m, first, hrs[:1], len(hrs), flush=True)
    if len(res["listing"]) % 25 == 0:
        json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)

# daily-cut verification on historical dates: sum of 24 hourly candles of UTC day vs daily candle
for day in ["2021-03-15", "2021-11-10", "2022-06-20", "2023-08-01", "2024-03-05", "2025-04-07"]:
    to = (dt.date.fromisoformat(day) + dt.timedelta(days=1)).isoformat() + "T00:00:00Z"
    for m in ["KRW-BTC", "KRW-XRP"]:
        d = get({"market": m, "to": to, "count": 48})
        res["cut_check"][f"{m}|{day}"] = d
json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
print("DONE", len(res["listing"]))
