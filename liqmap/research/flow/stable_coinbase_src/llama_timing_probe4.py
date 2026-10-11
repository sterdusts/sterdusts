"""Block-level probe: scan every block in [t0, t1] (UTC) for an exact (|diff|<1 USD) match of DefiLlama USDC/Ethereum minted(D).
usage: probe4.py stablecoin_2.json D HH:MM(start, relative to D 00:00, may be negative like -00:30) minutes"""
import json, sys, time, datetime as dt, urllib.request
RPCS = ["https://eth.drpc.org", "https://eth.merkle.io"]
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    for i in range(8):
        try:
            req = urllib.request.Request(RPCS[i % 2], data=body, headers={"content-type": "application/json", "User-Agent": "probe/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                j = json.loads(r.read())
            if "result" in j: return j["result"]
        except Exception: pass
        time.sleep(1 + i)
    raise RuntimeError(method)
def bts(n): return int(rpc("eth_getBlockByNumber", [hex(n), False])["timestamp"], 16)
L = int(rpc("eth_blockNumber", []), 16); LT = bts(L)
def block_at(t):
    n = min(L, L - (LT - t) // 12)
    for _ in range(6):
        d = (t - bts(n)) // 12
        if d == 0: break
        n = min(L, n + d)
    while bts(n) > t: n -= 1
    return n
def supply(n): return int(rpc("eth_call", [{"to": USDC, "data": "0x18160ddd"}, hex(n)]), 16) / 1e6
llama = {x["date"]: x["minted"]["peggedUSD"] for x in json.load(open(sys.argv[1]))["chainBalances"]["Ethereum"]["tokens"] if "minted" in x}
D = int(dt.datetime.strptime(sys.argv[2], "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())
off_min = int(sys.argv[3]); span = int(sys.argv[4])
v = llama[D]
a, b = block_at(D + off_min * 60), block_at(D + (off_min + span) * 60)
prev, hits = None, []
for n in range(a, b + 1):
    s = supply(n)
    if s != prev:
        if abs(s - v) < 1:
            hits.append(n)
        prev = s
first = [(n, dt.datetime.utcfromtimestamp(bts(n)).strftime("%m-%d %H:%M:%S")) for n in hits[:1]]
last = [(n, dt.datetime.utcfromtimestamp(bts(n)).strftime("%m-%d %H:%M:%S")) for n in hits[-1:]]
print(sys.argv[2], "minted", v, "blocks scanned", b - a + 1, "exact-match supply-change blocks:", len(hits), "first", first, "last", last, flush=True)
