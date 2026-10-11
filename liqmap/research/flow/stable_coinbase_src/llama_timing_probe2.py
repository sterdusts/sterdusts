"""Fine probe (5-min grid, D-1 21:00 .. D 03:00 UTC): exact on-chain moment matching DefiLlama USDC/Ethereum minted(D)."""
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
for ds in sys.argv[2:]:
    D = int(dt.datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())
    v = llama[D]
    t0 = D - 3 * 3600; t1 = min(D + 3 * 3600, LT)
    a, b = block_at(t0), block_at(t1)
    pts = []
    for k in range(0, (t1 - t0) // 300 + 1):
        t = t0 + 300 * k
        n = a + round((b - a) * (t - t0) / (t1 - t0))
        pts.append((t, n, supply(n))); time.sleep(0.05)
    best = sorted(pts, key=lambda r: abs(r[2] - v))[:4]
    exact = [dt.datetime.utcfromtimestamp(r[0]).strftime("%H:%M") for r in pts if abs(r[2] - v) < 1]
    print(ds, "minted", v, "exact-match 5min points:", exact[:3], "...", exact[-3:], len(exact),
          "| closest:", [(dt.datetime.utcfromtimestamp(r[0]).strftime("%m-%d %H:%M"), round(r[2] - v, 2)) for r in best], flush=True)
