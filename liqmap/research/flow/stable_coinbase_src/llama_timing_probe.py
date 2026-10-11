"""Probe: which intraday snapshot does DefiLlama's daily stablecoin value (timestamp = D 00:00 UTC) correspond to?
Compare DefiLlama USDC/Ethereum 'minted' (= ERC20 totalSupply) for date D against on-chain totalSupply at hourly
blocks from D-1 00:00 to D+1 23:00 UTC (public archive RPC). Pure data-timing check; no returns involved."""
import json, sys, time, datetime as dt, urllib.request

RPCS = ["https://eth.drpc.org", "https://eth.merkle.io"]
USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
RAW = sys.argv[1]
DATES = sys.argv[2:]


def rpc(method, params):
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    for i in range(8):
        u = RPCS[i % len(RPCS)]
        try:
            req = urllib.request.Request(u, data=body, headers={"content-type": "application/json", "User-Agent": "probe/1.0"})
            with urllib.request.urlopen(req, timeout=20) as r:
                j = json.loads(r.read())
            if "result" in j:
                return j["result"]
        except Exception:
            pass
        time.sleep(1 + i)
    raise RuntimeError(method)


def block_ts(n):
    return int(rpc("eth_getBlockByNumber", [hex(n), False])["timestamp"], 16)


LATEST = int(rpc("eth_blockNumber", []), 16)
LATEST_TS = block_ts(LATEST)


def block_at(t):
    """last block with timestamp <= t (approx, refined by slot arithmetic)"""
    n = LATEST - (LATEST_TS - t) // 12
    for _ in range(6):
        bt = block_ts(n)
        delta = (t - bt) // 12
        if delta == 0:
            break
        n += delta
    while block_ts(n) > t:
        n -= 1
    return n


def supply(n):
    return int(rpc("eth_call", [{"to": USDC, "data": "0x18160ddd"}, hex(n)]), 16) / 1e6


llama = {x["date"]: x["minted"]["peggedUSD"] for x in json.load(open(RAW))["chainBalances"]["Ethereum"]["tokens"] if "minted" in x}
for ds in DATES:
    D = int(dt.datetime.strptime(ds, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp())
    anchors = {k: block_at(D + (k - 1) * 86400) for k in range(4)}  # D-1, D, D+1, D+2 00:00
    rows = []
    for h in range(72):
        t = D - 86400 + 3600 * h
        k = h // 24
        a, b = anchors[k], anchors[k + 1]
        n = a + round((b - a) * (h % 24) / 24)
        rows.append((t, n, supply(n)))
        time.sleep(0.1)
    for lag in (-1, 0, 1):
        v = llama.get(D + lag * 86400)
        if v is None:
            continue
        best = sorted(rows, key=lambda r: abs(r[2] - v))[:3]
        print(ds, "llama date", dt.datetime.utcfromtimestamp(D + lag * 86400).date(), "minted", v,
              "| closest on-chain hours:", [(dt.datetime.utcfromtimestamp(r[0]).strftime("%m-%d %H:00"), round(r[2] - v, 2)) for r in best], flush=True)
