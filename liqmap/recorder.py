"""多交易所实时数据录制器（7×24 运行在你的服务器或东京 VPS 上）。

录制内容：
- 币安 U 本位：全市场强平 (!forceOrder@arr，注意每币每秒只推最新一笔，是欠采样)、标记价格/资金费 (!markPrice@arr@1s)、
  指定币的逐笔成交 (aggTrade) 和 20 档盘口 (depth20@100ms)；持仓量用 REST 每 60 秒轮询。
- Bybit 线性合约：allLiquidation、publicTrade、orderbook.50、tickers（含持仓量、资金费）。
- OKX 永续：liquidation-orders（全市场）、trades、books5、open-interest、funding-rate。
- Hyperliquid：trades（带买卖双方地址）、l2Book、activeAssetCtx；并从成交里积累地址库，定期抓取大户持仓和清算价。

存储：原始 JSON + 接收时间，按 交易所/频道/日期 分区写 Parquet（解析放到研究阶段做，录制端越简单越不容易丢数据）。
断线自动重连（指数退避），每次断线记录到 gaps.jsonl，便于之后检查数据缺口。

用法：python recorder.py --out /data/raw --symbols BTC,ETH,SOL,...
"""
import argparse
import asyncio
import json
import logging
import os
import random
import time
from collections import defaultdict

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq
import websockets

log = logging.getLogger("rec")
SCHEMA = pa.schema([("recv_ns", pa.int64()), ("channel", pa.string()), ("payload", pa.string())])


class Sink:
    """按 (交易所, 频道) 缓冲，定时或攒够行数后落盘。"""

    def __init__(self, root, flush_sec=60, max_rows=50_000):
        self.root, self.flush_sec, self.max_rows = root, flush_sec, max_rows
        self.buf = defaultdict(list)
        self.last = time.time()
        self.counts = defaultdict(int)

    def put(self, ex, ch, payload):
        key = (ex, ch)
        self.buf[key].append((time.time_ns(), ch, payload))
        self.counts[key] += 1
        if len(self.buf[key]) >= self.max_rows:
            self._flush(key)

    def _flush(self, key):
        rows = self.buf.pop(key, [])
        if not rows:
            return
        ex, ch = key
        day = time.strftime("%Y-%m-%d", time.gmtime(rows[0][0] / 1e9))
        d = os.path.join(self.root, ex, ch.replace("/", "_"), f"date={day}")
        os.makedirs(d, exist_ok=True)
        tbl = pa.table({"recv_ns": [r[0] for r in rows], "channel": [r[1] for r in rows],
                        "payload": [r[2] for r in rows]}, schema=SCHEMA)
        fn = os.path.join(d, f"part-{rows[0][0]}.parquet")
        pq.write_table(tbl, fn + ".tmp", compression="zstd")
        os.replace(fn + ".tmp", fn)

    def flush_all(self):
        for k in list(self.buf):
            self._flush(k)
        self.last = time.time()

    async def loop(self):
        while True:
            await asyncio.sleep(5)
            if time.time() - self.last >= self.flush_sec:
                self.flush_all()
                tot = {f"{e}/{c}": n for (e, c), n in self.counts.items()}
                log.info("flushed; msgs so far: %s", json.dumps(tot)[:500])


def note_gap(root, ex, reason):
    with open(os.path.join(root, "gaps.jsonl"), "a") as f:
        f.write(json.dumps({"ts": time.time(), "exchange": ex, "reason": reason}) + "\n")


async def ws_runner(name, url, subs, on_msg, root, ping=None, ping_every=20):
    """通用 WebSocket 循环：连接 → 订阅 → 收消息；断线指数退避重连。"""
    backoff = 1
    while True:
        try:
            async with websockets.connect(url, open_timeout=15, ping_interval=20, ping_timeout=20,
                                          max_size=2 ** 24) as ws:
                for s in subs:
                    await ws.send(json.dumps(s))
                log.info("%s connected (%d subs)", name, len(subs))
                backoff = 1
                last_ping = time.time()
                while True:
                    try:
                        m = await asyncio.wait_for(ws.recv(), timeout=ping_every)
                        on_msg(m)
                    except asyncio.TimeoutError:
                        pass
                    if ping and time.time() - last_ping >= ping_every:
                        await ws.send(ping if isinstance(ping, str) else json.dumps(ping))
                        last_ping = time.time()
        except Exception as e:  # noqa: BLE001
            log.warning("%s disconnected: %r; retry in %ss", name, e, backoff)
            note_gap(root, name, repr(e)[:200])
            await asyncio.sleep(backoff + random.random())
            backoff = min(backoff * 2, 60)


# ---------------- 各交易所 ----------------

def binance_tasks(sink, root, syms):
    ex = "binance"

    def on(m):
        st = json.loads(m).get("stream", "unknown@unknown")
        # "!forceOrder@arr" → forceOrder；"btcusdt@aggTrade" → aggTrade
        ch = st.split("@")[0].lstrip("!") if st.startswith("!") else st.split("@")[1]
        sink.put(ex, ch, m)

    streams = ["!forceOrder@arr", "!markPrice@arr@1s"]
    streams += [f"{s.lower()}usdt@aggTrade" for s in syms] + [f"{s.lower()}usdt@depth20@100ms" for s in syms]
    tasks = []
    # 每个连接最多 200 个流
    for i in range(0, len(streams), 200):
        url = "wss://fstream.binance.com/stream?streams=" + "/".join(streams[i:i + 200])
        tasks.append(ws_runner(f"binance{i // 200}", url, [], on, root))
    tasks.append(binance_oi_poll(sink, syms))
    return tasks


async def binance_oi_poll(sink, syms, every=60):
    async with aiohttp.ClientSession() as s:
        while True:
            t0 = time.time()
            for sym in syms:
                try:
                    async with s.get("https://fapi.binance.com/fapi/v1/openInterest",
                                     params={"symbol": f"{sym}USDT"}, timeout=aiohttp.ClientTimeout(total=10)) as r:
                        if r.status == 200:
                            sink.put("binance", "openInterest", await r.text())
                except Exception as e:  # noqa: BLE001
                    log.debug("oi poll %s %r", sym, e)
            await asyncio.sleep(max(every - (time.time() - t0), 1))


def bybit_tasks(sink, root, syms):
    ex = "bybit"

    def on(m):
        d = json.loads(m)
        if "topic" in d:
            sink.put(ex, d["topic"].split(".")[0], m)

    topics = []
    for s in syms:
        topics += [f"allLiquidation.{s}USDT", f"publicTrade.{s}USDT", f"orderbook.50.{s}USDT", f"tickers.{s}USDT"]
    subs = [{"op": "subscribe", "args": topics[i:i + 10]} for i in range(0, len(topics), 10)]
    return [ws_runner("bybit", "wss://stream.bybit.com/v5/public/linear", subs, on, root, ping={"op": "ping"})]


def okx_tasks(sink, root, syms):
    ex = "okx"

    def on(m):
        if m == "pong":
            return
        d = json.loads(m)
        if "data" in d:
            sink.put(ex, d["arg"]["channel"], m)

    args = [{"channel": "liquidation-orders", "instType": "SWAP"}]
    for s in syms:
        inst = f"{s}-USDT-SWAP"
        args += [{"channel": c, "instId": inst} for c in ("trades", "books5", "open-interest", "funding-rate")]
    subs = [{"op": "subscribe", "args": args[i:i + 50]} for i in range(0, len(args), 50)]
    return [ws_runner("okx", "wss://ws.okx.com:443/ws/v5/public", subs, on, root, ping="ping")]


class HLAddressBook:
    """从 Hyperliquid 成交里积累活跃地址，按近期成交额排序，供持仓快照轮询。"""

    def __init__(self, path, half_life_h=24):
        self.path = path
        self.ntl = defaultdict(float)
        self.ts = {}
        self.decay = half_life_h * 3600 / 0.693
        if os.path.exists(path):
            self.ntl.update(json.load(open(path)))

    def add(self, user, notional):
        now = time.time()
        old = self.ts.get(user, now)
        self.ntl[user] = self.ntl[user] * pow(2.718281828, -(now - old) / self.decay) + notional
        self.ts[user] = now

    def top(self, k):
        return [u for u, _ in sorted(self.ntl.items(), key=lambda x: -x[1])[:k]]

    def save(self):
        keep = dict(sorted(self.ntl.items(), key=lambda x: -x[1])[:50_000])
        json.dump(keep, open(self.path + ".tmp", "w"))
        os.replace(self.path + ".tmp", self.path)


def hyperliquid_tasks(sink, root, syms, top_k=2000, snap_every=900, req_per_min=240):
    ex = "hyperliquid"
    book = HLAddressBook(os.path.join(root, "hl_addresses.json"))

    def on(m):
        d = json.loads(m)
        ch = d.get("channel")
        if ch in ("subscriptionResponse", "pong"):
            return
        sink.put(ex, ch or "unknown", m)
        if ch == "trades":
            for tr in d["data"]:
                ntl = float(tr["px"]) * float(tr["sz"])
                for u in tr.get("users", []):
                    book.add(u, ntl)

    subs = []
    for s in syms:
        subs += [{"method": "subscribe", "subscription": {"type": t, "coin": s}}
                 for t in ("trades", "l2Book", "activeAssetCtx")]

    async def snapshot_loop():
        """每 snap_every 秒抓一轮前 top_k 个地址的持仓（含清算价、杠杆）。限速：req_per_min。"""
        await asyncio.sleep(120)  # 先积累一些地址
        async with aiohttp.ClientSession() as s:
            while True:
                t0 = time.time()
                users = book.top(top_k)
                book.save()
                for u in users:
                    try:
                        async with s.post("https://api.hyperliquid.xyz/info",
                                          json={"type": "clearinghouseState", "user": u},
                                          timeout=aiohttp.ClientTimeout(total=10)) as r:
                            if r.status == 200:
                                body = await r.json()
                                if body.get("assetPositions"):
                                    sink.put(ex, "clearinghouseState", json.dumps({"user": u, "t": time.time(), **body}))
                            elif r.status == 429:
                                await asyncio.sleep(10)
                    except Exception as e:  # noqa: BLE001
                        log.debug("hl snap %r", e)
                    await asyncio.sleep(60 / req_per_min)
                log.info("hyperliquid snapshot round: %d users in %.0fs", len(users), time.time() - t0)
                await asyncio.sleep(max(snap_every - (time.time() - t0), 5))

    return [ws_runner("hyperliquid", "wss://api.hyperliquid.xyz/ws", subs, on, root, ping={"method": "ping"},
                      ping_every=30), snapshot_loop()]


async def main(a):
    os.makedirs(a.out, exist_ok=True)
    sink = Sink(a.out, flush_sec=a.flush_sec)
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    tasks = [sink.loop()]
    ex = set(a.exchanges.split(","))
    if "binance" in ex:
        tasks += binance_tasks(sink, a.out, syms)
    if "bybit" in ex:
        tasks += bybit_tasks(sink, a.out, syms)
    if "okx" in ex:
        tasks += okx_tasks(sink, a.out, syms)
    if "hyperliquid" in ex:
        tasks += hyperliquid_tasks(sink, a.out, syms, top_k=a.hl_top_k)
    try:
        if a.duration:
            await asyncio.wait_for(asyncio.gather(*tasks), a.duration)
        else:
            await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        pass
    finally:
        sink.flush_all()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/user/data/raw")
    ap.add_argument("--symbols", default="BTC,ETH,SOL,XRP,DOGE,BNB,SUI,HYPE,ADA,LINK,AVAX,ENA,1000PEPE")
    ap.add_argument("--exchanges", default="binance,bybit,okx,hyperliquid")
    ap.add_argument("--flush-sec", type=int, default=60)
    ap.add_argument("--hl-top-k", type=int, default=2000)
    ap.add_argument("--duration", type=int, default=0, help="测试用：运行 N 秒后退出")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main(ap.parse_args()))
