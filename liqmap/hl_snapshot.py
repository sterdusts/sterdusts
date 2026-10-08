"""抓取 Hyperliquid 大户持仓快照（排行榜按账户规模取前 N），输出每个仓位的清算价、杠杆、名义价值。"""
import asyncio, json, sys, time
import aiohttp
import polars as pl

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
OUT = sys.argv[2] if len(sys.argv) > 2 else "/home/user/data/hl_snapshot.parquet"
MODE = sys.argv[3] if len(sys.argv) > 3 else "top"


async def main():
    async with aiohttp.ClientSession() as s:
        async with s.get("https://stats-data.hyperliquid.xyz/Mainnet/leaderboard") as r:
            rows = (await r.json(content_type=None))["leaderboardRows"]
        rows.sort(key=lambda x: -float(x["accountValue"]))
        if MODE == "top":
            users = [r["ethAddress"] for r in rows[:N]]
        else:  # 分层抽样：排除前 1500 大户，账户价值 > $1000，按规模十分位各抽 N/10
            import random
            random.seed(1)
            rest = [r for r in rows[1500:] if float(r["accountValue"]) > 1000]
            k = len(rest) // 10
            users = []
            for i in range(10):
                chunk = rest[i * k:(i + 1) * k]
                users += [r["ethAddress"] for r in random.sample(chunk, min(N // 10, len(chunk)))]
        async with s.post("https://api.hyperliquid.xyz/info", json={"type": "allMids"}) as r:
            mids = {k: float(v) for k, v in (await r.json()).items()}
        out, sem = [], asyncio.Semaphore(4)

        async def one(u):
            async with sem:
                for _ in range(3):
                    try:
                        async with s.post("https://api.hyperliquid.xyz/info",
                                          json={"type": "clearinghouseState", "user": u}) as r:
                            if r.status == 429:
                                await asyncio.sleep(5); continue
                            d = await r.json()
                            break
                    except Exception:
                        await asyncio.sleep(2)
                else:
                    return
                await asyncio.sleep(0.4)
                av = float(d["marginSummary"]["accountValue"])
                for ap in d.get("assetPositions", []):
                    p = ap["position"]
                    out.append({"user": u, "account_value": av, "coin": p["coin"], "szi": float(p["szi"]),
                                "entry_px": float(p["entryPx"] or 0),
                                "liq_px": float(p["liquidationPx"]) if p.get("liquidationPx") else None,
                                "lev_type": p["leverage"]["type"], "lev": float(p["leverage"]["value"]),
                                "ntl": float(p["positionValue"]), "margin": float(p["marginUsed"]),
                                "mid": mids.get(p["coin"])})
        t = time.time()
        await asyncio.gather(*[one(u) for u in users])
        df = pl.DataFrame(out).with_columns(pl.lit(time.time()).alias("snap_ts"))
        df.write_parquet(OUT)
        print(f"{len(users)} users, {df.height} positions, {time.time()-t:.0f}s")

asyncio.run(main())
