"""Arb check 2: naive raw recomputation of S5, P5 and A-sign inputs for random C_d rows (features only, d<=2026-03-22)."""
import json, os
from datetime import date, timedelta
import numpy as np, polars as pl
RUN = "/home/user/data/flow/D1P-spot-lead-flow_run"; FLOW = "/home/user/data/flow"
OUT = "/home/user/data/flow/D1P-spot-lead-flow_arb/arb_check2.json"; assert not os.path.exists(OUT)
END = date(2026, 3, 22)
F = pl.read_parquet(f"{RUN}/d1_features.parquet", columns=["date", "symbol", "in_C", "spot_base", "S5", "P5", "zS5", "zP5", "A", "in_S"])
C = F.filter(pl.col("in_C"))
rng = np.random.default_rng(11)
s = C[np.sort(rng.choice(C.height, 400, replace=False))]
syms = s["symbol"].unique().to_list(); bases = s["spot_base"].unique().to_list()
P = pl.scan_parquet(f"{FLOW}/perp_daily.parquet").select("date", "symbol", "taker_buy_quote_volume", "quote_volume").filter(pl.col("symbol").is_in(syms) & (pl.col("date") <= END)).collect()
B = pl.scan_parquet(f"{FLOW}/spot_daily_base.parquet").select("date", "base", "tbqv_usdt_USDT").filter(pl.col("base").is_in(bases) & (pl.col("date") <= END)).collect()
SP = pl.scan_parquet(f"{FLOW}/spot_daily_perp.parquet").select("date", "perp_symbol", "qv_usdt_USDT").filter(pl.col("perp_symbol").is_in(syms) & (pl.col("date") <= END)).collect()
pk = {(r[0], r[1]): (r[2], r[3]) for r in P.iter_rows()}
bk = {(r[0], r[1]): r[2] for r in B.iter_rows()}
sk = {(r[0], r[1]): r[2] for r in SP.iter_rows()}
dS, dP = [], []
for d, sym, base, S5, P5 in s.select("date", "symbol", "spot_base", "S5", "P5").iter_rows():
    ds = [d - timedelta(days=k) for k in range(5)]
    dS.append(abs(sum(bk[(x, base)] for x in ds) / sum(sk[(x, sym)] for x in ds) - S5))
    dP.append(abs(sum(pk[(x, sym)][0] for x in ds) / sum(pk[(x, sym)][1] for x in ds) - P5))
# A definition: clip((zS5-zP5) - median_S(zS5-zP5), -5, 5)
Sx = F.filter(pl.col("in_S")).with_columns((pl.col("zS5") - pl.col("zP5")).alias("dA"))
Sx = Sx.with_columns(((pl.col("dA") - pl.col("dA").median().over("date")).clip(-5, 5)).alias("A2"))
res = {"n": len(dS), "S5_max_abs_diff": float(max(dS)), "P5_max_abs_diff": float(max(dP)),
       "A_max_abs_diff": float((Sx["A2"] - Sx["A"]).abs().max()), "S5_median": float(C["S5"].median()), "P5_median": float(C["P5"].median())}
json.dump(res, open(OUT, "w"), indent=1); print(res)
