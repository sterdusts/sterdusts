"""G0 复核（只用特征，不读标签）：LPM R²(state ~ X + seg)。输出 g0_check.json（拒绝覆盖）。"""
import datetime as dt, json, pathlib, sys
import numpy as np, polars as pl
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import arb
OUT = arb.ARB / "g0_check.json"
assert not OUT.exists()
f = arb.build_features()
res = {}
for nm, end in [("main", arb.END14), ("dev", arb.DEV_END)]:
    d = f.filter(pl.col("date").is_between(arb.DEV0, end)).drop_nulls(["mt1_state", *arb.XC])
    Z = arb.Zmat(d, [], arb.XC, True); y = (d["mt1_state"].to_numpy() > 0).astype(float)
    b = np.linalg.lstsq(Z, y, rcond=None)[0]; u = y - Z @ b
    res[nm] = {"n": d.height, "lpm_r2": float(1 - u @ u / ((y - y.mean()) @ (y - y.mean())))}
st = f.filter(pl.col("date").is_between(arb.DEV0, arb.DEV_END))["mt1_state"].to_numpy()
res["switches_dev"] = int((st[1:] != st[:-1]).sum())
res["G0_pass"] = max(res["main"]["lpm_r2"], res["dev"]["lpm_r2"]) <= 0.50
OUT.write_text(json.dumps(res, indent=1)); print(json.dumps(res))
