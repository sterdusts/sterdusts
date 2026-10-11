# spot-check implementer's kl1h_um parquet against freshly downloaded archive zips
import sys, zipfile, io, glob, os
import polars as pl
dl = sys.argv[1]
for z in sorted(glob.glob(os.path.join(dl, "*.zip"))):
    sym = os.path.basename(z).split("-1h-")[0]
    with zipfile.ZipFile(z) as zf:
        raw = zf.read(zf.namelist()[0]).decode()
    lines = [l for l in raw.splitlines() if l and l[0].isdigit()]
    rows = [l.split(",") for l in lines]
    ot = [int(r[0]) for r in rows]; ot = [t // 1000 if t > 1e14 else t for t in ot]
    d = pl.DataFrame({"open_time": ot, "open_dl": [float(r[1]) for r in rows]})
    k = pl.read_parquet(f"/home/user/data/flow/kl1h_um/{sym}.parquet").select("open_time", "open")
    j = d.join(k, on="open_time", how="left")
    print(sym, os.path.basename(z), d.height, "missing_in_impl", j["open"].null_count(),
          "maxabsdiff", j.select((pl.col("open") - pl.col("open_dl")).abs().max()).item())
