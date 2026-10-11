"""MT1' (MT1P-stable-issuance-v2) 信号 v2 与控制变量构建 —— 只建特征，不读取任何标签列。

规格：/home/user/sterdusts/liqmap/research/flow_prereg_2026-10-10.json, id="MT1P-stable-issuance-v2"（commit b3f95f8）。
输出：本目录 mkt_timing_signals_v2.parquet（键 date=d；全部列在 decision_at=(d+1) 00:00 UTC 可知），
      本目录 mt1p_btc_leg_multiplier_dev.parquet（组合试验用的 m 与 decision_at，只覆盖开发期）。

锁定集保护：所有输入在读取时即过滤到 date <= 2026-03-31（原始 JSON 过滤 ts < 2026-04-01 00:00 UTC），
并断言读到的最大日期 <= 2026-03-31。信号只用正向 shift / rolling / cumsum（因果），故过滤不改变任何开发期取值。
本脚本不读取 fwd_* / res_* / fund_* 标签列。

运行：python3 -I build_signals_v2.py
"""
import datetime as dt
import hashlib
import json
import pathlib

import numpy as np
import polars as pl

FLOW = pathlib.Path("/home/user/data/flow")
RUN = FLOW / "MT1P-stable-issuance-v2_run"
OUT = RUN / "mkt_timing_signals_v2.parquet"
OUT_M = RUN / "mt1p_btc_leg_multiplier_dev.parquet"
DEV_END = dt.date(2026, 3, 31)
EXCL_BASE = ["BTC", "USDC", "FDUSD", "BUSD", "TUSD", "U", "USD1", "USTC", "PAXG", "XAUT"]
BASKET = ["usdt_total", "usdc_total", "busd_total", "fdusd_total", "tusd_total"]
CRY = ["dai_total", "usds_total", "usde_total"]
EXT_IDS = {262: "usd1", 286: "usdg", 120: "pyusd", 250: "rlusd", 306: "gate_gusd",
           17: "husd", 19: "gemini_gusd", 11: "usdp", 27: "usdk"}
CLIP = 0.01
W1, W2 = 14, 180  # g14 = L(D)-L(D-14); g166 = L(D-14)-L(D-180)
K = 14 / 166


def T(e: pl.Expr) -> pl.Expr:
    """同一变换 T(X)(D) = [X(D)-X(D-14)] - (14/166)[X(D-14)-X(D-180)]（行=日历日，需日历连续）。"""
    return (e - e.shift(W1)) - K * (e.shift(W1) - e.shift(W2))


def sgn(e: pl.Expr) -> pl.Expr:
    return pl.when(e.is_null()).then(None).when(e > 0).then(1).otherwise(-1).cast(pl.Int8)


def ffill0(c: str) -> pl.Expr:
    """首个非空值之后前向填充；首个非空值之前记 0。"""
    return pl.col(c).forward_fill().fill_null(0.0)


def dl_clip_cum(lnx: pl.Expr) -> pl.Expr:
    """ln 序列 -> 日差（首日 0）-> clip ±1% -> 累加。"""
    return (lnx - lnx.shift(1)).fill_null(0.0).clip(-CLIP, CLIP).cum_sum()


def assert_calendar(df: pl.DataFrame, name: str):
    d = df["date"]
    assert d.is_sorted() and d.is_unique().all(), name
    assert (d.diff().drop_nulls() == dt.timedelta(days=1)).all(), f"{name} 日历不连续"
    assert d.max() <= DEV_END, f"{name} 越过开发期"


def load_ext() -> pl.DataFrame:
    frames = []
    for sid, nm in EXT_IDS.items():
        raw = json.loads((RUN / "raw_ext" / f"sc_{sid}.json").read_bytes())
        ts = [int(r["date"]) for r in raw]
        assert all(b > a for a, b in zip(ts, ts[1:])) and all(t % 86400 == 0 for t in ts), sid
        rows = [(dt.date(1970, 1, 1) + dt.timedelta(days=int(r["date"]) // 86400),
                 (r.get("totalCirculating") or {}).get("peggedUSD")) for r in raw]
        f = pl.DataFrame(rows, schema={"date": pl.Date, f"ext_{nm}": pl.Float64}, orient="row")
        frames.append(f.filter(pl.col("date") <= DEV_END))  # 锁定集保护
    return frames


def stable_block() -> pl.DataFrame:
    s = (pl.scan_parquet(FLOW / "stablecoins_daily.parquet")
         .filter(pl.col("date") <= DEV_END)  # 锁定集保护
         .select(["date", "known_at"] + BASKET + CRY).collect().sort("date"))
    assert_calendar(s, "stablecoins_daily")
    for f in load_ext():
        s = s.join(f, on="date", how="left")
    ext_cols = [c for c in s.columns if c.startswith("ext_")]
    s = s.with_columns([ffill0(c).alias("x_" + c.replace("_total", "")) for c in BASKET + CRY + ext_cols])
    xb = [f"x_{c.replace('_total', '')}" for c in BASKET]
    xe = [f"x_{c}" for c in ext_cols]
    s = s.with_columns(
        pl.sum_horizontal(xb).alias("mt1_F"),
        pl.sum_horizontal([f"x_{c.replace('_total', '')}" for c in CRY]).alias("cry_F"),
        (pl.sum_horizontal(xb) + pl.sum_horizontal(xe)).alias("ext_F"),
    )
    s = s.with_columns((pl.col("mt1_F").log() - pl.col("mt1_F").log().shift(1)).fill_null(0.0).alias("mt1_dl"))
    s = s.with_columns(pl.col("mt1_dl").clip(-CLIP, CLIP).alias("mt1_dlc"))
    s = s.with_columns(pl.col("mt1_dlc").cum_sum().alias("mt1_L"))
    s = s.with_columns((pl.col("mt1_L") - pl.col("mt1_L").shift(W1)).alias("mt1_g14"),
                       (pl.col("mt1_L").shift(W1) - pl.col("mt1_L").shift(W2)).alias("mt1_g166"))
    s = s.with_columns((pl.col("mt1_g14") - K * pl.col("mt1_g166")).alias("mt1_I"))
    s = s.with_columns(sgn(pl.col("mt1_I")).alias("mt1_state"))
    # 同变换派生状态
    s = s.with_columns(
        T(dl_clip_cum(pl.col("x_usdt").log())).alias("T_USDT"),
        T(dl_clip_cum(pl.col("x_usdc").log())).alias("T_USDC"),
        T(dl_clip_cum(pl.col("cry_F").log())).alias("T_cry"),
        T(dl_clip_cum(pl.col("ext_F").log())).alias("T_ext"),
    )
    # 因果纳入篮子：候选 = 主篮子 5 + 扩展 9（均为法币抵押美元币）；成员 S(D-1)={c: x_c(D-1)>=1e9}，
    # dl(D) = ln Σ_{c∈S(D-1)} x_c(D) - ln Σ_{c∈S(D-1)} x_c(D-1)（链式，成员变化不产生跳变；只用 ≤D 的数据）。
    cand = xb + xe
    X = s.select(cand).to_numpy()
    mem = X >= 1e9
    dlc_c = np.zeros(len(X))
    for i in range(1, len(X)):
        m = mem[i - 1]
        a, b = X[i, m].sum(), X[i - 1, m].sum()
        dlc_c[i] = np.log(a / b) if (m.any() and a > 0 and b > 0) else 0.0
    s = s.with_columns(pl.Series("causal_dl", dlc_c), pl.Series("causal_n", mem.sum(1).astype(np.int16)))
    s = s.with_columns(T(pl.col("causal_dl").clip(-CLIP, CLIP).cum_sum()).alias("T_causal"))
    s = s.with_columns([sgn(pl.col(f"T_{k}")).alias(f"state_{k}") for k in ("USDT", "USDC", "cry", "ext", "causal")])
    # state_3c：mt1_state 连续 3 天同号才切换（只报告）
    raw = s["mt1_state"].cast(pl.Float64).to_numpy()
    out = np.full(len(raw), np.nan)
    cur = np.nan
    for i in range(len(raw)):
        r = raw[i]
        if np.isnan(r):
            continue
        if np.isnan(cur):
            cur = r
        elif r != cur and i >= 2 and raw[i - 1] == r and raw[i - 2] == r:
            cur = r
        out[i] = cur
    s = s.with_columns(pl.Series("state_3c", out).cast(pl.Int8, strict=False))
    keep = (["date", pl.col("known_at").alias("mt1_known_at"), "mt1_F", "mt1_dl", "mt1_dlc", "mt1_L", "mt1_g14",
             "mt1_g166", "mt1_I", "mt1_state", "state_3c", "causal_n"]
            + [f"T_{k}" for k in ("USDT", "USDC", "cry", "ext", "causal")]
            + [f"state_{k}" for k in ("USDT", "USDC", "cry", "ext", "causal")])
    return s.select(keep)


def spot_block() -> pl.DataFrame:
    sm = (pl.read_parquet(FLOW / "spot_map.parquet").filter(pl.col("status") == "ok")
          .group_by("spot_base").agg(pl.col("perp_first_live").min().alias("pfl")))
    b = (pl.scan_parquet(FLOW / "spot_daily_base.parquet")
         .filter(pl.col("date") <= DEV_END)  # 锁定集保护
         .select("date", "base", "qv_usdt_USDT", "tbqv_usdt_USDT")
         .filter(~pl.col("base").is_in(EXCL_BASE))
         .join(sm.lazy(), left_on="base", right_on="spot_base", how="inner")
         .filter(pl.col("pfl") <= pl.col("date"))  # 因果选币 B_t
         .group_by("date").agg(pl.col("qv_usdt_USDT").sum().alias("sp_qv"),
                               pl.col("tbqv_usdt_USDT").sum().alias("sp_tb"), pl.len().alias("sp_nB"))
         .collect().sort("date"))
    assert_calendar(b, "spot_daily_base")
    b = b.with_columns(
        (pl.col("sp_tb").rolling_sum(7, min_samples=7) / pl.col("sp_qv").rolling_sum(7, min_samples=7) - 0.5)
        .alias("c_tkr7"),
        pl.col("sp_qv").rolling_sum(7, min_samples=7).log().alias("SV"),
    )
    b = b.with_columns(T(pl.col("SV")).alias("T_SV"))
    b = b.with_columns(sgn(pl.col("T_SV")).alias("state_SV"))
    return b.select("date", "sp_nB", "c_tkr7", "SV", "T_SV", "state_SV")


def perp_block() -> pl.DataFrame:
    p = (pl.scan_parquet(FLOW / "perp_daily.parquet")
         .filter((pl.col("symbol") == "BTCUSDT") & (pl.col("date") <= DEV_END))  # 锁定集保护
         .select("date", "bar_status", "close", "ret_1d", "funding_1d", "oi", "oi_value", "quote_volume")
         .collect().sort("date"))
    assert_calendar(p, "perp_daily BTCUSDT")
    assert (p["bar_status"] == "trade").all()
    lc = pl.col("close").log()
    p = p.with_columns(
        (lc - lc.shift(7)).alias("c_mom7"),
        (lc - lc.shift(30)).alias("c_mom30"),
        pl.col("ret_1d").rolling_std(30, min_samples=25).alias("c_rv30"),
        pl.col("funding_1d").rolling_sum(7, min_samples=7).alias("c_fund7"),
        (pl.col("oi").log() - pl.col("oi").log().shift(7)).alias("c_doi7"),
        pl.col("funding_1d").rolling_sum(30, min_samples=30).alias("fund30"),
        T(lc).alias("P_I"),
        (lc - lc.shift(90)).alias("mom90"),
        pl.when(pl.col("close").rolling_mean(200, min_samples=200).is_null()).then(None)
        .when(pl.col("close") > pl.col("close").rolling_mean(200, min_samples=200)).then(1).otherwise(0)
        .cast(pl.Int8).alias("ma200"),
        (pl.col("quote_volume").rolling_mean(7, min_samples=7)
         / pl.col("quote_volume").rolling_mean(90, min_samples=90)).log().alias("dvol30"),
        pl.col("oi_value").forward_fill(limit=3).alias("oi_value_ff3"),
        (lc - lc.shift(30)).alias("tsmom30"),
    )
    p = p.with_columns(T(pl.col("oi_value_ff3").log()).alias("T_OI"))
    p = p.with_columns(sgn(pl.col("P_I")).alias("state_P"), sgn(pl.col("T_OI")).alias("state_OI"))
    return p.select("date", "close", "c_mom7", "c_mom30", "c_rv30", "c_fund7", "c_doi7", "fund30", "P_I",
                    "mom90", "ma200", "dvol30", "tsmom30", "T_OI", "state_P", "state_OI")


def main():
    assert not OUT.exists() and not OUT_M.exists(), "输出已存在，拒绝覆盖"
    cal = pl.DataFrame({"date": pl.date_range(dt.date(2020, 1, 1), DEV_END, "1d", eager=True)})
    df = cal.join(stable_block(), on="date", how="left").join(perp_block(), on="date", how="left") \
        .join(spot_block(), on="date", how="left")
    df = df.with_columns(
        pl.col("date").cast(pl.Datetime("us", "UTC")).dt.offset_by("1d").alias("decision_at"),
        pl.when(pl.col("date").dt.year() <= 2022).then(pl.lit("2022"))
        .when(pl.col("date").dt.year() == 2023).then(pl.lit("2023"))
        .when(pl.col("date").dt.year() == 2024).then(pl.lit("2024"))
        .otherwise(pl.lit("2025-01..2026-03")).alias("seg"),
    )
    assert df["date"].max() == DEV_END and df["date"].is_unique().all()
    df.write_parquet(OUT)
    # 组合试验用：BTC 腿乘数 m(d)=1.0 (+1) / 0.5 (-1)，decision_at=(d+1) 00:00 UTC；只到 decision_at<=2026-03-31 00:00
    m = (df.filter(pl.col("date").is_between(dt.date(2022, 1, 1), DEV_END - dt.timedelta(days=1)))
         .select("date", "decision_at", "mt1_I", "mt1_state",
                 pl.when(pl.col("mt1_state") == 1).then(1.0).when(pl.col("mt1_state") == -1).then(0.5)
                 .otherwise(None).alias("m")))
    assert m["decision_at"].max() <= dt.datetime(2026, 3, 31, tzinfo=dt.timezone.utc)
    assert m["m"].null_count() == 0
    m.write_parquet(OUT_M)
    sha = lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
    print(json.dumps({"rows": df.height, "cols": df.width, "max_date": str(df["date"].max()),
                      "script_sha256": sha(__file__), "signals_sha256": sha(OUT),
                      "multiplier_sha256": sha(OUT_M)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
