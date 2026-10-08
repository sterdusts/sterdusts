"""代理清算地图模拟器。

思路（只用币安公开归档数据）：
- 每根 5 分钟 K 线，持仓量 (OI) 增加 → 视为新开仓。按主动买入占比 b 把新仓位分给多头 (b) 和空头 (1-b)，
  开仓价取 K 线 VWAP，再按假设的杠杆分布换算成清算价，累加到对数价格网格上。
- OI 减少 → 平仓。主动卖出主要是多头平仓，主动买入主要是空头平仓，按比例缩小两侧的仓位密度。
- 换手：即使 OI 不变，也有一部分老仓位平掉、在当前价重新开出（比例 = churn * 成交量 / OI）。
- 清算触发：K 线最低价 ≤ 多头清算价 → 这些多头被清算并移除；最高价 ≥ 空头清算价 → 空头被清算。
- 输出每小时的特征：价格上下各距离内的清算密度、密度不平衡、估计清算量等。

所有特征在整点 H 只使用 H 之前已经收盘的 K 线和时间戳 ≤ H 的 OI 快照。
"""
import numpy as np
import polars as pl
from numba import njit

DATA = "/home/user/data"
BIN = 0.002  # 网格宽度：0.2%（对数价格）
LEV = np.array([3, 5, 10, 20, 25, 50, 75, 100], dtype=np.float64)
LEV_W = np.array([0.10, 0.15, 0.25, 0.20, 0.10, 0.12, 0.04, 0.04], dtype=np.float64)
MMR = 0.005
# 校准版：按 Hyperliquid 散户分层抽样（2026-10-08，2000 账户）的真实清算距离分布设定
# 距离 d → 等效杠杆 1/(d+MMR)；约 21% 名义无清算价或距离 >100%，不进入清算网格
CALIB_D = np.array([0.005, 0.010, 0.015, 0.025, 0.04, 0.075, 0.15, 0.25, 0.40, 0.75])
CALIB_W = np.array([0.025, 0.020, 0.027, 0.015, 0.061, 0.086, 0.170, 0.146, 0.119, 0.122])
CALIB_LEV = 1.0 / (CALIB_D + MMR)
DISTS = np.array([0.01, 0.02, 0.03, 0.05, 0.10])


def liq_offsets(lev, lev_w):
    d = np.maximum(1.0 / lev - MMR, 0.004)
    lo = np.round(np.log(1 - d) / BIN).astype(np.int64)  # 多头清算价相对开仓价的格数（负）
    so = np.round(np.log(1 + d) / BIN).astype(np.int64)  # 空头（正）
    return lo, so, lev_w / lev_w.sum()


@njit(cache=True)
def simulate(lo_p, hi_p, vwap, close, vol, buy_frac, oi, new_seg, is_hour_end,
             lo_off, so_off, w, grid0, nbins, churn, dists_bins, dists_bins_dn):
    n = len(close)
    nd = len(dists_bins)
    L = np.zeros(nbins)
    S = np.zeros(nbins)
    sL = 1.0
    sS = 1.0
    TL = 0.0
    TS = 0.0
    out_up = np.full((n, nd), np.nan)    # 上方空头清算密度（相对 OI）
    out_dn = np.full((n, nd), np.nan)    # 下方多头清算密度
    out_gu = np.full(n, np.nan)          # 距离加权（上）
    out_gd = np.full(n, np.nan)
    liqL = np.zeros(n)                   # 本根 K 线估计清算量（相对 OI）
    liqS = np.zeros(n)
    prev_oi = np.nan
    for t in range(n):
        if new_seg[t] or np.isnan(prev_oi):
            L[:] = 0.0; S[:] = 0.0; sL = 1.0; sS = 1.0; TL = 0.0; TS = 0.0
            prev_oi = oi[t]
            continue
        if np.isnan(oi[t]) or np.isnan(close[t]):
            continue
        # 1) 清算触发
        il = int(np.floor((np.log(lo_p[t]) - grid0) / BIN))
        ih = int(np.ceil((np.log(hi_p[t]) - grid0) / BIN))
        il = max(il, 0); ih = min(ih, nbins - 1)
        ql = 0.0
        for k in range(il, nbins):
            ql += L[k]; L[k] = 0.0
        ql *= sL
        qs = 0.0
        for k in range(0, ih + 1):
            qs += S[k]; S[k] = 0.0
        qs *= sS
        TL = max(TL - ql, 0.0); TS = max(TS - qs, 0.0)
        cur = max(oi[t], 1e-12)
        liqL[t] = ql / cur
        liqS[t] = qs / cur
        b = buy_frac[t]
        if np.isnan(b):
            b = 0.5
        d_oi = oi[t] - prev_oi
        prev_oi = oi[t]
        # 2) 平仓（扣除已被清算解释的部分）+ 换手
        tot = TL + TS
        closed = max(-d_oi - ql - qs, 0.0)
        ch = 0.0
        if tot > 0:
            ch = min(churn * vol[t] / cur, 0.5) * tot
        cl_L = closed * (1 - b) + ch * 0.5
        cl_S = closed * b + ch * 0.5
        if TL > 0:
            f = max(1.0 - cl_L / TL, 0.0)
            sL *= f; TL *= f
        if TS > 0:
            f = max(1.0 - cl_S / TS, 0.0)
            sS *= f; TS *= f
        if sL < 1e-6:
            L *= sL; sL = 1.0
        if sS < 1e-6:
            S *= sS; sS = 1.0
        # 3) 新开仓（净增 OI + 换手重新开出）
        add = max(d_oi, 0.0) + ch
        if add > 0:
            ic = int(np.round((np.log(vwap[t]) - grid0) / BIN))
            aL = add * b
            aS = add * (1 - b)
            for j in range(len(w)):
                kL = ic + lo_off[j]
                kS = ic + so_off[j]
                if 0 <= kL < nbins:
                    L[kL] += aL * w[j] / sL
                if 0 <= kS < nbins:
                    S[kS] += aS * w[j] / sS
            TL += aL; TS += aS
        # 4) 整点输出
        if is_hour_end[t]:
            ip = int(np.round((np.log(close[t]) - grid0) / BIN))
            for di in range(nd):
                db = dists_bins[di]
                su = 0.0
                for k in range(ip + 1, min(ip + db + 1, nbins)):
                    su += S[k]
                sd = 0.0
                for k in range(max(ip - dists_bins_dn[di], 0), ip):
                    sd += L[k]
                out_up[t, di] = su * sS / cur
                out_dn[t, di] = sd * sL / cur
            gu = 0.0; gd = 0.0
            for k in range(ip + 1, min(ip + 76, nbins)):
                gu += S[k] * np.exp(-(k - ip) / 15.0)   # 衰减尺度 3%
            for k in range(max(ip - 75, 0), ip):
                gd += L[k] * np.exp(-(ip - k) / 15.0)
            out_gu[t] = gu * sS / cur
            out_gd[t] = gd * sL / cur
    return out_up, out_dn, out_gu, out_gd, liqL, liqS


def load_symbol(sym):
    kl = pl.read_parquet(f"{DATA}/kl5m/{sym}.parquet").with_columns(
        pl.from_epoch("open_time", time_unit="ms").alias("t"))
    mt = pl.read_parquet(f"{DATA}/metrics/{sym}.parquet").with_columns(
        pl.col("ts").dt.truncate("5m")).unique("ts", keep="last")
    # OI 快照时间戳 = K 线收盘时刻 (t + 5m)
    kl = kl.with_columns((pl.col("t") + pl.duration(minutes=5)).alias("t_end"))
    df = kl.join(mt.select(pl.col("ts").alias("t_end"), "oi", "oi_value", "top_ls_pos", "ls_acct"),
                 on="t_end", how="left").sort("t")
    return df


def run_symbol(sym, lev=LEV, lev_w=LEV_W, churn=0.05):
    df = load_symbol(sym)
    if df.height < 2000:
        return None
    t = df["t"].to_numpy()
    gap = np.diff(t.astype("datetime64[m]").astype(np.int64), prepend=-10**9) > 60  # 超过 1 小时缺口 → 重置
    oi = df["oi"].to_numpy().astype(np.float64)
    # 数据清洗：OI<=0 是缺失；5 分钟内变化超过 ±30% 视为异常点
    oi = np.where(oi > 0, oi, np.nan)
    lr = np.abs(np.diff(np.log(oi), prepend=np.nan))
    lr_next = np.abs(np.diff(np.log(oi), append=np.nan))
    oi = np.where((lr > 0.3) & (lr_next > 0.3), np.nan, oi)
    # OI 缺失连续超过 1 小时也重置；短缺失前向填充
    oi_s = pl.Series(oi).fill_null(strategy="forward", limit=12).to_numpy()
    valid = ~np.isnan(oi_s)
    newseg = gap | (valid & ~np.roll(valid, 1))
    vol = df["volume"].to_numpy()
    vwap = np.where(vol > 0, df["quote_volume"].to_numpy() / np.maximum(vol, 1e-12), df["close"].to_numpy())
    buy = np.where(vol > 0, df["taker_buy_volume"].to_numpy() / np.maximum(vol, 1e-12), 0.5)
    hour_end = (df["t_end"].dt.minute() == 0).to_numpy()
    lp = np.log(np.concatenate([df["low"].to_numpy(), df["high"].to_numpy()]))
    grid0 = lp.min() - 1.0
    nbins = int((lp.max() + 1.0 - grid0) / BIN) + 2
    lo_off, so_off, w = liq_offsets(lev, lev_w)
    db = np.round(np.log(1 + DISTS) / BIN).astype(np.int64)
    dbd = np.round(-np.log(1 - DISTS) / BIN).astype(np.int64)
    up, dn, gu, gd, liqL, liqS = simulate(
        df["low"].to_numpy(), df["high"].to_numpy(), vwap, df["close"].to_numpy(), vol, buy, oi_s,
        newseg, hour_end, lo_off, so_off, w, grid0, nbins, churn, db, dbd)
    out = df.select("t_end", "close", "volume", "quote_volume", "taker_buy_volume", "oi", "oi_value",
                    "top_ls_pos", "ls_acct").with_columns(
        pl.Series("liqL", liqL), pl.Series("liqS", liqS),
        pl.Series("gup", gu), pl.Series("gdn", gd),
        *[pl.Series(f"up{int(d*100)}", up[:, i]) for i, d in enumerate(DISTS)],
        *[pl.Series(f"dn{int(d*100)}", dn[:, i]) for i, d in enumerate(DISTS)],
    )
    return out.with_columns(pl.lit(sym).alias("symbol"))
