"""MT1' G0 预检：只用信号与控制变量，不读任何标签（不打开 perp_daily 的 fwd_/fund_ 列）。

判据：线性概率模型 state ~ X + seg 的 R² > 0.50 → 放弃 MT1'（不占试验）。
运行：python3 -I g0.py
"""
import datetime as dt
import json
import pathlib

import numpy as np
import polars as pl
import statsmodels.api as sm

RUN = pathlib.Path("/home/user/data/flow/MT1P-stable-issuance-v2_run")
SIG = RUN / "mkt_timing_signals_v2.parquet"
V1 = pathlib.Path("/home/user/data/flow/mkt_timing_prereg/mkt_timing_signals.parquet")
OUT = RUN / "g0_result.json"
DEV0, DEV_END, MAIN_END = dt.date(2022, 1, 1), dt.date(2026, 3, 31), dt.date(2026, 3, 17)
XCOLS = ["c_mom7", "c_mom30", "c_rv30", "c_fund7", "c_doi7", "c_tkr7", "fund30", "P_I", "mom90", "ma200", "dvol30"]


def runs(x: np.ndarray):
    """返回 [(value, start_idx, length)]。"""
    out, s = [], 0
    for i in range(1, len(x) + 1):
        if i == len(x) or x[i] != x[s]:
            out.append((x[s], s, i - s))
            s = i
    return out


def design(df: pl.DataFrame):
    X = df.select(XCOLS).to_numpy().astype(float)
    segs = sorted(df["seg"].unique().to_list())
    D = np.column_stack([(df["seg"] == s).to_numpy().astype(float) for s in segs[1:]])
    return np.column_stack([np.ones(len(df)), X, D])


def auc(score, y):
    order = np.argsort(score)
    r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
    # 处理并列
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stats_for(df: pl.DataFrame, label: str):
    st = df["mt1_state"].to_numpy().astype(float)
    y01 = (st > 0).astype(float)
    Z = design(df)
    lpm = sm.OLS(y01, Z).fit()
    lg = sm.Logit(y01, Z).fit(disp=0, maxiter=200)
    return {"sample": label, "n": int(len(df)), "lpm_r2": float(lpm.rsquared), "lpm_adj_r2": float(lpm.rsquared_adj),
            "logit_auc_in_sample": auc(lg.predict(Z), y01.astype(int)), "logit_pseudo_r2": float(lg.prsquared)}


def main():
    assert not OUT.exists()
    s = pl.read_parquet(SIG).filter(pl.col("date").is_between(DEV0, DEV_END))
    assert s["date"].max() <= DEV_END
    v1 = (pl.read_parquet(V1, columns=["date", "mt1_state", "mt1_I"])
          .filter(pl.col("date").is_between(DEV0, DEV_END))
          .rename({"mt1_state": "v1_state", "mt1_I": "v1_I"}))
    j = s.join(v1, on="date", how="left")
    res = {"spec_id": "MT1P-stable-issuance-v2", "x_cols": XCOLS, "seg_fe": "2022/2023/2024/2025-01..2026-03"}
    for lo, hi, nm in [(DEV0, DEV_END, "dev_2022-01-01..2026-03-31"), (DEV0, MAIN_END, "main_2022-01-01..2026-03-17")]:
        q = j.filter(pl.col("date").is_between(lo, hi))
        a = q.filter(pl.col("v1_state").is_not_null() & pl.col("mt1_state").is_not_null())
        res[f"agree_v2_v1_{nm}"] = float((a["mt1_state"] == a["v1_state"]).mean())
        res[f"disagree_days_{nm}"] = int((a["mt1_state"] != a["v1_state"]).sum())
    # 切换/闪变/长段（开发期 2022-01-01..2026-03-31）
    x = s["mt1_state"].to_numpy()
    assert not np.isnan(x.astype(float)).any()
    rr = runs(x)
    sw = int((x[1:] != x[:-1]).sum())
    years = (DEV_END - DEV0).days / 365.25 + 1 / 365.25
    flick = [r for k, r in enumerate(rr) if r[2] == 1 and 0 < k < len(rr) - 1]
    res.update({
        "switches_dev": sw, "switches_per_year": sw / years,
        "switches_by_year": {str(y): int(((s["date"].dt.year().to_numpy()[1:] == y) & (x[1:] != x[:-1])).sum())
                             for y in range(2022, 2027)},
        "n_runs": len(rr), "flicker_1day_runs": len(flick),
        "flicker_dates": [str(s["date"][r[1]]) for r in flick],
        "runs_ge30": int(sum(1 for r in rr if r[2] >= 30)),
        "runs_ge14": int(sum(1 for r in rr if r[2] >= 14)),
        "plus1_frac_dev": float((x > 0).mean()),
        "plus1_days_in_runs_ge14_frac": float(sum(r[2] for r in rr if r[0] > 0 and r[2] >= 14) / max(1, (x > 0).sum())),
        "median_run_len": float(np.median([r[2] for r in rr])),
    })
    # R² / AUC
    main = s.filter(pl.col("date").is_between(DEV0, MAIN_END))
    assert main.select(XCOLS + ["mt1_state", "seg"]).null_count().sum_horizontal().item() == 0
    res["r2_main"] = stats_for(main, "main_2022-01-01..2026-03-17")
    res["r2_dev"] = stats_for(s.drop_nulls(XCOLS + ["mt1_state"]), "dev_2022-01-01..2026-03-31")
    # 同表：其他状态与主状态的一致率（只用特征）
    for c in ["state_USDT", "state_USDC", "state_cry", "state_ext", "state_causal", "state_3c", "state_P", "state_SV",
              "state_OI"]:
        q = main.filter(pl.col(c).is_not_null())
        res[f"agree_mt1_{c}_main"] = float((q[c] == q["mt1_state"]).mean())
    r2max = max(res["r2_main"]["lpm_r2"], res["r2_dev"]["lpm_r2"])
    res["G0_rule"] = "LPM R² > 0.50 → 放弃（不占试验）；保守取两个样本中较大的 R²"
    res["G0_r2_used"] = r2max
    res["G0_pass"] = bool(r2max <= 0.50)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
