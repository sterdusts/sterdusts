"""关卡 1：在开发期（2022-01 ~ 2026-03）检验 H1 磁吸 和 H2 级联反转 两组因子。"""
import sys
sys.path.insert(0, "/home/user/sterdusts/liqmap")
import polars as pl
from evaluate import make_factors, evaluate_factor, fmt, n_trials

tag = sys.argv[1] if len(sys.argv) > 1 else "base"
p = make_factors(pl.read_parquet(f"/home/user/data/panel_{tag}.parquet"))
print("panel", p.shape, p["t"].min(), p["t"].max())
FACS = {
    "h1_imb2": "H1 磁吸：±2% 内 空头清算密度 - 多头清算密度 的不平衡",
    "h1_imb5": "H1 磁吸：±5% 不平衡",
    "h1_imb10": "H1 磁吸：±10% 不平衡",
    "h1_gimb": "H1 磁吸：距离加权（3% 衰减）不平衡",
    "h1_diff5": "H1 磁吸：±5% 密度差（按 OI 归一）",
    "h2_net1h": "H2 级联反转：过去 1h 多头清算 - 空头清算（按 OI）",
    "h2_net4h": "H2 级联反转：过去 4h",
    "h2_net24h": "H2 级联反转：过去 24h",
    "h2_net4h_rel": "H2 级联反转：4h 净清算 / 30 日平均清算",
    "fragility5": "辅助：±5% 内总清算密度（脆弱度）",
}
for f, d in FACS.items():
    r = evaluate_factor(p, f, f"[{tag}] {d}")
    print(f"\n{f}  —  {d}\n{fmt(r)}")
print("\n累计登记试验数:", n_trials())
