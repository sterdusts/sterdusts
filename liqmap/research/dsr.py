"""紧缩夏普比率（Bailey & López de Prado 2014）：考虑试验次数后，真实夏普 > 门槛的概率。"""
import json, sys
import numpy as np
from scipy.stats import norm, skew, kurtosis

def trial_sharpes(path):
    out = []
    for line in open(path):
        r = json.loads(line)["result"]
        vals = []
        def walk(d):
            if isinstance(d, dict):
                for k, v in d.items():
                    if k == "sharpe" and isinstance(v, (int, float)):
                        vals.append(v)
                    else:
                        walk(v)
        walk(r)
        if vals:
            out.append(max(vals))
    return np.array(out)

def dsr(r, n_trials, sr_trials_ann, ppy, sr_bench_ann=0.0):
    r = np.asarray(r)
    sr = r.mean() / r.std(ddof=1)
    g3, g4 = skew(r), kurtosis(r, fisher=False)
    v = np.var(sr_trials_ann, ddof=1) / ppy
    em = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - em) * norm.ppf(1 - 1 / n_trials) + em * norm.ppf(1 - 1 / (n_trials * np.e)))
    sr0 = max(sr0, sr_bench_ann / np.sqrt(ppy))
    z = (sr - sr0) * np.sqrt(len(r) - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return {"sr_ann": sr * np.sqrt(ppy), "sr0_ann": sr0 * np.sqrt(ppy), "skew": g3, "kurt": g4, "prob": norm.cdf(z)}
