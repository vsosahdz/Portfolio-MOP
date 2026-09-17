"""Measure how much of the attainable objective range the optimiser spans (D33).

Attainment compares a front against a reference produced by the same optimiser at a larger
budget, so a bias the two share is invisible to it. Both extremes of the unconstrained
problem have a closed form -- maximum expected return is the whole budget in the
highest-mean asset, minimum variance is the analytic portfolio -- so the attainable range is
computable exactly and the front can be measured against it instead.

Writes results/front_reachability_sweep.csv.
"""
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from bmvport.config import load_config
from bmvport.grid import load_grid_inputs
from bmvport.optimization.baselines import minimum_volatility
from bmvport.optimization.nsga import derive_seed, run_nsga3
from bmvport.optimization.objectives import estimate_moments

config = load_config("configs/default.yaml")
inputs = load_grid_inputs(config)
months = sorted(
    pd.read_parquet("results/results_combined.parquet")["month"].unique())[::4][:5]

rows = []
for month in months:
    history = inputs.daily_returns[
        inputs.daily_returns.index < pd.Timestamp(f"{month}-01")]
    pool = [s for s in inputs.membership.get(month, ()) if s in history.columns]
    for size in (10, 20, 40, 80, len(pool)):
        names = pool[:size]
        try:
            moments = estimate_moments(
                history, names,
                trailing_days=config.optimization.covariance_trailing_days,
                estimator=config.optimization.covariance_estimator)
            moments.validate()
        except Exception:  # noqa: BLE001
            continue
        mu = np.asarray(moments.expected_returns, float)
        analytic = minimum_volatility(moments, cap=1.0)
        if analytic.failed:
            continue
        vertex = np.zeros(len(names))
        vertex[int(np.argmax(mu))] = 1.0
        return_span = abs(float(mu.max()) - float(analytic.weights @ mu))
        risk_span = abs(
            float(np.sqrt(vertex @ moments.covariance @ vertex))
            - float(np.sqrt(analytic.weights @ moments.covariance @ analytic.weights)))
        for replicate in range(3):
            seed = derive_seed(
                config.fingerprint(), month, "reach2", "x", 1.0, size * 10 + replicate)
            front, _ = run_nsga3(
                moments, seed=seed,
                population_size=config.optimization.population_size,
                generations=config.optimization.generations, cap=1.0)
            produced_return = -front.objectives[:, 0]
            produced_risk = front.objectives[:, 1]
            rows.append({
                "month": month, "n": size, "seed": replicate,
                "ret_cov": float(produced_return.max() - produced_return.min()) / return_span
                if return_span > 0 else np.nan,
                "risk_cov": float(produced_risk.max() - produced_risk.min()) / risk_span
                if risk_span > 0 else np.nan,
                "front": int(front.size),
            })

frame = pd.DataFrame(rows)
frame.to_csv("results/front_reachability_sweep.csv", index=False)
grouped = frame.groupby("n").agg(
    ret=("ret_cov", "mean"), ret_sd=("ret_cov", "std"),
    risk=("risk_cov", "mean"), risk_sd=("risk_cov", "std"),
    front=("front", "mean"), k=("ret_cov", "size"))
print(f"months={len(months)} seeds=3  instances={len(frame)}\n")
print(f"  {'n':>5}{'return coverage':>18}{'risk coverage':>18}{'front':>9}{'k':>5}")
for size, row in grouped.iterrows():
    print(f"  {size:5d}{row['ret']:13.1%} ±{row['ret_sd']:4.1%}"
          f"{row['risk']:13.1%} ±{row['risk_sd']:4.1%}{row['front']:9.0f}{int(row['k']):5d}")
for column, label in (("ret_cov", "return"), ("risk_cov", "risk")):
    rho, p_value = spearmanr(frame["n"], frame[column])
    print(f"\n  Spearman n vs {label} coverage: rho={rho:+.3f} p={p_value:.2e}")
