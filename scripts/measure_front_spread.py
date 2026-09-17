"""Measure front spread and concentration against problem size (D33).

Separates two things that look alike in the objective plot: how wide the front is, and how
concentrated the portfolios on it are. The spread collapses with dimension while the
concentration ratio against equal weight stays near two and a half, which is what says the
optimiser is still finding structure and it is the span that fails.

Writes results/front_spread_check.json.
"""
import json
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from bmvport.config import load_config
from bmvport.grid import load_grid_inputs
from bmvport.optimization.front import extract_portfolios
from bmvport.optimization.nsga import derive_seed, run_nsga3
from bmvport.optimization.objectives import estimate_moments
from bmvport.screening.selection import run_screening

config = load_config("configs/default.yaml")
inputs = load_grid_inputs(config)
LABELING = "historical_and_t1"

screening = run_screening(
    inputs.features, config, labeling=LABELING, screener="svm",
    membership=inputs.membership)
live = sorted([s for s in screening.selections if not s.skipped], key=lambda s: s.month)
selection = live[len(live) // 2]
month = selection.month
history = inputs.daily_returns[inputs.daily_returns.index < pd.Timestamp(f"{month}-01")]
pool = [s for s in inputs.membership.get(month, ()) if s in history.columns]

print(f"month={month}  universe={len(pool)}\n")
print(f"  {'n':>5}{'front':>8}{'ret range':>12}{'risk range':>13}"
      f"{'max weight':>12}{'eff names':>12}{'HHI':>9}")
out = []
for size in (10, 20, 40, 80, len(pool)):
    names = pool[:size]
    moments = estimate_moments(
        history, names, trailing_days=config.optimization.covariance_trailing_days,
        estimator=config.optimization.covariance_estimator)
    moments.validate()
    seed = derive_seed(config.fingerprint(), month, LABELING, "spread", 1.0, size)
    front, _ = run_nsga3(
        moments, seed=seed, population_size=config.optimization.population_size,
        generations=config.optimization.generations, cap=1.0)
    returns = -front.objectives[:, 0]
    risks = front.objectives[:, 1]
    extracted = list(extract_portfolios(front.weights, front.objectives))
    record = {
        "n": size, "front": int(front.size),
        "ret_range": float(returns.max() - returns.min()),
        "risk_range": float(risks.max() - risks.min()),
        "max_weight": float(np.max([p.weights.max() for p in extracted])),
        "equal_weight": 1.0 / size,
        "effective_names": float(np.mean([(p.weights > 1e-4).sum() for p in extracted])),
        "hhi": float(np.mean([(p.weights ** 2).sum() for p in extracted])),
        "hhi_equal": 1.0 / size,
    }
    out.append(record)
    print(f"  {size:5d}{record['front']:8d}{record['ret_range'] * 100:11.2f}%"
          f"{record['risk_range'] * 100:12.2f}%{record['max_weight'] * 100:11.1f}%"
          f"{record['effective_names']:12.1f}{record['hhi']:9.4f}")

print(f"\n  {'n':>5}{'max weight / equal':>21}{'HHI / HHI equal':>18}")
for record in out:
    print(f"  {record['n']:5d}{record['max_weight'] / record['equal_weight']:20.2f}x"
          f"{record['hhi'] / record['hhi_equal']:17.2f}x")
json.dump(out, open("results/front_spread_check.json", "w"), indent=2)
