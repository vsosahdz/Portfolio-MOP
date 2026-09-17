"""Draw one Pareto front with its three extracted portfolios.

The instance is chosen by rule, not for how it looks: the median evaluation month of the
window, under the no-screening control and the unconstrained cap, at the base replicate. The
rule is recorded in the figure's metadata so a reader can regenerate exactly this front.

Writes paper/figures/fig_pareto_front.png, its data and its metadata.
"""
import json
import pathlib
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import pandas as pd

from bmvport.config import load_config
from bmvport.grid import load_grid_inputs
from bmvport.optimization.front import extract_portfolios, hypervolume
from bmvport.optimization.nsga import derive_seed, run_nsga3
from bmvport.optimization.objectives import estimate_moments
from bmvport.reporting.figures import figure_pareto_front
from bmvport.screening.selection import run_screening

LABELING, SCREENER, CAP = "historical_and_t1", "all_stocks", 1.0

config = load_config("configs/default.yaml")
inputs = load_grid_inputs(config)
screening = run_screening(
    inputs.features, config, labeling=LABELING, screener=SCREENER,
    membership=inputs.membership)
live = [s for s in screening.selections if not s.skipped]
selection = sorted(live, key=lambda s: s.month)[len(live) // 2]
month = selection.month

history = inputs.daily_returns[inputs.daily_returns.index < pd.Timestamp(f"{month}-01")]
usable = [s for s in selection.selected if s in history.columns]
moments = estimate_moments(
    history, usable, trailing_days=config.optimization.covariance_trailing_days,
    estimator=config.optimization.covariance_estimator)
moments.validate()
seed = derive_seed(config.fingerprint(), month, LABELING, SCREENER, CAP, 0)
front, _ = run_nsga3(
    moments, seed=seed, population_size=config.optimization.population_size,
    generations=config.optimization.generations, cap=CAP)
extracted = list(extract_portfolios(front.weights, front.objectives))
hv = hypervolume(front.objectives, config.optimization.hypervolume_reference)

path = figure_pareto_front(
    front.weights, front.objectives, extracted,
    pathlib.Path("paper/figures/fig_pareto_front.png"))
json.dump(
    {"month": month, "labeling": LABELING, "screener": SCREENER, "weight_cap": CAP,
     "seed": int(seed), "n_assets": len(usable), "front_size": int(front.size),
     "hypervolume": float(hv),
     "selection_rule": ("median evaluation month, no-screening control, "
                        "unconstrained cap, base replicate"),
     "extracted": [
         {"allocator": p.allocator,
          "expected_return_monthly": float(p.expected_return),
          "sigma_ex_ante_monthly": float(p.sigma_ex_ante),
          "max_weight": float(p.weights.max()),
          "effective_names": int((p.weights > 1e-4).sum())} for p in extracted]},
    open("paper/figures/fig_pareto_front_meta.json", "w"), indent=2)

print(f"month={month}  assets={len(usable)}  front={front.size}  HV={hv:.4f}  seed={seed}\n")
print(f"  {'portfolio':16s}{'monthly ret':>13}{'risk':>9}{'max weight':>12}{'names':>8}")
for portfolio in extracted:
    print(f"  {portfolio.allocator:16s}{portfolio.expected_return * 100:12.2f}%"
          f"{portfolio.sigma_ex_ante * 100:8.2f}%{portfolio.weights.max() * 100:11.1f}%"
          f"{int((portfolio.weights > 1e-4).sum()):8d}")
