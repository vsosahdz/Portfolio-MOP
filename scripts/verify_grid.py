"""Completeness, skip-reason and level checks over the executed grid (tasks 9.5-9.7).

Run before any analysis is trusted. Completeness is delegated to `completeness_report`,
which separates a shortfall that carries a skip from one that carries nothing: a count
mismatch alone cannot tell those apart, and the difference is the whole point of the check.

Writes results/grid_verification.json.
"""
import json
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import pandas as pd

from bmvport.backtest.results import completeness_report
from bmvport.config import load_config

config = load_config("configs/default.yaml")
results = pd.read_parquet("results/results_combined.parquet")
ablation = pd.read_csv("results/ablation_search_space.csv")
issues: list[str] = []

# --- 9.5 completeness ------------------------------------------------------------------
report = completeness_report(results, config)
months = sorted(results["month"].unique())
print(f"9.5 COMPLETENESS  cells={report['cells']}  complete={report['complete']}  "
      f"short with skip={len(report['short_with_recorded_skip'])}  "
      f"short without skip={len(report['short_without_any_skip'])}\n")
for record in report["short_with_recorded_skip"]:
    print(f"  {record['labeling']}/{record['screener']} {record['month']} "
          f"cap={record['weight_cap']}: {record['present']}/{record['expected']} "
          f"-> {record['skip_reasons']}")
if not report["passes"]:
    issues.append(f"{len(report['short_without_any_skip'])} cells short with no skip recorded")

expected_ablation = len(config.ablation.selection_sizes) * len(months) * config.ablation.seeds
print(f"\n  ablation instances: {len(ablation)}/{expected_ablation}")
if len(ablation) != expected_ablation:
    issues.append(f"ablation: {expected_ablation - len(ablation)} instances missing")

comparators = results[results.arm_kind == "comparator"]
print(f"  comparator arms: {comparators.allocator.nunique()} "
      f"({', '.join(sorted(comparators.allocator.unique()))})")
per_arm = comparators.groupby("allocator")["month"].nunique()
for arm, count in per_arm[per_arm < len(months)].items():
    print(f"    {arm}: {count}/{len(months)} months")

# --- 9.6 skip reasons ------------------------------------------------------------------
skipped = results[results.skip_reason.astype(bool)]
print(f"\n9.6 SKIPS  {len(skipped):,} of {len(results):,} rows "
      f"({len(skipped) / len(results):.2%})\n")
reasons = {}
if len(skipped):
    print(f"  {'reason':56s}{'rows':>8}{'months':>8}")
    for reason, group in skipped.groupby(skipped.skip_reason.astype(str)):
        print(f"  {reason[:56]:56s}{len(group):8,}{group.month.nunique():8d}")
        reasons[reason] = int(len(group))
    # A skip is legitimate only when it names an infeasibility, never a failure.
    DEFECT = ("failed", "error", "exception", "traceback", "nan")
    suspect = [r for r in reasons if any(d in r.lower() for d in DEFECT)]
    if suspect:
        issues.append(f"skips suggesting a defect rather than infeasibility: {suspect}")
    print(f"\n  classified as genuine infeasibility: {len(reasons) - len(suspect)}/{len(reasons)}")

# --- 9.7 realised levels ---------------------------------------------------------------
evaluated = results[
    (results.arm_kind == "proposed") & (~results.skip_reason.astype(bool))
    & (results.cost_bps == config.evaluation.primary_cost_scenario_bps)]
print("\n9.7 REALISED LEVELS\n")
print(f"  {'screener':16s}{'selection':>12}{'holdings':>11}{'turnover':>11}")
for screener, group in evaluated.groupby("screener"):
    print(f"  {screener:16s}{group.selection_size.mean():11.1f}{group.holdings.mean():11.1f}"
          f"{group.turnover.mean():10.1%}")
universe = evaluated.groupby("month").selection_size.max()
print(f"\n  universe per month: {universe.min()}-{universe.max()} instruments")
print(f"  turnover: mean {evaluated.turnover.mean():.1%}, max {evaluated.turnover.max():.1%}")
if not (0 <= evaluated.turnover.min() and evaluated.turnover.max() <= 2.0):
    issues.append(f"turnover outside [0,2]: max={evaluated.turnover.max():.3f}")
if evaluated.selection_size.max() > universe.max():
    issues.append("selection exceeds the month's universe")
if not (60 <= universe.min() and universe.max() <= 120):
    issues.append(f"universe outside the expected 60-120: {universe.min()}-{universe.max()}")

json.dump(
    {"completeness": report, "skips": reasons,
     "levels": {"universe_min": int(universe.min()), "universe_max": int(universe.max()),
                "turnover_mean": float(evaluated.turnover.mean()),
                "turnover_max": float(evaluated.turnover.max())},
     "issues": issues},
    open("results/grid_verification.json", "w"), indent=2, default=str)
print(f"\n{'=' * 64}\nISSUES: {len(issues)}")
for issue in issues:
    print(f"  - {issue}")
