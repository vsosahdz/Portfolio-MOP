"""Compare the shrinkage arm against the sample-covariance conclusions (D34).

The comparison is within a contrast and never across: whether the estimator changes anything
is a question about the same contrast under two estimators. Writes
results/shrinkage_sensitivity.json. Run after run_shrinkage_sensitivity.py.
"""
import json
import sys
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import pandas as pd

from bmvport.config import load_config

config = load_config("configs/default.yaml")
COST = config.evaluation.primary_cost_scenario_bps
SCREENERS = ["all_stocks", "svm", "random_forest"]


def prepare(path: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    return frame[
        (frame.arm_kind == "proposed") & (frame.cost_bps == COST)
        & (~frame.skip_reason.astype(bool)) & (frame.labeling == "historical_and_t1")
        & (frame.screener.isin(SCREENERS))
    ]


sample = prepare("results/results_combined.parquet")
shrunk = prepare("results/sensitivity_shrinkage.parquet")
print(f"sample={len(sample):,} rows   shrinkage={len(shrunk):,} rows\n")
print(f"  {'screener':16s}{'sample':>10}{'shrinkage':>11}{'change':>9}")
for screener in SCREENERS:
    a = sample[sample.screener == screener].groupby("month").return_net.median().mean() * 1200
    b = shrunk[shrunk.screener == screener].groupby("month").return_net.median().mean() * 1200
    print(f"  {screener:16s}{a:9.1f}%{b:10.1f}%{b - a:+8.1f}%")

records = []
for name, frame in (("muestra", sample), ("shrinkage", shrunk)):
    pivot = frame.pivot_table(
        index="month", columns="screener", values="return_net", aggfunc="mean")
    for screener in ["svm", "random_forest"]:
        difference = (pivot[screener] - pivot["all_stocks"]).dropna()
        records.append({
            "arm": name, "contrast": f"{screener} - all_stocks",
            "effect": float(difference.mean()) * 1200,
            "sign": "negative" if difference.mean() < 0 else "POSITIVE",
            "months_negative": int((difference < 0).sum()), "n": int(difference.size),
        })

table = pd.DataFrame(records)
print(f"\n  {'arm':12s}{'contrast':28s}{'annual effect':>14}{'sign':>11}{'months<0':>10}")
for _, row in table.iterrows():
    print(f"  {row['arm']:12s}{row['contrast']:28s}{row['effect']:13.1f}%"
          f"{row['sign']:>11}{row['months_negative']:6d}/{row['n']}")
reverses = table.groupby("contrast")["sign"].nunique().gt(1)
print(f"\n  sign reversals: {int(reverses.sum())} of {len(reverses)} contrasts")
json.dump({"reverses": bool(reverses.any()), "contrasts": table.to_dict("records")},
          open("results/shrinkage_sensitivity.json", "w"), indent=2)
