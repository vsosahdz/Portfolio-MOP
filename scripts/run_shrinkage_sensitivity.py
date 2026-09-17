"""Run the shrinkage-covariance sensitivity arm required by the benchmark spec (D34).

Reduced to one labeling, three seeds and the three screeners the screening conclusion turns
on: the no-screening control, and the best and worst machine-learning arms. The spec
constrains the allocators this arm must cover, not the screeners. Writes
results/sensitivity_shrinkage.parquet.
"""
import json
import sys
import time
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from bmvport.backtest.results import build_results_table, check_key_uniqueness
from bmvport.config import load_config
from bmvport.grid import load_grid_inputs, run_grid

SCREENERS = ["all_stocks", "svm", "random_forest"]
LABELING = "historical_and_t1"

config = load_config("configs/sensitivity_shrinkage.yaml")
assert config.optimization.covariance_estimator == "ledoit_wolf"
print(f"estimator={config.optimization.covariance_estimator} "
      f"seeds={config.optimization.seeds}", flush=True)

started = time.time()
inputs = load_grid_inputs(config)
rows, failures = run_grid(
    config, inputs, labelings=[LABELING], screeners=SCREENERS, progress=True)
frame = build_results_table(rows)
check_key_uniqueness(frame)
frame.to_parquet("results/sensitivity_shrinkage.parquet", index=False)

minutes = (time.time() - started) / 60
json.dump(
    {"rows": len(frame), "failures": len(failures), "minutes": round(minutes, 1),
     "estimator": "ledoit_wolf", "seeds": config.optimization.seeds,
     "labelings": [LABELING], "screeners": SCREENERS},
    open("results/sensitivity_shrinkage_summary.json", "w"), indent=2)
print(f"done: {len(frame):,} rows, {len(failures)} failures, {minutes:.1f} min", flush=True)
