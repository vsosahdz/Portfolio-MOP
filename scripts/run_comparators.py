"""Run the three comparator tiers and combine them with the grid.

Writes results/comparators_rerun.parquet and results/results_combined_rerun.parquet.
"""
import sys, time, json, warnings; sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import pandas as pd
from bmvport.config import load_config
from bmvport.grid import load_grid_inputs
from bmvport.benchmarks.runner import run_comparators
from bmvport.backtest.results import build_results_table, check_key_uniqueness
cfg=load_config("configs/default.yaml"); t0=time.time()
inp=load_grid_inputs(cfg)
rows,fail=run_comparators(cfg,inp,progress=True)
df=build_results_table(rows)
df.to_parquet("results/comparators_rerun.parquet",index=False)
print(f"comparadores: {len(df):,} filas, {len(fail)} fallos, {(time.time()-t0)/60:.1f} min",flush=True)
grid=pd.read_parquet("results/full_rerun.parquet")
comb=pd.concat([grid,df],ignore_index=True)
check_key_uniqueness(comb)
comb.to_parquet("results/results_combined_rerun.parquet",index=False)
print(f"combinado: {len(comb):,} filas  ({comb.arm_kind.value_counts().to_dict()})",flush=True)
