"""Run the full factorial and write results/full_rerun.parquet.

This is the run of record (D35). Roughly 150 minutes on twelve cores.
"""
import sys, time, json, warnings
sys.path.insert(0, "src"); warnings.filterwarnings("ignore")
import pandas as pd
from bmvport.config import load_config
from bmvport.grid import load_grid_inputs, run_grid
from bmvport.backtest.results import build_results_table, check_key_uniqueness, KEY_FIELDS
cfg=load_config("configs/default.yaml")
t0=time.time(); inp=load_grid_inputs(cfg)
rows,fail=run_grid(cfg,inp,progress=True)
df=build_results_table(rows); check_key_uniqueness(df)
df.to_parquet("results/full_rerun.parquet",index=False)
mins=(time.time()-t0)/60
json.dump({"rows":len(df),"failures":len(fail),"minutes":round(mins,1)},
          open("results/full_rerun_summary.json","w"),indent=2)
print(f"malla lista: {len(df):,} filas, {len(fail)} fallos, {mins:.1f} min",flush=True)
