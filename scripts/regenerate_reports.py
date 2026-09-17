"""Regenerate every downstream artefact from the rerun, which is now the run of record.

The stored table was produced under a configuration whose fingerprint no longer matches the
shipped one, so the shipped package did not regenerate its own results. This rebuilds the
chain from the rerun: manifest, tests, tables, figures, manuscript and appendix, all from
one results file produced by the shipped code and configuration.
"""
import sys, json, dataclasses, warnings, shutil
sys.path.insert(0,"src"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from bmvport.config import load_config
from bmvport.manifest import build_manifest, write_manifest, Substitution
from bmvport.stats.inference import (confirmatory_tests, comparator_confrontation,
    factor_effects, mixed_model, hierarchical_friedman, critical_difference)
from bmvport.reporting.tables import write_tables
from bmvport.reporting.figures import write_figures, figure_critical_difference
from bmvport.reporting.appendix import write_appendix
from bmvport.reporting.front_matter import write_front_matter
from bmvport.reporting.manuscript import write_manuscript
from pathlib import Path

cfg=load_config("configs/default.yaml")
r=pd.read_parquet("results/results_combined_rerun.parquet")
print(f"resultados de referencia: {len(r):,} filas  fingerprint={cfg.fingerprint()}")

old=json.load(open("results/manifest.json"))
man=build_manifest(cfg, data_vintage=old["data_vintage"],
    substitutions=[Substitution(**s) for s in old["substitutions"]],
    notes=old["notes"])
write_manifest(man,"results")
print(f"manifiesto refrescado: fingerprint={json.load(open('results/manifest.json'))['config_fingerprint']}")

tests=confirmatory_tests(r,cfg)+comparator_confrontation(r,cfg)
json.dump([t.to_record() for t in tests],open("results/tests.json","w"),indent=2,default=str)
factor_effects(r,cfg).to_csv("results/factor_effects.csv",index=False)
mm=mixed_model(r,cfg); json.dump(mm,open("results/mixed_model.json","w"),indent=2,default=str)
fr=hierarchical_friedman(r,cfg); fr.to_csv("results/friedman.csv",index=False)
print(f"estadistica: {len(tests)} pruebas, modelo mixto ajustado={mm['fitted']}")

C=cfg.evaluation.primary_cost_scenario_bps
write_tables(r,tests,C,"paper/tables")
write_figures(r,C,"paper/figures",ablation=pd.read_csv("results/ablation_search_space.csv"))
figure_critical_difference(r,C,Path("paper/figures/fig_critical_difference.png"),
                           headline_set_size=cfg.stats.headline_set_size)
print("tablas y figuras regeneradas")

cd=pd.read_csv("paper/figures/fig_critical_difference.csv")
terms={t["term"]:t for t in mm["terms"]}
ratios=[c["std_error_monthly"]/terms[f"C(screener)[T.{c['level']}]"]["std_error_monthly"]
        for c in mm["dependence_check"]["contrasts"]
        if f"C(screener)[T.{c['level']}]" in terms]
art=dict(ablation=pd.read_csv("results/ablation_search_space.csv"),
    reachability=pd.read_csv("results/front_reachability_sweep.csv"),
    tests=json.load(open("results/tests.json")), friedman=fr,
    shrinkage=json.load(open("results/shrinkage_sensitivity.json")),
    mixed_model=mm, se_ratios=ratios, n_cell_months=mm["n_observations"],
    critical_difference=float(cd["critical_difference"].iloc[0]), headline_cells=len(cd))
write_manuscript(r,cfg,art,"paper")
write_appendix(r,cfg,json.load(open("results/manifest.json")),"paper")
write_front_matter("paper")
from bmvport.reporting.manuscript import render_sections, collect_facts, venue_limit_violations
violations=venue_limit_violations(render_sections(collect_facts(r,cfg,art)))
print(f"manuscrito, apendice y portada regenerados; limites de revista: "
      f"{violations if violations else 'ok'}")

shutil.copy("results/results_combined_rerun.parquet","results/results_combined.parquet")
print("results_combined.parquet reemplazado por la corrida de referencia")
