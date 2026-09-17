# Portfolio selection on the Mexican Stock Exchange (BMV)

Replication package for a study combining supervised machine-learning stock screening with
multi-objective portfolio optimisation (NSGA-III) on the Mexican Stock Exchange, evaluated
over January 2025 – June 2026.

Authors, in order: Víctor Adrián Sosa Hernández (first, corresponding), Raúl Monroy
Borja, Samuel Guadalupe Rodríguez Rodríguez.

The methodology derives from Chapter 3 of Rodríguez Rodríguez's 2022 MSc thesis
(Tecnológico de Monterrey). **This is not a replication of that work**: the evaluation
window, universe construction, evaluation protocol and statistical treatment differ by
design, and no comparison against the 2022 result tables is made.

The specification, the design decisions and the record of what was tried and rejected are
kept in a separate repository held by the authors. Everything needed to reproduce the
published results is here.

## Environment

Python 3.12 exactly. Dependency versions are pinned in `pyproject.toml`.

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

The repository's `python3` on the development machine may resolve to an unrelated
virtualenv; always invoke `.venv/bin/python` explicitly.

## Layout

```
src/bmvport/          pipeline package
  config.py             resolved run configuration (single source of truth)
  manifest.py           run manifest: data vintage, versions, seeds
  marketdata/           universe, prices, quality gates, rates
  features/             43 daily indicators -> 129 monthly features
  screening/            labeling, screeners, model selection
  optimization/         objectives, NSGA-III, fronts, baselines
  backtest/             executable pricing, costs, metrics, attribution
  benchmarks/           three-tier comparator suite
  ablation/             selection-size sweep and search instrumentation
  stats/                mixed model, nonparametric tests, bootstrap
  reporting/            manuscript tables and figures
configs/              default.yaml (full study), smoke.yaml (fast end-to-end check)
scripts/              the run of record, the supporting experiments, the checks
data/                 cache (dataset of record), universe, features, checksums
results/              run of record, statistics, ablation sweeps, verification
paper/                tables, figures, manuscript sections, appendix, front matter
tests/                unit and invariant tests
```

## Data of record

Prices are retrieved once and cached under `data/cache/`, and that cache is committed to this
repository so the study reproduces without contacting any provider. It is redistributed to
support replication and carries its providers' terms rather than this package's licence;
`data/checksums.json` records the digests so a reader who obtains the series independently
can verify they match.

Every downstream stage reads only from the cache, so no stage contacts a provider. The
retrieval date is the study's **data vintage** and is recorded in every run manifest.

## Running the study

Scripts are run from the repository root, in this order. The grid dominates the wall-clock;
everything after it is minutes.

```bash
.venv/bin/python scripts/run_grid.py                    # ~150 min, the run of record
.venv/bin/python scripts/run_comparators.py             # comparator tiers, combines with the grid
.venv/bin/python scripts/verify_grid.py                 # completeness, skips, levels — must report 0 issues
.venv/bin/python scripts/regenerate_reports.py          # manifest, statistics, tables, figures, manuscript, appendix
```

The supporting experiments are independent of each other and of the order above:

```bash
.venv/bin/python scripts/run_shrinkage_sensitivity.py   # covariance-estimator sensitivity arm
.venv/bin/python scripts/compare_shrinkage_arm.py       # …and whether it reverses anything
.venv/bin/python scripts/run_reachability_sweep.py      # how much of the attainable range the search spans
.venv/bin/python scripts/measure_front_spread.py        # front width and concentration against problem size
.venv/bin/python scripts/make_pareto_figure.py          # one front, drawn from a rule-chosen instance
.venv/bin/python scripts/check_offline.py               # proves the cache-only claim with DNS blocked
```

## Reproducibility

The offline claim is tested rather than asserted: with DNS resolution blocked, the smoke
grid runs end to end from the cache and the reporting pipeline regenerates every artefact.

Same cache + same configuration + same seeds produces identical outputs. Every number in the
manuscript is generated into `paper/tables/`, `paper/figures/` or `paper/manuscript_facts.json`;
none is entered by hand, and a test enforces that the section templates contain no digits.

Seeds are derived from the configuration fingerprint, so a configuration change reseeds the
genetic algorithm and the grid must be rerun. The manifest records the fingerprint the
results were produced under; if it disagrees with the shipped configuration, the results
predate it and are not the ones the package regenerates, and `tests/` fails rather than
letting that pass silently. This is checked because it happened once during development.
