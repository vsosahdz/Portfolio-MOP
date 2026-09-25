# Reproducibility appendix

## Data

- **vintage**: `2026-08-22`
- **evaluation window**: `2025-01 to 2026-06`
- **evaluation blocks**: `18`
- **download window**: `2023-01-01 to 2026-06-30`
- **risk free series**: `cetes_28d`
- **banxico series**:
  - cetes 28d: `SF43936`
  - fx fix: `SF43718`

## Universe

- **monthly sizes**:
  - months: `18`
  - minimum: `88`
  - median: `93`
  - maximum: `101`
- **screen thresholds**:
  - trailing months: `3`
  - min traded day fraction: `0.9`
  - min median traded value mxn: `500000.0`
  - max stale run days: `5`
  - min universe size: `60`
  - require full window coverage: `True`
  - min window coverage fraction: `0.95`
  - max edge gap days: `7`
  - max zero return fraction: `0.5`
  - min month median volume: `1.0`
  - max abs daily return: `0.5`

## Features

- **dictionary version**: `1.0.0`
- **daily indicators**: `43`
- **monthly features**: `129`

## Screening

- **screeners**: `all_stocks, svm, random_forest, neural_network, xgboost, tabpfn`
- **labeling strategies**: `historical, t_plus_1, historical_and_t1`
- **inner metric**: `balanced_accuracy`
- **inner folds**: `3`

## Optimisation

- **algorithm**: `NSGA-III`
- **population size**: `100`
- **generations**: `100`
- **objective evaluations**: `10000`
- **seeds**: `10`
- **weight caps**: `1.0, 0.2`
- **covariance estimator**: `sample`
- **covariance trailing days**: `252`

## Evaluation

- **cost scenarios bps**: `0.0, 25.0, 50.0, 100.0`
- **primary cost scenario bps**: `50.0`
- **realised turnover mean**: `0.6090362827909316`

## Environment

- **python**: `3.12.13 (main, Mar  3 2026, 12:39:30) [Clang 21.0.0 (clang-2100.0.123.102)]`
- **platform**:
  - cpu count: `12`
  - machine: `arm64`
  - processor: `arm`
  - release: `25.3.0`
  - system: `Darwin`
  - thread env: `{'MKL_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}`
- **revision**: `3adbefc23d375a91bb5bcee180103405b4519774+dirty`
- **config fingerprint**: `2658d460ef0feacc`
- **packages**:
  - PBC4cip: `0.0.0.8`
  - PyYAML: `6.0.3`
  - matplotlib: `3.11.1`
  - numpy: `2.5.1`
  - pandas: `3.0.5`
  - pyarrow: `25.0.0`
  - pymoo: `0.6.2`
  - scikit-learn: `1.9.0`
  - scipy: `1.18.0`
  - statsmodels: `0.14.6`
  - yfinance: `1.5.2`

## Substitutions

- **pbc4cip** — PBC4cip reference implementation with a compatibility patch (does not affect results)
  - Reason: replaced np.object with the builtin object in PBC4cip.core.Helpers.combine_instances and .convert_to_ndarray (np.object was removed in NumPy 1.24; it was an alias for object)

## Known limitations

### Survivorship exposure

The universe is screened point-in-time from the price history, not from an exchange listing record, because the exchange's listing endpoints were unavailable and no archived snapshot exists. No instrument in the evaluation window ceased trading, and 36 were excluded as new listings. That measurement is a lower bound on the exposure, not a proof of its absence: an instrument delisted before the download date cannot appear in the retrieved history at all.

### Evaluation block count

Conclusions rest on 18 monthly blocks. The minimum detectable effect at that size is 13 to 17 percent annualised, so every reported non-rejection is a statement about the design's resolution and not evidence of equivalence. Effects below that bound are reported as unresolved rather than as null.

### Post-hoc transaction costs

Costs are applied after optimisation, not inside the objective. The optimiser therefore never trades off return against the cost of achieving it, and the reported net figures understate what a cost-aware formulation could reach. Realised turnover averages 61% per month, so the gap is material rather than academic.

### Flat spread assumption

Every instrument is charged the same round-trip cost, at 50 basis points in the primary scenario. Real spreads on this exchange widen for the smaller and less liquid names, which the screeners select more often than the no-screening control does. The assumption therefore flatters the screened arms relative to the control, and the direction of that bias runs against the study's own conclusion rather than toward it.
