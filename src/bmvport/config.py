"""Resolved run configuration.

Every stage of the pipeline reads its parameters from a single ``RunConfig`` instance
loaded from YAML. Nothing is hard-coded in the stage modules: a value that affects a
result must be visible here so that it can be recorded in the run manifest and reported
in the manuscript's reproducibility appendix.

The dataclasses are frozen. A stage that wants a variation gets a new config, it does not
mutate the one it was given -- otherwise the manifest would no longer describe the run.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

__all__ = [
    "RunConfig",
    "WindowConfig",
    "UniverseConfig",
    "FeatureConfig",
    "ScreeningConfig",
    "OptimizationConfig",
    "AblationConfig",
    "EvaluationConfig",
    "BenchmarkConfig",
    "StatsConfig",
    "PrimaryHypothesis",
    "PathsConfig",
    "load_config",
]

# Labeling strategy identifiers. Defined here rather than as free strings so that a typo
# in a config file fails at load time instead of silently producing an empty factor level.
LABELING_STRATEGIES = ("historical", "t_plus_1", "historical_and_t1")

# Screener identifiers, chosen as a taxonomy rather than an accumulation:
#   all_stocks       the no-screening baseline, classifying every candidate as positive
#   svm, random_forest, neural_network, pbc4cip
#                    the original study's set, kept for methodological fidelity. PBC4cip
#                    also earns its place empirically: the proposed labeling runs 4.7:1
#                    negative-to-positive on this window, which is what it was built for.
#   xgboost          modern gradient boosting, the standard strong tabular baseline.
#                    Measured here at balanced accuracy 0.689 against 0.646 for scikit-learn's
#                    histogram booster and 0.619 for CatBoost, so it carries the family alone
#                    rather than shipping three variants of the same idea.
#   tabpfn           a tabular foundation model, the 2026 state of the art in exactly the
#                    small-sample regime this study occupies. Gated on a licence token; see
#                    screening.screeners for the disclosure path when it is unavailable.
SCREENERS = (
    "all_stocks", "svm", "random_forest", "neural_network", "pbc4cip", "xgboost", "tabpfn",
)

# Allocator identifiers. The first three are extracted from an NSGA-III front; the last
# three are deterministic convex or closed-form solutions.
ALLOCATORS = (
    "ga_min_risk",
    "ga_knee",
    "ga_max_return",
    "equal_weight",
    "min_volatility",
    "max_sharpe",
)

GA_ALLOCATORS = ("ga_min_risk", "ga_knee", "ga_max_return")

INVESTOR_PROFILES = ("aggressive", "conservative", "moderate")

# Profile -> metric mapping, fixed by the study design.
PROFILE_METRIC = {
    "aggressive": "return_net",
    "conservative": "sigma_realized",
    "moderate": "sharpe",
}

COVARIANCE_ESTIMATORS = ("sample", "ledoit_wolf")

# Fewest positions the momentum comparator may hold and still function as a diversified
# baseline. Below this its month-to-month variance is driven by individual names, so losing
# to it -- or beating it -- says little about momentum as a strategy.
MIN_MOMENTUM_POSITIONS = 10

# Largest confirmatory family that still deserves the name. Beyond a handful of
# pre-specified tests, "confirmatory" stops meaning anything and the correction stops
# protecting anything.
MAX_PRIMARY_HYPOTHESES = 5

# Only factor contrasts may carry a confirmatory claim on this design. See PrimaryHypothesis
# for the measured reason: head-to-head arm comparisons cannot reject anything worth
# claiming at 18 blocks.
CONFIRMATORY_KINDS = frozenset({"factor_effect"})


class ConfigError(ValueError):
    """Raised when a configuration file is internally inconsistent."""


@dataclass(frozen=True, slots=True)
class WindowConfig:
    """Download and evaluation windows.

    The download window is necessarily wider than the evaluation window: indicator
    warm-up, the per-observation context requirement, the ``t+1`` label and the expanding
    training window all consume history before the first evaluable month.
    """

    download_start: dt.date
    download_end: dt.date
    eval_start_month: str  # "YYYY-MM", first evaluation month inclusive
    eval_end_month: str  # "YYYY-MM", last evaluation month inclusive

    def eval_months(self) -> tuple[str, ...]:
        """Return the evaluation months as ``YYYY-MM`` strings, ascending."""
        start_y, start_m = (int(p) for p in self.eval_start_month.split("-"))
        end_y, end_m = (int(p) for p in self.eval_end_month.split("-"))
        months: list[str] = []
        y, m = start_y, start_m
        while (y, m) <= (end_y, end_m):
            months.append(f"{y:04d}-{m:02d}")
            m += 1
            if m == 13:
                y, m = y + 1, 1
        return tuple(months)

    def __post_init__(self) -> None:
        if self.download_start >= self.download_end:
            raise ConfigError("download_start must precede download_end")
        months = self.eval_months()
        if not months:
            raise ConfigError("evaluation window contains no months")
        first_eval = dt.date(int(months[0][:4]), int(months[0][5:]), 1)
        if first_eval <= self.download_start:
            raise ConfigError(
                "evaluation window must start after download_start; indicator warm-up, "
                "label lag and the expanding training window need prior history"
            )


@dataclass(frozen=True, slots=True)
class UniverseConfig:
    """Monthly point-in-time universe screen.

    Thresholds are deliberately not given defaults that look authoritative: they are set
    from the observed distribution of the candidate universe (tasks 2.5) and recorded.

    ``require_full_window_coverage`` restricts the universe to instruments that traded
    throughout the study window. It removes incomplete series, and it also removes
    instruments that listed or stopped trading inside the window -- which means it
    *introduces* survivorship bias rather than mitigating it, because the instruments that
    stop trading are disproportionately the losers. When it is enabled, the count and
    identity of the excluded instruments MUST be recorded so the bias is quantified rather
    than merely declared.
    """

    trailing_months: int
    min_traded_day_fraction: float
    min_median_traded_value_mxn: float
    max_stale_run_days: int
    min_universe_size: int
    require_full_window_coverage: bool = False
    min_window_coverage_fraction: float = 0.95
    max_edge_gap_days: int = 7
    # Per ticker-month quality gates. A day-count rule alone does not catch an instrument
    # that is listed and quoted but not actually traded: the exchange is open, prices are
    # published, and every return is zero. América Móvil's B series behaved exactly that way
    # through 2023 -- 117 consecutive sessions at a fixed price on a median volume of zero --
    # while the series was being created. Admitting such months as training observations
    # feeds the screeners feature vectors in which every momentum and ratio indicator has
    # collapsed to zero.
    max_zero_return_fraction: float = 0.50
    min_month_median_volume: float = 1.0
    max_abs_daily_return: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 < self.min_traded_day_fraction <= 1.0:
            raise ConfigError("min_traded_day_fraction must lie in (0, 1]")
        if self.trailing_months < 1:
            raise ConfigError("trailing_months must be at least 1")
        if self.min_universe_size < 2:
            raise ConfigError("a universe of fewer than 2 tickers cannot be diversified")
        if not 0.0 < self.min_window_coverage_fraction <= 1.0:
            raise ConfigError("min_window_coverage_fraction must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class FeatureConfig:
    """Technical-indicator and monthly-aggregation settings.

    ``expected_daily_indicators`` and ``expected_monthly_features`` are asserted
    invariants, not descriptions: the pipeline fails if the produced matrix disagrees.
    Counting the thesis's indicator table gives 43 daily columns (window variants
    included), and {mean, std, last} aggregation gives 129 monthly columns.
    """

    ma_windows: tuple[int, ...]
    ratio_windows: tuple[int, ...]
    rsi_window: int
    mfi_window: int
    stochastic_window: int
    min_trading_days_per_month: int
    expected_daily_indicators: int = 43
    expected_monthly_features: int = 129
    aggregations: tuple[str, ...] = ("mean", "std", "last")
    dictionary_version: str = "1.0.0"

    def __post_init__(self) -> None:
        # 8 scalar indicators + |ma_windows| + TP + 3 money-flow + SO
        # + 5 ratio families x |ratio_windows|
        counted = 8 + len(self.ma_windows) + 1 + 3 + 1 + 5 * len(self.ratio_windows)
        if counted != self.expected_daily_indicators:
            raise ConfigError(
                f"indicator windows imply {counted} daily columns but "
                f"expected_daily_indicators is {self.expected_daily_indicators}"
            )
        implied = counted * len(self.aggregations)
        if implied != self.expected_monthly_features:
            raise ConfigError(
                f"{counted} daily columns x {len(self.aggregations)} aggregations = "
                f"{implied}, but expected_monthly_features is "
                f"{self.expected_monthly_features}"
            )

    def warmup_days(self) -> int:
        """Trading days of history an indicator row needs before it is complete."""
        return max(
            max(self.ma_windows),
            max(self.ratio_windows),
            self.rsi_window,
            self.mfi_window,
            self.stochastic_window,
        )


@dataclass(frozen=True, slots=True)
class ScreeningConfig:
    """Labeling, screeners and nested model selection."""

    labeling_strategies: tuple[str, ...]
    screeners: tuple[str, ...]
    inner_metric: str
    inner_folds: int
    min_training_observations: int
    min_training_months: int = 3
    hyperparameter_grids: Mapping[str, Mapping[str, Sequence[Any]]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        unknown = set(self.labeling_strategies) - set(LABELING_STRATEGIES)
        if unknown:
            raise ConfigError(f"unknown labeling strategies: {sorted(unknown)}")
        unknown = set(self.screeners) - set(SCREENERS)
        if unknown:
            raise ConfigError(f"unknown screeners: {sorted(unknown)}")
        if self.inner_folds < 2:
            raise ConfigError("inner_folds must be at least 2")
        # A screener whose grid is absent silently receives an empty settings dict. For
        # TabPFN that resolves to a gated checkpoint needing two account registrations, so
        # the requirement is checked here rather than discovered mid-grid.
        if "tabpfn" in self.screeners:
            grid = self.hyperparameter_grids.get("tabpfn", {})
            paths = grid.get("model_path") or []
            if not paths:
                raise ConfigError(
                    "tabpfn is in the screener set but its hyperparameter grid names no "
                    "model_path; the default resolves to a gated checkpoint requiring a "
                    "Prior Labs licence and a HuggingFace account, which would make the "
                    "study unreproducible. Pin a public checkpoint such as "
                    "'tabpfn-v2-classifier.ckpt'."
                )
        if self.min_training_months < 1:
            raise ConfigError("min_training_months must be at least 1")


@dataclass(frozen=True, slots=True)
class OptimizationConfig:
    """NSGA-III search, weight cap factor and the ex-ante moment estimates."""

    population_size: int
    generations: int
    weight_caps: tuple[float, ...]
    seeds: int
    covariance_trailing_days: int
    covariance_estimator: str
    knee_normalization: str
    hypervolume_reference: tuple[float, float]

    @property
    def evaluation_budget(self) -> int:
        """Objective evaluations per run, the budget the main grid grants."""
        return self.population_size * self.generations

    def __post_init__(self) -> None:
        for cap in self.weight_caps:
            if not 0.0 < cap <= 1.0:
                raise ConfigError(f"weight cap {cap} must lie in (0, 1]")
        if self.covariance_estimator not in COVARIANCE_ESTIMATORS:
            raise ConfigError(f"unknown covariance estimator {self.covariance_estimator!r}")
        if self.seeds < 1:
            raise ConfigError("seeds must be at least 1")
        if self.covariance_trailing_days < 2:
            raise ConfigError("covariance_trailing_days must exceed 1")


@dataclass(frozen=True, slots=True)
class AblationConfig:
    """Selection-size sweep and search instrumentation.

    ``selection_sizes`` may contain ``None``, meaning "no preselection": pass the entire
    monthly universe to the optimiser. That level is the point of the ablation.
    """

    enabled: bool
    labeling_strategy: str
    screeners: tuple[str, ...]
    selection_sizes: tuple[int | None, ...]
    weight_cap: float
    seeds: int
    budget_multiplier: int
    checkpoint_evaluations: tuple[int, ...]

    def extended_budget(self, base_budget: int) -> int:
        """Objective evaluations granted to an ablation run."""
        return base_budget * self.budget_multiplier

    def __post_init__(self) -> None:
        if self.labeling_strategy not in LABELING_STRATEGIES:
            raise ConfigError(f"unknown labeling strategy {self.labeling_strategy!r}")
        if None not in self.selection_sizes:
            raise ConfigError(
                "selection_sizes must include null (no preselection); without it the "
                "ablation cannot answer whether preselection is justified"
            )
        if self.budget_multiplier < 1:
            raise ConfigError("budget_multiplier must be at least 1")
        if tuple(sorted(self.checkpoint_evaluations)) != self.checkpoint_evaluations:
            raise ConfigError("checkpoint_evaluations must be ascending")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Executable pricing, transaction costs and metric conventions."""

    cost_scenarios_bps: tuple[float, ...]
    primary_cost_scenario_bps: float
    risk_free_series: str
    trading_days_per_month: int

    def __post_init__(self) -> None:
        if 0.0 not in self.cost_scenarios_bps:
            raise ConfigError(
                "cost_scenarios_bps must include 0 so the gross upper bound is explicit"
            )
        if self.primary_cost_scenario_bps not in self.cost_scenarios_bps:
            raise ConfigError("primary_cost_scenario_bps must be one of cost_scenarios_bps")
        if self.primary_cost_scenario_bps == 0.0:
            raise ConfigError(
                "the primary cost scenario must not be the zero-cost case; gross results "
                "are reported only as an upper bound"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Comparator suite and the pre-specified best-arm rule.

    ``best_arm_rule`` is fixed before the grid runs. The manifest records that it predates
    result generation, which is what stops the best proposed arm from being chosen after
    seeing which cell happened to win.
    """

    reference_series: tuple[str, ...]
    tier1_methods: tuple[str, ...]
    tier2_methods: tuple[str, ...]
    best_arm_rule: str
    momentum_formation_months: int
    momentum_skip_months: int
    momentum_selection_fraction: float
    sensitivity_estimator: str

    def __post_init__(self) -> None:
        if self.sensitivity_estimator not in COVARIANCE_ESTIMATORS:
            raise ConfigError(f"unknown sensitivity estimator {self.sensitivity_estimator!r}")
        if not 0.0 < self.momentum_selection_fraction <= 1.0:
            raise ConfigError("momentum_selection_fraction must lie in (0, 1]")


@dataclass(frozen=True, slots=True)
class PrimaryHypothesis:
    """One pre-specified confirmatory test.

    Declaring these before the grid runs separates a confirmatory claim from a finding
    selected after seeing the results. Everything not listed here is exploratory.

    **Only factor effects may be confirmatory.** Measured on this universe and window, a
    head-to-head comparison between two individual arms has a minimum detectable effect of
    roughly 24% annualised, because the paired difference between two ~20-name portfolios
    carries about 2.9% monthly dispersion. A realistic advantage -- the 6.1% annual gap
    measured between an equal-weight arm and a momentum arm -- is four times smaller than
    that. Such a test cannot reject, and declaring it confirmatory would stage a null result
    that was determined by the design rather than by the data.

    A factor contrast averages over the cells sharing a level: with 216 cells and three
    labeling levels, 72 cells per level, dispersion falls to about 0.44% monthly and the
    minimum detectable effect to roughly 3.7% annualised -- below the effects worth
    claiming. That is where the confirmatory family belongs.

    Head-to-head comparisons remain in the paper, reported as effect sizes with confidence
    intervals and an explicit detectable-effect bound, not as significance tests.
    """

    name: str
    kind: str  # "factor_effect"
    factor: str
    contrast: str
    metric: str

    def __post_init__(self) -> None:
        if self.kind not in CONFIRMATORY_KINDS:
            raise ConfigError(
                f"hypothesis {self.name!r} has kind {self.kind!r}; only "
                f"{sorted(CONFIRMATORY_KINDS)} may be confirmatory. A head-to-head "
                "comparison is underpowered on this design and belongs in the exploratory "
                "set, reported as an effect size with its detectable-effect bound."
            )


@dataclass(frozen=True, slots=True)
class StatsConfig:
    """Inference settings.

    ``primary_hypotheses`` fixes the confirmatory family. The study will produce tests
    across three investor profiles, four cost scenarios, several metrics and a comparator
    suite; correcting within each family separately leaves the error rate across everything
    reported uncontrolled. Naming a small confirmatory set in advance, and labelling the
    rest exploratory, is what makes the headline claims defensible.
    """

    alpha: float
    bootstrap_resamples: int
    bootstrap_scheme: str
    bootstrap_seed: int
    correction_method: str
    seed_aggregation: str
    headline_set_size: int
    target_power: float = 0.80
    primary_hypotheses: tuple[PrimaryHypothesis, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ConfigError("alpha must lie in (0, 1)")
        if self.bootstrap_resamples < 100:
            raise ConfigError("bootstrap_resamples below 100 gives unusable intervals")
        if not 0.0 < self.target_power < 1.0:
            raise ConfigError("target_power must lie in (0, 1)")
        if not self.primary_hypotheses:
            raise ConfigError(
                "no primary hypotheses declared: without a pre-specified confirmatory "
                "family, every reported test is exploratory and no headline claim is "
                "defensible"
            )
        if len(self.primary_hypotheses) > MAX_PRIMARY_HYPOTHESES:
            raise ConfigError(
                f"{len(self.primary_hypotheses)} primary hypotheses declared; a "
                f"confirmatory family larger than {MAX_PRIMARY_HYPOTHESES} is not "
                "confirmatory in any useful sense"
            )
        names = [h.name for h in self.primary_hypotheses]
        if len(set(names)) != len(names):
            raise ConfigError("primary hypothesis names must be unique")


@dataclass(frozen=True, slots=True)
class PathsConfig:
    """Filesystem layout, resolved to absolute paths at load time."""

    root: Path
    cache: Path
    universe: Path
    features: Path
    results: Path
    paper: Path

    def ensure(self) -> None:
        """Create every configured directory. Idempotent."""
        for p in (self.cache, self.universe, self.features, self.results, self.paper):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True, slots=True)
class RunConfig:
    """The fully resolved configuration for one pipeline run."""

    name: str
    windows: WindowConfig
    universe: UniverseConfig
    features: FeatureConfig
    screening: ScreeningConfig
    optimization: OptimizationConfig
    ablation: AblationConfig
    evaluation: EvaluationConfig
    benchmarks: BenchmarkConfig
    stats: StatsConfig
    paths: PathsConfig

    def factorial_cell_count(self) -> int:
        """Number of (labeling, screener, allocator, cap) cells in the main grid."""
        return (
            len(self.screening.labeling_strategies)
            * len(self.screening.screeners)
            * len(ALLOCATORS)
            * len(self.optimization.weight_caps)
        )

    def nsga_run_count(self) -> int:
        """Number of NSGA-III runs the main grid requires.

        One front serves all three GA allocators, so the allocator factor does not
        multiply here -- a distinction worth keeping explicit, since conflating the two
        overstates the compute by a factor of three.
        """
        return (
            len(self.screening.labeling_strategies)
            * len(self.screening.screeners)
            * len(self.optimization.weight_caps)
            * len(self.windows.eval_months())
            * self.optimization.seeds
        )

    def fingerprint(self) -> str:
        """Stable hash of the configuration, for manifest cross-referencing."""
        payload = json.dumps(_jsonable(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view, suitable for embedding in the run manifest."""
        return _jsonable(self)


def _jsonable(obj: Any) -> Any:
    """Convert a config tree into JSON-safe primitives."""
    if isinstance(obj, PathsConfig):
        return {k: str(v) for k, v in asdict(obj).items()}
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _jsonable(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _as_date(value: Any, field_name: str) -> dt.date:
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise ConfigError(f"{field_name} must be an ISO date, got {value!r}")


def load_config(path: str | Path, root: str | Path | None = None) -> RunConfig:
    """Load and validate a configuration file.

    Args:
        path: YAML configuration file.
        root: Project root that relative paths resolve against. Defaults to the parent
            of the config file's directory, which makes ``configs/x.yaml`` resolve
            against the repository root.

    Raises:
        ConfigError: if the file is internally inconsistent. Validation happens here so
            that a bad config fails before any data is downloaded or any grid is launched.
    """
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = Path(root).expanduser().resolve() if root else path.parent.parent

    def section(key: str) -> dict[str, Any]:
        value = raw.get(key)
        if not isinstance(value, dict):
            raise ConfigError(f"configuration section {key!r} is missing or not a mapping")
        return value

    w = section("windows")
    windows = WindowConfig(
        download_start=_as_date(w["download_start"], "download_start"),
        download_end=_as_date(w["download_end"], "download_end"),
        eval_start_month=str(w["eval_start_month"]),
        eval_end_month=str(w["eval_end_month"]),
    )

    u = section("universe")
    universe = UniverseConfig(
        trailing_months=int(u["trailing_months"]),
        min_traded_day_fraction=float(u["min_traded_day_fraction"]),
        min_median_traded_value_mxn=float(u["min_median_traded_value_mxn"]),
        max_stale_run_days=int(u["max_stale_run_days"]),
        min_universe_size=int(u["min_universe_size"]),
        require_full_window_coverage=bool(u.get("require_full_window_coverage", False)),
        min_window_coverage_fraction=float(u.get("min_window_coverage_fraction", 0.95)),
        max_edge_gap_days=int(u.get("max_edge_gap_days", 7)),
        max_zero_return_fraction=float(u.get("max_zero_return_fraction", 0.50)),
        min_month_median_volume=float(u.get("min_month_median_volume", 1.0)),
        max_abs_daily_return=float(u.get("max_abs_daily_return", 0.50)),
    )

    f = section("features")
    features = FeatureConfig(
        ma_windows=tuple(int(x) for x in f["ma_windows"]),
        ratio_windows=tuple(int(x) for x in f["ratio_windows"]),
        rsi_window=int(f["rsi_window"]),
        mfi_window=int(f["mfi_window"]),
        stochastic_window=int(f["stochastic_window"]),
        min_trading_days_per_month=int(f["min_trading_days_per_month"]),
        expected_daily_indicators=int(f.get("expected_daily_indicators", 43)),
        expected_monthly_features=int(f.get("expected_monthly_features", 129)),
        aggregations=tuple(f.get("aggregations", ("mean", "std", "last"))),
        dictionary_version=str(f.get("dictionary_version", "1.0.0")),
    )

    s = section("screening")
    screening = ScreeningConfig(
        labeling_strategies=tuple(s["labeling_strategies"]),
        screeners=tuple(s["screeners"]),
        inner_metric=str(s["inner_metric"]),
        inner_folds=int(s["inner_folds"]),
        min_training_observations=int(s["min_training_observations"]),
        min_training_months=int(s.get("min_training_months", 3)),
        hyperparameter_grids=s.get("hyperparameter_grids", {}),
    )

    o = section("optimization")
    optimization = OptimizationConfig(
        population_size=int(o["population_size"]),
        generations=int(o["generations"]),
        weight_caps=tuple(float(x) for x in o["weight_caps"]),
        seeds=int(o["seeds"]),
        covariance_trailing_days=int(o["covariance_trailing_days"]),
        covariance_estimator=str(o["covariance_estimator"]),
        knee_normalization=str(o["knee_normalization"]),
        hypervolume_reference=(
            float(o["hypervolume_reference"][0]),
            float(o["hypervolume_reference"][1]),
        ),
    )

    a = section("ablation")
    ablation = AblationConfig(
        enabled=bool(a["enabled"]),
        labeling_strategy=str(a["labeling_strategy"]),
        screeners=tuple(a["screeners"]),
        selection_sizes=tuple(None if x is None else int(x) for x in a["selection_sizes"]),
        weight_cap=float(a["weight_cap"]),
        seeds=int(a["seeds"]),
        budget_multiplier=int(a["budget_multiplier"]),
        checkpoint_evaluations=tuple(int(x) for x in a["checkpoint_evaluations"]),
    )

    e = section("evaluation")
    evaluation = EvaluationConfig(
        cost_scenarios_bps=tuple(float(x) for x in e["cost_scenarios_bps"]),
        primary_cost_scenario_bps=float(e["primary_cost_scenario_bps"]),
        risk_free_series=str(e["risk_free_series"]),
        trading_days_per_month=int(e["trading_days_per_month"]),
    )

    b = section("benchmarks")
    benchmarks = BenchmarkConfig(
        reference_series=tuple(b["reference_series"]),
        tier1_methods=tuple(b["tier1_methods"]),
        tier2_methods=tuple(b["tier2_methods"]),
        best_arm_rule=str(b["best_arm_rule"]),
        momentum_formation_months=int(b["momentum_formation_months"]),
        momentum_skip_months=int(b["momentum_skip_months"]),
        momentum_selection_fraction=float(b["momentum_selection_fraction"]),
        sensitivity_estimator=str(b["sensitivity_estimator"]),
    )

    t = section("stats")
    stats = StatsConfig(
        alpha=float(t["alpha"]),
        bootstrap_resamples=int(t["bootstrap_resamples"]),
        bootstrap_scheme=str(t["bootstrap_scheme"]),
        bootstrap_seed=int(t["bootstrap_seed"]),
        correction_method=str(t["correction_method"]),
        seed_aggregation=str(t["seed_aggregation"]),
        headline_set_size=int(t["headline_set_size"]),
        target_power=float(t.get("target_power", 0.80)),
        primary_hypotheses=tuple(
            PrimaryHypothesis(
                name=str(h["name"]),
                kind=str(h.get("kind", "factor_effect")),
                factor=str(h["factor"]),
                contrast=str(h["contrast"]),
                metric=str(h["metric"]),
            )
            for h in t.get("primary_hypotheses", [])
        ),
    )

    p = raw.get("paths", {}) or {}

    def resolve(key: str, default: str) -> Path:
        value = Path(str(p.get(key, default))).expanduser()
        return value if value.is_absolute() else (base / value).resolve()

    paths = PathsConfig(
        root=base,
        cache=resolve("cache", "data/cache"),
        universe=resolve("universe", "data/universe"),
        features=resolve("features", "data/features"),
        results=resolve("results", "results"),
        paper=resolve("paper", "paper"),
    )

    # Calendar feasibility. The download window must leave room for indicator warm-up, the
    # trailing three-month return, the t+1 label lag, and the minimum training history --
    # otherwise the first evaluation months silently train on nothing, or on one month of
    # data, and the screening stage produces noise that looks like a result.
    warmup_months = -(-features.warmup_days() // evaluation.trading_days_per_month)
    first_feature_month = warmup_months + 1
    first_ma3_month = 5  # returns begin in month 2; three priors land in month 5
    first_labelable = max(first_feature_month, first_ma3_month)
    first_available = first_labelable + 1  # the t+1 label needs the following month closed
    required_lead = first_available + screening.min_training_months

    download_index = windows.download_start.year * 12 + windows.download_start.month
    eval_year, eval_month = (int(p) for p in windows.eval_start_month.split("-"))
    lead = (eval_year * 12 + eval_month) - download_index + 1
    if lead < required_lead:
        raise ConfigError(
            f"evaluation starts {lead} months after download_start but needs at least "
            f"{required_lead}: {warmup_months} months of indicator warm-up, a trailing "
            f"three-month return, the t+1 label lag, and {screening.min_training_months} "
            f"labelable training months. Move eval_start_month later or download_start "
            f"earlier."
        )

    # Momentum feasibility. The Tier 1 momentum comparator needs a formation window of
    # monthly returns ending before the evaluation month, and monthly returns only begin
    # one month after the download start. Without this check a configuration can name the
    # conventional 12-1 rule while silently having only part of it, which would be reported
    # as a standard baseline it is not.
    months_of_returns = (eval_year * 12 + eval_month) - download_index - 1
    momentum_need = benchmarks.momentum_formation_months + benchmarks.momentum_skip_months
    if months_of_returns < momentum_need:
        raise ConfigError(
            f"momentum comparator needs {momentum_need} months of returns before the first "
            f"evaluation month ({benchmarks.momentum_formation_months} formation + "
            f"{benchmarks.momentum_skip_months} skip) but only {months_of_returns} are "
            f"available from download_start. Extend download_start, move eval_start_month "
            f"later, or shorten the formation window and declare the deviation."
        )

    # Momentum baseline concentration. A selection rule expressed as a fraction behaves very
    # differently on a 93-instrument universe than on the thousands the convention assumes:
    # a top-decile rule here would hold nine positions, and a baseline that concentrated
    # loses on idiosyncratic variance rather than on the absence of momentum. Losing to an
    # unfairly noisy baseline is as misleading as losing to none at all.
    momentum_positions = benchmarks.momentum_selection_fraction * universe.min_universe_size
    if momentum_positions < MIN_MOMENTUM_POSITIONS:
        raise ConfigError(
            f"the momentum comparator would hold about "
            f"{momentum_positions:.0f} positions at the smallest admissible universe "
            f"({universe.min_universe_size} instruments x "
            f"{benchmarks.momentum_selection_fraction:.2f}), below the "
            f"{MIN_MOMENTUM_POSITIONS} needed for it to be a fair baseline rather than a "
            f"concentrated bet. Raise momentum_selection_fraction or min_universe_size."
        )

    # Cross-section consistency: the ablation may only use screeners the run trains.
    missing = set(ablation.screeners) - set(screening.screeners)
    if ablation.enabled and missing:
        raise ConfigError(
            f"ablation references screeners not in the run: {sorted(missing)}"
        )
    if ablation.enabled and ablation.labeling_strategy not in screening.labeling_strategies:
        raise ConfigError(
            "ablation labeling strategy is not among the run's labeling strategies"
        )
    if ablation.enabled:
        budget = ablation.extended_budget(optimization.evaluation_budget)
        if optimization.evaluation_budget not in ablation.checkpoint_evaluations:
            raise ConfigError(
                "checkpoint_evaluations must include the main grid budget "
                f"({optimization.evaluation_budget}); the fixed-budget result is read "
                "from that checkpoint rather than from a separate run"
            )
        if max(ablation.checkpoint_evaluations) > budget:
            raise ConfigError(
                f"checkpoint at {max(ablation.checkpoint_evaluations)} exceeds the "
                f"extended ablation budget of {budget}"
            )

    return RunConfig(
        name=str(raw.get("name", path.stem)),
        windows=windows,
        universe=universe,
        features=features,
        screening=screening,
        optimization=optimization,
        ablation=ablation,
        evaluation=evaluation,
        benchmarks=benchmarks,
        stats=stats,
        paths=paths,
    )
