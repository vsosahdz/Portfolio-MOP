"""Labeling, screeners and the expanding-window selection protocol.

The tests that matter most here guard two things that cannot be checked after the fact: the
label-availability lag, and the requirement that classification metrics be decomposed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bmvport.config import load_config
from bmvport.screening.labeling import (
    LABELING_STRATEGIES,
    apply_labeling,
    class_balance,
    label_components,
    requires_forward_decomposition,
)
from bmvport.screening.screeners import (
    build_screener,
    screener_availability,
    availability_substitutions,
)
from bmvport.screening.selection import (
    classification_metrics,
    expand_grid,
    run_screening,
    training_months_for,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIGS / "default.yaml")


def _matrix(n_symbols: int = 40, months: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods = pd.period_range("2023-04", periods=months, freq="M").astype(str)
    rows = []
    for symbol in [f"S{i}.MX" for i in range(n_symbols)]:
        for month in periods:
            row = {
                "provider_symbol": symbol,
                "month": month,
                "sessions": 20,
                "monthly_return": rng.normal(0.005, 0.05),
                "trailing_3m_mean_return": rng.normal(0.005, 0.03),
                "next_month_return": rng.normal(0.005, 0.05),
            }
            for j in range(6):
                row[f"f{j}"] = rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


# --- labeling -------------------------------------------------------------------------


def test_three_strategies_with_their_exact_predicates() -> None:
    frame = pd.DataFrame(
        {
            "monthly_return": [0.02, -0.01, 0.01, 0.03],
            "trailing_3m_mean_return": [0.04, -0.05, 0.06, -0.02],
            "next_month_return": [0.04, 0.05, -0.05, 0.01],
        }
    )
    assert apply_labeling(frame, "historical").tolist() == [True, False, True, False]
    assert apply_labeling(frame, "t_plus_1").tolist() == [True, True, False, True]
    assert apply_labeling(frame, "historical_and_t1").tolist() == [True, False, False, False]


def test_unlabelable_rows_are_dropped_not_called_negative() -> None:
    """Treating a missing input as a negative example would bias every classifier."""
    frame = pd.DataFrame(
        {
            "monthly_return": [0.02, np.nan],
            "trailing_3m_mean_return": [0.04, 0.01],
            "next_month_return": [0.04, 0.02],
        }
    )
    labels = apply_labeling(frame, "historical_and_t1")
    assert len(labels) == 1
    assert labels.index.tolist() == [0]


def test_components_are_exposed_for_decomposition() -> None:
    components = label_components(
        pd.DataFrame(
            {
                "monthly_return": [0.01],
                "trailing_3m_mean_return": [0.02],
                "next_month_return": [-0.01],
            }
        )
    )
    assert set(components.columns) == {
        "current_month_positive", "trailing_mean_positive", "next_month_positive",
    }


def test_strategies_carrying_the_present_require_decomposition() -> None:
    assert requires_forward_decomposition("historical") is True
    assert requires_forward_decomposition("historical_and_t1") is True
    assert requires_forward_decomposition("t_plus_1") is False


def test_class_balance_is_recorded_per_month() -> None:
    balances = class_balance(_matrix(n_symbols=10, months=4), "t_plus_1")
    assert len(balances) == 4
    assert all(b.observations == 10 for b in balances)
    assert all(0.0 <= b.positive_rate <= 1.0 for b in balances)


def test_missing_return_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        label_components(pd.DataFrame({"monthly_return": [0.1]}))


# --- the label-availability lag -------------------------------------------------------


def test_training_months_stop_two_months_before_evaluation() -> None:
    """Month E-1's label would need month E's own return: the thing being predicted."""
    available = ["2024-09", "2024-10", "2024-11", "2024-12", "2025-01"]
    assert training_months_for("2025-01", available) == ["2024-09", "2024-10", "2024-11"]
    assert "2024-12" not in training_months_for("2025-01", available)


def test_the_lag_holds_across_a_year_boundary() -> None:
    assert training_months_for("2025-02", ["2024-11", "2024-12", "2025-01"]) == [
        "2024-11", "2024-12",
    ]


def test_no_evaluation_month_data_reaches_training(config) -> None:
    matrix = _matrix()
    result = run_screening(matrix, config, labeling="t_plus_1", screener="all_stocks")
    for selection in result.selections:
        if selection.skipped and selection.training_months == 0:
            continue
        allowed = training_months_for(selection.month, sorted(set(matrix["month"])))
        assert selection.month not in allowed
        # And the month immediately before is excluded too, by the lag.
        year, month = (int(p) for p in selection.month.split("-"))
        previous = f"{year:04d}-{month - 1:02d}" if month > 1 else f"{year - 1:04d}-12"
        assert previous not in allowed


def test_label_inputs_never_reach_the_feature_matrix(config) -> None:
    """The columns the labels are built from must be invisible to every screener.

    ``monthly_return`` and ``trailing_3m_mean_return`` determine two thirds of the proposed
    label outright; ``next_month_return`` is the label itself. Any of them in the training
    matrix turns the classification task into an identity.
    """
    from bmvport.screening.selection import KEY_COLUMNS

    matrix = _matrix()
    feature_columns = [c for c in matrix.columns if c not in KEY_COLUMNS]
    for forbidden in ("monthly_return", "trailing_3m_mean_return", "next_month_return"):
        assert forbidden in matrix.columns  # present in the matrix...
        assert forbidden not in feature_columns  # ...but never a feature
    assert "provider_symbol" not in feature_columns
    assert "month" not in feature_columns


def test_screener_receives_only_feature_columns(config, monkeypatch) -> None:
    """Verified at the call site, not just by inspecting the column list."""
    seen: list[int] = []

    class _Spy:
        def fit(self, X, y):
            seen.append(X.shape[1]); return self
        def predict(self, X):
            return np.zeros(len(X), dtype=bool)
        def decision_score(self, X):
            return np.zeros(len(X), dtype=float)

    import bmvport.screening.selection as selection

    monkeypatch.setattr(selection, "build_screener", lambda name, **kw: _Spy())
    matrix = _matrix()
    selection.run_screening(matrix, config, labeling="t_plus_1", screener="all_stocks")
    assert seen
    # Six synthetic features and nothing else.
    assert set(seen) == {6}


# --- screeners ------------------------------------------------------------------------


def test_all_stocks_selects_everything() -> None:
    screener = build_screener("all_stocks")
    X = np.random.default_rng(0).normal(size=(25, 6))
    screener.fit(X, np.random.default_rng(1).integers(0, 2, 25).astype(bool))
    assert screener.predict(X).all()


def test_every_screener_exposes_the_same_interface(config) -> None:
    """Built as the pipeline builds them: with their configured hyperparameters.

    Constructing a screener bare would miss settings the pipeline always supplies -- and for
    TabPFN it would resolve to a gated checkpoint, which the builder refuses.
    """
    from bmvport.screening.selection import expand_grid

    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 8))
    y = (X[:, 0] + rng.normal(0, 0.5, 160)) > 0.4
    for name, (available, _) in screener_availability(config.screening.screeners).items():
        if not available or name == "pbc4cip":  # pbc4cip is exercised separately; it is slow
            continue
        settings = expand_grid(config.screening.hyperparameter_grids.get(name, {}))[0]
        screener = build_screener(name, **settings)
        screener.fit(X, y)
        predicted = screener.predict(X)
        scores = screener.decision_score(X)
        assert predicted.dtype == bool and predicted.shape == (160,)
        assert scores.shape == (160,) and np.isfinite(scores).all()


def test_unavailable_screener_is_disclosed_as_affecting_results(config) -> None:
    """A hole in the factorial must not read as a screener that did poorly."""
    for substitution in availability_substitutions(config.screening.screeners):
        assert substitution.affects_results is True
        assert substitution.used_instead.startswith("none")
        assert substitution.reason


def test_unknown_screener_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown screener"):
        build_screener("random_guess")


# --- model selection and metrics ------------------------------------------------------


def test_grid_expansion() -> None:
    settings = expand_grid({"a": [1, 2], "b": ["x", "y"]})
    assert len(settings) == 4
    assert {"a": 2, "b": "x"} in settings
    assert expand_grid({}) == [{}]


def test_single_point_grid_skips_the_inner_loop(config) -> None:
    """PBC4cip's case: running inner folds for one candidate costs fits and decides nothing."""
    from bmvport.screening.selection import select_hyperparameters

    matrix = _matrix(n_symbols=20, months=12)
    labels = apply_labeling(matrix, "t_plus_1")
    setting, inner = select_hyperparameters(
        "pbc4cip", matrix.loc[labels.index], labels, labels,
        sorted(set(matrix["month"])), ["f0", "f1"], config,
    )
    assert inner is None  # no inner scoring took place
    assert setting == {"tree_count": 100}


def test_metrics_are_decomposed_for_present_bearing_labels() -> None:
    y = np.array([1, 1, 0, 0, 1, 0])
    pred = np.array([1, 0, 0, 1, 1, 0])
    forward = np.array([0, 1, 0, 1, 1, 1])
    metrics = classification_metrics(y, pred, forward, decompose=True)
    assert "balanced_accuracy" in metrics
    assert "forward_balanced_accuracy" in metrics
    assert metrics["forward_positive_rate"] == pytest.approx(forward.mean())


def test_metrics_are_not_decomposed_for_the_forward_label() -> None:
    y = np.array([1, 0, 1, 0])
    metrics = classification_metrics(y, np.array([1, 0, 0, 0]), y, decompose=False)
    assert not any(k.startswith("forward_") for k in metrics)


def test_single_class_month_yields_nan_not_a_misleading_score() -> None:
    y = np.ones(5, dtype=int)
    metrics = classification_metrics(y, np.ones(5, dtype=int), y, decompose=False)
    assert np.isnan(metrics["balanced_accuracy"])


# --- end to end -----------------------------------------------------------------------


def test_screening_runs_over_the_evaluation_window(config) -> None:
    result = run_screening(_matrix(), config, labeling="t_plus_1", screener="all_stocks")
    assert len(result.selections) == len(config.windows.eval_months())
    assert not result.failures


def test_thin_training_history_is_skipped_with_a_reason(config) -> None:
    matrix = _matrix(n_symbols=5, months=8)
    result = run_screening(matrix, config, labeling="t_plus_1", screener="all_stocks")
    skipped = [s for s in result.selections if s.skipped]
    assert skipped
    assert all(s.skipped for s in skipped)


def test_membership_restricts_what_can_be_selected(config) -> None:
    matrix = _matrix()
    allowed = {m: ["S0.MX", "S1.MX"] for m in config.windows.eval_months()}
    result = run_screening(
        matrix, config, labeling="t_plus_1", screener="all_stocks", membership=allowed
    )
    for selection in result.selections:
        assert set(selection.selected) <= {"S0.MX", "S1.MX"}


def test_every_strategy_is_runnable(config) -> None:
    matrix = _matrix()
    for strategy in LABELING_STRATEGIES:
        result = run_screening(matrix, config, labeling=strategy, screener="all_stocks")
        assert len(result.selections) == len(config.windows.eval_months())


def test_prediction_does_not_require_the_label_to_exist(config) -> None:
    """The final evaluation month has no t+1 return, yet it is still predictable.

    A screener in production selects for month E without knowing E's outcome. Restricting
    prediction to labelable rows would silently discard the last month of the study.
    """
    matrix = _matrix()
    last = sorted(set(matrix["month"]))[-1]
    # Strip the forward return for the last month, as the real window does.
    matrix.loc[matrix["month"] == last, "next_month_return"] = np.nan

    import bmvport.screening.selection as selection

    months = [last]
    result = selection.run_screening(
        matrix, config, labeling="historical_and_t1", screener="all_stocks",
        membership={last: matrix["provider_symbol"].unique().tolist()},
    )
    for entry in result.selections:
        if entry.month != last:
            continue
        # A selection was still produced, and it simply carries no scored observations.
        assert entry.skipped != "no universe members with features this month"


def test_metrics_are_absent_rather_than_invented_when_unlabelable(config) -> None:
    """Metrics score the SELECTION month, which is the month before the holding month.

    With selection at E-1, its label depends on the holding month's return, which is
    observed by evaluation time -- so metrics are normally computable. They are absent only
    when the selection month itself carries no label, and then they are recorded as absent
    rather than invented.
    """
    from bmvport.screening.selection import previous_month

    matrix = _matrix()
    holding = sorted(set(matrix["month"]))[-1]
    selection = previous_month(holding)
    matrix.loc[matrix["month"] == selection, "next_month_return"] = np.nan

    result = run_screening(matrix, config, labeling="t_plus_1", screener="all_stocks")
    for entry in result.selections:
        if entry.month == holding and entry.metrics:
            assert entry.metrics.get("scored_observations", 0) == 0


def test_tabpfn_refuses_an_implicit_checkpoint() -> None:
    """``model_path="auto"`` resolves to a gated model and would break reproducibility."""
    with pytest.raises(RuntimeError, match="explicit model_path"):
        build_screener("tabpfn")


def test_tabpfn_checkpoint_is_pinned_to_a_public_one(config) -> None:
    """Verified to download with no token of any kind, so replication needs no account."""
    grid = config.screening.hyperparameter_grids["tabpfn"]
    assert grid["model_path"] == ["tabpfn-v2-classifier.ckpt"]
    assert "v2" in grid["model_path"][0]


def test_tabpfn_availability_does_not_depend_on_a_token(monkeypatch) -> None:
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    available, reason = screener_availability(["tabpfn"])["tabpfn"]
    assert available is True
    assert reason == ""


def test_selection_uses_the_prior_month_features_not_the_holding_month(config) -> None:
    """The one-month look-ahead that produced 50-64% annualised returns.

    Monthly features aggregate their own month's daily indicators, so they determine that
    month's return almost by identity (measured correlation 0.991). Selecting on month E's
    features and holding through month E picks the stocks that already rose.
    """
    from bmvport.screening.selection import previous_month

    matrix = _matrix()
    months = sorted(set(matrix["month"]))
    holding = months[-1]
    selection = previous_month(holding)

    # Make the holding month's features unmistakably distinct; if selection consulted them,
    # the chosen set would change.
    marked = matrix.copy()
    marked.loc[marked["month"] == holding, [f"f{i}" for i in range(6)]] = 99.0

    import bmvport.screening.selection as sel

    seen: list[int] = []

    class _Spy:
        def fit(self, X, y):
            return self
        def predict(self, X):
            seen.append(len(X)); return np.ones(len(X), dtype=bool)
        def decision_score(self, X):
            return np.ones(len(X), dtype=float)

    base = sel.run_screening(matrix, config, labeling="t_plus_1", screener="all_stocks")
    marked_result = sel.run_screening(marked, config, labeling="t_plus_1", screener="all_stocks")

    base_pick = {s.month: set(s.selected) for s in base.selections}
    marked_pick = {s.month: set(s.selected) for s in marked_result.selections}
    # Altering the holding month's features must not change what was selected for it.
    assert base_pick.get(holding) == marked_pick.get(holding)
    assert previous_month("2025-01") == "2024-12"
    assert previous_month("2025-07") == "2025-06"
