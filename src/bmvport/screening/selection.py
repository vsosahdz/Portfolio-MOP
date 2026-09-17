"""Expanding-window training, nested model selection, and per-month stock selection.

**The label-availability rule, which is the easy thing to get wrong.** An observation for
month *m* carries a label that depends on month *m+1*'s return. That label does not exist
until *m+1* has closed. Predicting evaluation month *E* may therefore use observations up to
``m <= E - 2``: month *E-1*'s label would require *E*'s own return, which is precisely the
thing being predicted.

Using ``m <= E - 1`` instead is a one-month leak that no downstream check can detect. It
would look like a screener with genuine foresight, in exactly the months that mattered.

**Model selection stays inside the training window.** Hyperparameters are chosen by an inner
expanding-window split over the training observations only, scored on a classification
metric rather than on portfolio return -- the screening stage is judged on the task it
performs, and the portfolio outcome stays out of sample.

**The inner criterion uses the forward component of the label.** Measured on this feature
matrix, a classifier scores AUC 0.979 on the ``historical`` component and 0.470 on
``t_plus_1``. Tuning on a label containing the first optimises partly for re-describing the
present. The inner score is therefore computed against ``t_plus_1``, where the objective and
the intent agree, while the model still trains on the strategy's own label.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from ..config import RunConfig
from .labeling import (
    FORWARD_COMPONENT,
    apply_labeling,
    requires_forward_decomposition,
)
from .screeners import build_screener

__all__ = [
    "ScreeningResult",
    "MonthlySelection",
    "training_months_for",
    "expand_grid",
    "select_hyperparameters",
    "classification_metrics",
    "run_screening",
    "write_screening_outputs",
]

KEY_COLUMNS = (
    "provider_symbol",
    "month",
    "sessions",
    "monthly_return",
    "trailing_3m_mean_return",
    "next_month_return",
)


@dataclass(frozen=True, slots=True)
class MonthlySelection:
    """What one screener selected for one evaluation month."""

    month: str
    labeling: str
    screener: str
    selected: tuple[str, ...]
    scores: Mapping[str, float]
    training_observations: int
    training_months: int
    hyperparameters: Mapping[str, Any]
    inner_score: float | None
    metrics: Mapping[str, float]
    skipped: str | None = None

    @property
    def size(self) -> int:
        return len(self.selected)


@dataclass(slots=True)
class ScreeningResult:
    """All selections plus the exclusions and failures encountered."""

    selections: list[MonthlySelection] = field(default_factory=list)
    failures: list[dict[str, Any]] = field(default_factory=list)


def previous_month(month: str) -> str:
    """The calendar month before ``month``."""
    year, index = (int(p) for p in month.split("-"))
    if index == 1:
        return f"{year - 1:04d}-12"
    return f"{year:04d}-{index - 1:02d}"


def training_months_for(evaluation_month: str, available: Sequence[str]) -> list[str]:
    """Months whose labels are known before ``evaluation_month`` begins.

    An observation for month *m* needs *m+1*'s return, so it becomes usable only once *m+1*
    has closed. The last admissible training month is therefore ``E - 2``.
    """
    year, month = (int(p) for p in evaluation_month.split("-"))
    index = year * 12 + month
    cutoff = index - 2

    def to_index(value: str) -> int:
        y, m = (int(p) for p in value.split("-"))
        return y * 12 + m

    return sorted({m for m in available if to_index(m) <= cutoff})


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a hyperparameter grid, as a list of settings."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def _inner_folds(months: Sequence[str], folds: int) -> Iterator[tuple[list[str], list[str]]]:
    """Expanding-window splits over the training months.

    Each fold trains on a prefix and validates on the block that follows, so the inner
    selection mirrors the outer protocol instead of shuffling time away.
    """
    if len(months) < folds + 1:
        return
    block = max(len(months) // (folds + 1), 1)
    for k in range(1, folds + 1):
        train_end = block * k
        validate_end = min(block * (k + 1), len(months))
        if train_end >= validate_end:
            continue
        yield list(months[:train_end]), list(months[train_end:validate_end])


def _score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    if len(np.unique(y_true)) < 2:
        return float("nan")
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if metric == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    raise ValueError(f"unsupported inner metric {metric!r}")


def select_hyperparameters(
    screener: str,
    frame: pd.DataFrame,
    labels: pd.Series,
    forward: pd.Series,
    months: Sequence[str],
    feature_columns: Sequence[str],
    config: RunConfig,
) -> tuple[dict[str, Any], float | None]:
    """Choose hyperparameters by inner expanding-window validation.

    Returns the chosen setting and its inner score. When the grid holds a single point the
    inner loop is skipped entirely -- there is nothing to choose, and running it would cost
    several fits of an expensive classifier for no decision.
    """
    grid = expand_grid(config.screening.hyperparameter_grids.get(screener, {}))
    if len(grid) == 1:
        return grid[0], None

    best: tuple[float, dict[str, Any]] | None = None
    for setting in grid:
        scores: list[float] = []
        for train_months, validate_months in _inner_folds(months, config.screening.inner_folds):
            train_mask = frame["month"].isin(train_months)
            validate_mask = frame["month"].isin(validate_months)
            if train_mask.sum() < config.screening.min_training_observations:
                continue
            if validate_mask.sum() == 0 or labels[train_mask].nunique() < 2:
                continue
            try:
                model = build_screener(screener, **setting)
                model.fit(
                    frame.loc[train_mask, feature_columns].to_numpy(float),
                    labels[train_mask].to_numpy(),
                )
                predicted = model.predict(
                    frame.loc[validate_mask, feature_columns].to_numpy(float)
                )
            except Exception:  # noqa: BLE001 - a failed setting is simply not selected
                continue
            # Scored against the forward component: tuning on a label that contains a
            # present-describable conjunct would optimise for re-describing the present.
            scores.append(
                _score(
                    forward[validate_mask].to_numpy().astype(int),
                    np.asarray(predicted).astype(int),
                    config.screening.inner_metric,
                )
            )
        finite = [s for s in scores if np.isfinite(s)]
        if not finite:
            continue
        mean_score = float(np.mean(finite))
        if best is None or mean_score > best[0]:
            best = (mean_score, setting)

    if best is None:
        return grid[0], None
    return best[1], best[0]


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    forward_true: np.ndarray,
    *,
    decompose: bool,
) -> dict[str, float]:
    """Metrics on the label, and separately on its forward component.

    The decomposition is not optional presentation. A label conjoining a condition on the
    current month with one on the next inherits almost all of its apparent predictability
    from the first, because the features contain the current month's own return statistics.
    Reporting only the label's metrics would advertise a screener that is re-describing the
    present as one that is predicting the future.
    """
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
    )

    def block(true: np.ndarray, prefix: str) -> dict[str, float]:
        if len(np.unique(true)) < 2:
            return {
                f"{prefix}accuracy": float(accuracy_score(true, y_pred)),
                f"{prefix}balanced_accuracy": float("nan"),
                f"{prefix}precision": float(precision_score(true, y_pred, zero_division=0)),
                f"{prefix}recall": float(recall_score(true, y_pred, zero_division=0)),
            }
        return {
            f"{prefix}accuracy": float(accuracy_score(true, y_pred)),
            f"{prefix}balanced_accuracy": float(balanced_accuracy_score(true, y_pred)),
            f"{prefix}precision": float(precision_score(true, y_pred, zero_division=0)),
            f"{prefix}recall": float(recall_score(true, y_pred, zero_division=0)),
        }

    metrics = block(y_true.astype(int), "")
    metrics["positive_rate"] = float(np.mean(y_true))
    metrics["selected_rate"] = float(np.mean(y_pred))
    if decompose:
        metrics.update(block(forward_true.astype(int), "forward_"))
        metrics["forward_positive_rate"] = float(np.mean(forward_true))
    return metrics


def _screen_one_month(args: tuple) -> tuple[MonthlySelection | None, dict | None]:
    """Worker for one evaluation month. Module level so it can be pickled."""
    (month, matrix, config, labeling, screener, membership) = args
    result = _run_screening_months(
        matrix, config, labeling=labeling, screener=screener,
        membership=membership, months=[month],
    )
    selection = result.selections[0] if result.selections else None
    failure = result.failures[0] if result.failures else None
    return selection, failure


def run_screening(
    matrix: pd.DataFrame,
    config: RunConfig,
    *,
    labeling: str,
    screener: str,
    membership: Mapping[str, Sequence[str]] | None = None,
    n_jobs: int = 1,
) -> ScreeningResult:
    """Train and predict month by month, optionally across processes.

    Evaluation months are independent given the feature matrix -- each fits on its own
    expanding training window -- so they parallelise cleanly. This matters for one screener
    only: PBC4cip scales as O(n^1.83) and the grid's 54 fits take roughly 83 hours serially
    at its reference ensemble size. Parallelising keeps the classifier at its reference
    settings instead of crippling it to fit the schedule.
    """
    months = list(config.windows.eval_months())
    if n_jobs <= 1 or len(months) <= 1:
        return _run_screening_months(
            matrix, config, labeling=labeling, screener=screener,
            membership=membership, months=months,
        )

    from concurrent.futures import ProcessPoolExecutor

    combined = ScreeningResult()
    payload = [(m, matrix, config, labeling, screener, membership) for m in months]
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        for selection, failure in pool.map(_screen_one_month, payload):
            if selection is not None:
                combined.selections.append(selection)
            if failure is not None:
                combined.failures.append(failure)
    combined.selections.sort(key=lambda s: s.month)
    return combined


def _run_screening_months(
    matrix: pd.DataFrame,
    config: RunConfig,
    *,
    labeling: str,
    screener: str,
    membership: Mapping[str, Sequence[str]] | None = None,
    months: Sequence[str] | None = None,
) -> ScreeningResult:
    """Train and predict month by month across the evaluation window.

    Args:
        matrix: The monthly feature matrix.
        config: Resolved run configuration.
        labeling: Labeling strategy name.
        screener: Screener name.
        membership: Optional universe membership per evaluation month. Prediction is
            restricted to members, so the screener never selects an instrument the universe
            screen excluded.

    Returns:
        Selections for every evaluation month, including months recorded as skipped.
    """
    result = ScreeningResult()
    labels = apply_labeling(matrix, labeling)
    forward = apply_labeling(matrix, FORWARD_COMPONENT)
    labelled = matrix.loc[labels.index]
    forward = forward.reindex(labels.index)
    feature_columns = [c for c in matrix.columns if c not in KEY_COLUMNS]
    decompose = requires_forward_decomposition(labeling)
    available_months = sorted(set(labelled["month"]))

    for month in (months if months is not None else config.windows.eval_months()):
        train_months = training_months_for(month, available_months)
        train_mask = labelled["month"].isin(train_months)
        n_train = int(train_mask.sum())

        def skip(reason: str) -> None:
            result.selections.append(
                MonthlySelection(
                    month=month, labeling=labeling, screener=screener, selected=(),
                    scores={}, training_observations=n_train,
                    training_months=len(train_months), hyperparameters={},
                    inner_score=None, metrics={}, skipped=reason,
                )
            )

        if len(train_months) < config.screening.min_training_months:
            skip(f"training_months={len(train_months)}<{config.screening.min_training_months}")
            continue
        if n_train < config.screening.min_training_observations:
            skip(f"training_observations={n_train}<{config.screening.min_training_observations}")
            continue
        if labels[train_mask].nunique() < 2:
            skip("training labels are single-class")
            continue

        # Selection for holding month E uses the features of month E-1, the last complete
        # information before the position is opened.
        #
        # Using month E's own features would be a one-month look-ahead, and a devastating
        # one: the monthly features aggregate that month's daily indicators, so
        # corr(return_mean of month m, return of month m) is 0.991 while the correlation
        # with month m+1 is 0.068. Selecting on month E's features and holding through
        # month E means picking the stocks that already rose. An earlier version of this
        # stage did exactly that and produced 50-64% annualised returns with median Sharpe
        # above 4 -- numbers that are their own refutation.
        #
        # Prediction needs features, not labels: the selection month's label depends on the
        # holding month's return, which is precisely what is being predicted.
        selection_month = previous_month(month)
        predict_mask = matrix["month"] == selection_month
        if membership is not None:
            # Membership is the holding month's universe: an instrument must be admissible
            # when it is held, not when it was screened.
            allowed = set(membership.get(month, ()))
            predict_mask &= matrix["provider_symbol"].isin(allowed)
        if predict_mask.sum() == 0:
            skip(f"no universe members with features in {selection_month}")
            continue

        setting, inner = select_hyperparameters(
            screener, labelled[train_mask], labels[train_mask], forward[train_mask],
            train_months, feature_columns, config,
        )
        try:
            model = build_screener(screener, **setting)
            model.fit(
                labelled.loc[train_mask, feature_columns].to_numpy(float),
                labels[train_mask].to_numpy(),
            )
            X = matrix.loc[predict_mask, feature_columns].to_numpy(float)
            predicted = np.asarray(model.predict(X)).astype(bool)
            scores = np.asarray(model.decision_score(X), dtype=float)
        except Exception as exc:  # noqa: BLE001
            result.failures.append(
                {"month": month, "labeling": labeling, "screener": screener,
                 "error": f"{type(exc).__name__}: {exc}"[:300]}
            )
            skip(f"screener failed: {type(exc).__name__}")
            continue

        symbols = matrix.loc[predict_mask, "provider_symbol"].to_numpy()
        selected = tuple(symbols[predicted])

        # Metrics are computed only where labels exist. A month whose t+1 return lies
        # outside the window still produces a selection; it simply cannot be scored, and
        # reporting empty metrics is honest where inventing them would not be.
        scored = matrix.index[predict_mask].intersection(labels.index)
        if len(scored) > 0:
            positions = {idx: i for i, idx in enumerate(matrix.index[predict_mask])}
            take = np.array([positions[i] for i in scored])
            metrics = classification_metrics(
                labels.loc[scored].to_numpy(),
                predicted[take],
                forward.loc[scored].to_numpy(),
                decompose=decompose,
            )
            metrics["scored_observations"] = float(len(scored))
        else:
            metrics = {"scored_observations": 0.0}
        result.selections.append(
            MonthlySelection(
                month=month, labeling=labeling, screener=screener, selected=selected,
                scores=dict(zip(symbols.tolist(), scores.tolist())),
                training_observations=n_train, training_months=len(train_months),
                hyperparameters=setting, inner_score=inner, metrics=metrics,
                # An empty selection is a real outcome, marked so the optimiser skips the
                # cell rather than receiving a substitute portfolio.
                skipped=None if selected else "empty selection",
            )
        )
    return result


def write_screening_outputs(
    results: Sequence[ScreeningResult], directory: str | Path
) -> dict[str, Path]:
    """Persist selections, decision scores, metrics and chosen hyperparameters.

    Four files rather than one, because they have different shapes and different readers:
    the optimiser consumes selections, the ablation consumes scores, the manuscript consumes
    metrics, and the reproducibility appendix consumes the chosen settings.
    """
    import json

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    selections, scores, metrics, settings = [], [], [], []
    for result in results:
        for s in result.selections:
            base = {"month": s.month, "labeling": s.labeling, "screener": s.screener}
            selections.append(
                {
                    **base,
                    "selection_size": s.size,
                    "training_observations": s.training_observations,
                    "training_months": s.training_months,
                    "skipped": s.skipped or "",
                    "selected": ";".join(s.selected),
                }
            )
            for symbol, score in s.scores.items():
                scores.append({**base, "provider_symbol": symbol, "decision_score": score})
            if s.metrics:
                metrics.append({**base, "selection_size": s.size, **s.metrics})
            settings.append(
                {
                    **base,
                    "hyperparameters": json.dumps(dict(s.hyperparameters), sort_keys=True),
                    "inner_score": s.inner_score if s.inner_score is not None else "",
                }
            )

    paths: dict[str, Path] = {}
    for name, rows in (
        ("selections", selections), ("scores", scores),
        ("metrics", metrics), ("hyperparameters", settings),
    ):
        path = directory / f"screening_{name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        paths[name] = path

    failures = [f for r in results for f in r.failures]
    failures_path = directory / "screening_failures.csv"
    pd.DataFrame(failures).to_csv(failures_path, index=False)
    paths["failures"] = failures_path
    return paths
