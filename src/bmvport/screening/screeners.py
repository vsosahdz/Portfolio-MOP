"""Screeners behind one interface.

Seven levels, chosen as a taxonomy rather than accumulated:

``all_stocks``
    The no-screening baseline. Selects everything, so the factorial can ask whether
    screening changes anything at all -- the study's central question.
``svm``, ``random_forest``, ``neural_network``, ``pbc4cip``
    The original study's set, kept for methodological fidelity. PBC4cip also earns its place
    empirically: the proposed labeling runs 4.7 negatives per positive on this window, and
    class imbalance is what it was designed for.
``xgboost``
    Modern gradient boosting, the standard strong tabular baseline. Measured on this feature
    matrix at balanced accuracy 0.689, against 0.646 for scikit-learn's histogram booster and
    0.619 for CatBoost, so one member carries the family instead of three variants of the
    same idea.
``tabpfn``
    A tabular foundation model, the current state of the art in the small-sample regime this
    study occupies. Pinned to the **public v2 checkpoint**: version 8.5 defaults to gated
    models needing both a Prior Labs licence and a HuggingFace account with repository
    access, while the v2 weights download with no token at all. Trading a model generation
    for a study anyone can reproduce is the right side of that bargain, and the manuscript
    says so rather than leaving the choice unexplained.

Every screener exposes ``fit``, ``predict`` and ``decision_score``. The score is what the
selection-size ablation truncates on, so a screener without one cannot participate in it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from ..manifest import Substitution
from ._pbc4cip_compat import (
    PBC4CIP_AVAILABLE,
    PBC4CIP_UNAVAILABLE_REASON,
    decode_predictions,
    get_pbc4cip_class,
)

__all__ = [
    "Screener",
    "PBC4CIP_OMISSION",
    "SCREENER_BUILDERS",
    "build_screener",
    "screener_availability",
    "availability_substitutions",
    "TABPFN_TOKEN_ENV_VAR",
]

TABPFN_TOKEN_ENV_VAR = "TABPFN_TOKEN"


class Screener(Protocol):
    """What every screener must provide."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Screener": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    def decision_score(self, X: np.ndarray) -> np.ndarray: ...


@dataclass(slots=True)
class _SklearnScreener:
    """Adapter over any scikit-learn compatible estimator."""

    estimator: Any
    _classes: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_SklearnScreener":
        self.estimator.fit(X, y)
        self._classes = np.asarray(self.estimator.classes_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict(X)).astype(bool)

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        """Probability of the positive class, for the ablation's top-k truncation."""
        proba = self.estimator.predict_proba(X)
        if self._classes is None:
            raise RuntimeError("decision_score called before fit")
        positive = int(np.argmax(self._classes.astype(int)))
        return np.asarray(proba)[:, positive]


@dataclass(slots=True)
class AllStocksScreener:
    """The no-screening baseline: every candidate is selected.

    Its decision score is constant, which makes it ineligible for the selection-size
    ablation -- a fact the ablation checks rather than discovering through a meaningless
    ranking.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AllStocksScreener":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X), dtype=bool)

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        return np.ones(len(X), dtype=float)


@dataclass(slots=True)
class PBC4cipScreener:
    """Adapter over the reference contrast-pattern classifier.

    Two API traps are handled here rather than left to callers: ``predict`` returns class
    indices rather than labels, and ``score_samples`` returns one column per class.
    """

    tree_count: int = 100
    _model: Any = None
    _labels: list[Any] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PBC4cipScreener":
        import pandas as pd

        frame = pd.DataFrame(np.asarray(X, dtype=float))
        frame.columns = [f"f{i}" for i in range(frame.shape[1])]
        target = pd.DataFrame({"class": np.asarray(y).astype(int).astype(str)})
        self._model = get_pbc4cip_class()(tree_count=self.tree_count)
        self._model.fit(frame, target)
        self._labels = list(self._model.dataset.Class[1])
        return self

    def _frame(self, X: np.ndarray):
        import pandas as pd

        frame = pd.DataFrame(np.asarray(X, dtype=float))
        frame.columns = [f"f{i}" for i in range(frame.shape[1])]
        return frame

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._labels is None:
            raise RuntimeError("predict called before fit")
        decoded = decode_predictions(self._model.predict(self._frame(X)), self._labels)
        return np.asarray([str(v) == "1" for v in decoded], dtype=bool)

    def decision_score(self, X: np.ndarray) -> np.ndarray:
        if self._model is None or self._labels is None:
            raise RuntimeError("decision_score called before fit")
        scores = np.asarray(self._model.score_samples(self._frame(X)), dtype=float)
        positive = self._labels.index("1") if "1" in self._labels else scores.shape[1] - 1
        column = scores[:, positive]
        total = scores.sum(axis=1)
        # Contrast-pattern supports are unnormalised; a share of total support is the
        # comparable quantity for ranking.
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(total > 0, column / total, 0.0)
        return share


def _build_svm(**params: Any) -> Screener:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    # Scaling is not optional for an SVM on 129 heterogeneous indicator columns, and the
    # thesis's grid over kernels and degrees is meaningless without it.
    params.setdefault("probability", True)
    params.setdefault("class_weight", "balanced")
    params.setdefault("random_state", 0)
    return _SklearnScreener(make_pipeline(StandardScaler(), SVC(**params)))


def _build_random_forest(**params: Any) -> Screener:
    from sklearn.ensemble import RandomForestClassifier

    params.setdefault("n_estimators", 200)
    params.setdefault("class_weight", "balanced")
    params.setdefault("random_state", 0)
    params.setdefault("n_jobs", 1)
    return _SklearnScreener(RandomForestClassifier(**params))


def _build_neural_network(**params: Any) -> Screener:
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    params.setdefault("max_iter", 500)
    params.setdefault("random_state", 0)
    if isinstance(params.get("hidden_layer_sizes"), list):
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    return _SklearnScreener(make_pipeline(StandardScaler(), MLPClassifier(**params)))


def _build_xgboost(**params: Any) -> Screener:
    import xgboost as xgb

    params.setdefault("n_estimators", 200)
    params.setdefault("random_state", 0)
    params.setdefault("verbosity", 0)
    params.setdefault("n_jobs", 1)
    params.setdefault("eval_metric", "logloss")
    return _SklearnScreener(xgb.XGBClassifier(**params))


def _build_pbc4cip(**params: Any) -> Screener:
    return PBC4cipScreener(tree_count=int(params.get("tree_count", 100)))


def _build_tabpfn(**params: Any) -> Screener:
    """Tabular foundation model, pinned to the public v2 checkpoint.

    TabPFN 8.5 defaults to gated model versions that require both a Prior Labs licence and a
    HuggingFace account with access granted to the repository. The v2 checkpoints are public
    and were verified to download with no token of any kind, so the study -- and anyone
    reproducing it -- needs no registration. That is a deliberate trade-off of model
    generation for reproducibility, and the manuscript states it as such.

    The checkpoint is required rather than defaulted, because ``model_path="auto"`` would
    silently resolve to a gated model and reintroduce the barrier.
    """
    from tabpfn import TabPFNClassifier

    if not params.get("model_path"):
        raise RuntimeError(
            "tabpfn requires an explicit model_path; leaving it unset resolves to a gated "
            "checkpoint that needs a Prior Labs licence and a HuggingFace account, which "
            "would make the study unreproducible without registration"
        )
    return _SklearnScreener(TabPFNClassifier(**params))


SCREENER_BUILDERS = {
    "all_stocks": lambda **_: AllStocksScreener(),
    "svm": _build_svm,
    "random_forest": _build_random_forest,
    "neural_network": _build_neural_network,
    "pbc4cip": _build_pbc4cip,
    "xgboost": _build_xgboost,
    "tabpfn": _build_tabpfn,
}


def build_screener(name: str, **params: Any) -> Screener:
    """Construct a screener by name.

    Raises:
        ValueError: for an unknown name.
        RuntimeError: when the screener exists but cannot run in this environment. The
            caller records a substitution and discloses it; it never falls back silently.
    """
    if name not in SCREENER_BUILDERS:
        raise ValueError(f"unknown screener {name!r}; known: {sorted(SCREENER_BUILDERS)}")
    return SCREENER_BUILDERS[name](**params)


def screener_availability(names: Sequence[str]) -> dict[str, tuple[bool, str]]:
    """Report which screeners can actually run, and why not when they cannot."""
    status: dict[str, tuple[bool, str]] = {}
    for name in names:
        if name == "pbc4cip":
            status[name] = (
                PBC4CIP_AVAILABLE,
                "" if PBC4CIP_AVAILABLE else str(PBC4CIP_UNAVAILABLE_REASON),
            )
        elif name == "tabpfn":
            try:
                import tabpfn  # noqa: F401
            except Exception as exc:  # noqa: BLE001
                status[name] = (False, f"not installed: {exc}")
                continue
            # The public checkpoint needs no credentials, so availability is installation
            # alone. A gated checkpoint would reintroduce the token requirement, which is
            # why build_screener refuses to run without an explicit model_path.
            status[name] = (True, "")
        else:
            status[name] = (True, "")
    return status


# PBC4cip is omitted from the run, and the omission is a decision with evidence rather than
# a gap. Measured cost at its reference ensemble size (tree_count=100, the implementation's
# own default): 152 minutes per fit at the final training size, 137 hours for the grid's 54
# fits. Parallelising the independent monthly fits broke the process pool at six and ten
# workers, and memory was ruled out -- the footprint is flat at 19 MB regardless of ensemble
# size, against 24 GB available.
#
# What settles it is that no conclusion depends on it. H1 pools the monthly medians of the
# machine-learning screeners against the no-screening baseline. Injecting a seventh screener
# returning 100% annualised -- three and a half times the strongest comparator -- moves the
# pooled effect from -8.94% to -6.40% and leaves it below the design's resolution:
#
#     PBC4cip returns      pooled H1      MDE     resolvable
#                   0%        -8.19%    15.4%     no
#                  28%        -7.19%    15.1%     no
#                 100%        -6.40%    15.5%     no
#
# There is also a mechanistic reason to expect nothing: PBC4cip consumes the same features,
# whose forward component measures AUC 0.470. Its design advantage is handling class
# imbalance, not extracting signal that is absent.
#
# The manuscript reports the omission, its measured cost and this analysis. A seventh
# screener run at a tenth of its reference ensemble would need a caveat in every table it
# appears in, which is a worse trade.
PBC4CIP_OMISSION = Substitution(
    component="pbc4cip",
    used_instead="none - omitted from the screener set",
    reason=(
        "152 min per fit at reference tree_count=100, 137 h for the grid; parallel pool "
        "unstable at 6 and 10 workers. No conclusion depends on it: a seventh screener "
        "returning 100% annualised would move the pooled H1 effect from -8.94% to -6.40%, "
        "still below the 15% detectable bound."
    ),
    affects_results=True,
)


def availability_substitutions(names: Sequence[str]) -> list[Substitution]:
    """Substitutions to record for screeners that cannot run.

    An unavailable screener is a hole in the factorial, and a hole that is not disclosed
    looks like a screener that was tried and did poorly.
    """
    out: list[Substitution] = []
    if "pbc4cip" not in names:
        out.append(PBC4CIP_OMISSION)
    for name, (available, reason) in screener_availability(names).items():
        if available:
            continue
        out.append(
            Substitution(
                component=name,
                used_instead="none - screener omitted from the factorial",
                reason=reason,
                affects_results=True,
            )
        )
    return out
