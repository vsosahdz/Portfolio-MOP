"""The PBC4cip compatibility layer, and the two API traps it documents.

These tests exist because PBC4cip is a required component of the study, not an optional
one, and because both of its API quirks fail silently rather than loudly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bmvport.screening import _pbc4cip_compat as compat

pytestmark = pytest.mark.skipif(
    not compat.PBC4CIP_AVAILABLE,
    reason=f"PBC4cip unusable: {compat.PBC4CIP_UNAVAILABLE_REASON}",
)


def test_global_numpy_alias_is_not_restored() -> None:
    """The patch must be surgical.

    NumPy warns that ``np.object`` will be reintroduced as a scalar type, so a global alias
    would eventually shadow a real attribute and change unrelated code's meaning.
    """
    assert "object" not in dir(np) or not isinstance(getattr(np, "object", None), type(object))


def test_patch_targets_both_bindings() -> None:
    """DecisionTreeBuilder imports the helper by name, so one rebinding is not enough."""
    import PBC4cip.core.DecisionTreeBuilder as dtb
    import PBC4cip.core.Helpers as helpers

    assert helpers.combine_instances is compat._combine_instances
    assert dtb.combine_instances is compat._combine_instances
    assert helpers.convert_to_ndarray is compat._convert_to_ndarray


def test_fit_predict_on_imbalanced_data() -> None:
    """Roughly the class balance of historical_and_t1 (~25% positives)."""
    rng = np.random.default_rng(0)
    n = 240
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.DataFrame({"class": np.where(X["f0"] > 0.7, "True", "False")})
    assert 0.1 < (y["class"] == "True").mean() < 0.4

    clf = compat.get_pbc4cip_class()(tree_count=15)
    clf.fit(X, y)

    labels = list(clf.dataset.Class[1])
    decoded = compat.decode_predictions(clf.predict(X), labels)
    assert decoded.shape == (n,)
    assert set(decoded.tolist()) <= set(labels)
    # Separable by construction; a correct decode must recover most of it.
    assert (decoded == y["class"].to_numpy()).mean() > 0.9


def test_raw_predict_returns_indices_not_labels() -> None:
    """Documents the trap: comparing raw output to labels yields 0 accuracy, no error."""
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(150, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.DataFrame({"class": np.where(X["f0"] > 0.5, "True", "False")})

    clf = compat.get_pbc4cip_class()(tree_count=10)
    clf.fit(X, y)
    raw = np.asarray(clf.predict(X)).ravel()

    assert set(np.unique(raw).astype(str)) <= {"0", "1"}
    naive_accuracy = (raw.astype(str) == y["class"].to_numpy().astype(str)).mean()
    assert naive_accuracy == 0.0


def test_score_samples_gives_the_ablation_its_decision_score() -> None:
    """The selection-size sweep truncates on a continuous score; this is its source."""
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.DataFrame({"class": np.where(X["f0"] > 0.5, "True", "False")})

    clf = compat.get_pbc4cip_class()(tree_count=10)
    clf.fit(X, y)
    scores = np.asarray(clf.score_samples(X))

    assert scores.shape == (120, len(clf.dataset.Class[1]))
    assert np.issubdtype(scores.dtype, np.floating)
    assert np.isfinite(scores).all()


def test_decode_rejects_out_of_range_index() -> None:
    with pytest.raises(ValueError, match="outside the label range"):
        compat.decode_predictions(np.array([0, 1, 7]), ["False", "True"])


def test_substitution_is_disclosed_but_marked_result_neutral() -> None:
    substitution = compat.pbc4cip_substitution()
    assert substitution is not None
    assert substitution.component == "pbc4cip"
    assert substitution.affects_results is False
    assert "np.object" in substitution.reason
