"""Compatibility layer for the reference PBC4cip implementation.

PBC4cip is the contrast-pattern classifier of Loyola-Gonzalez et al., and it is a required
component of this study rather than a convenience: it was the screener that performed best
on the conservative profile in the original work, and one of its authors is an author here.
Dropping it silently is not an acceptable outcome, so this module exists to make the
reference implementation usable rather than to replace it.

**What is patched and why.** Release 0.0.0.8 calls ``np.asarray(..., dtype=np.object)`` in
two helper functions. ``np.object`` was removed in NumPy 1.24; it was never anything other
than an alias for the builtin ``object``, so restoring the intended behaviour is a
mechanical fix with no effect on the algorithm.

**Why surgically, and not by restoring the global alias.** Setting ``numpy.object = object``
would work today, but NumPy emits a ``FutureWarning`` stating that ``np.object`` will be
reintroduced as a NumPy scalar type. A global alias would then shadow a real attribute and
change the meaning of unrelated code. Patching the two call sites keeps the blast radius at
exactly the two lines that are wrong.

**Disclosure.** The patch is recorded as a substitution with ``affects_results=False``,
because the algorithm is the authors' own and its behaviour is unchanged. If the patch ever
has to become an actual model swap, that entry flips to ``True`` and the manuscript must
report it wherever those results appear.

Two API details of the reference implementation are handled by the caller
(``screening.screeners``), not here, but are documented for the reader:

- ``predict`` returns **class indices**, not class labels. The label order is available as
  ``fitted.dataset.Class[1]``. Comparing raw ``predict`` output against string labels
  silently yields zero accuracy.
- ``score_samples`` returns an ``(n_samples, n_classes)`` array of per-class scores. This is
  the continuous decision score the selection-size ablation truncates on.
"""

from __future__ import annotations

from itertools import chain
from typing import Any

import numpy as np
import pandas as pd

from ..manifest import Substitution

__all__ = [
    "PBC4CIP_AVAILABLE",
    "PBC4CIP_UNAVAILABLE_REASON",
    "get_pbc4cip_class",
    "pbc4cip_substitution",
    "decode_predictions",
]

PBC4CIP_AVAILABLE: bool = False
PBC4CIP_UNAVAILABLE_REASON: str | None = None

_PATCH_DESCRIPTION = (
    "replaced np.object with the builtin object in "
    "PBC4cip.core.Helpers.combine_instances and .convert_to_ndarray "
    "(np.object was removed in NumPy 1.24; it was an alias for object)"
)


def _combine_instances(X: Any, y: Any) -> np.ndarray:
    """Corrected ``PBC4cip.core.Helpers.combine_instances``."""
    return np.asarray(
        [list(chain(*[value, y[i]])) for i, value in enumerate(X)], dtype=object
    )


def _convert_to_ndarray(y: Any) -> pd.Series:
    """Corrected ``PBC4cip.core.Helpers.convert_to_ndarray``."""
    return pd.Series([np.array([element], dtype=object) for element in y], name="class")


def _apply_patch() -> None:
    """Patch the two defective helpers in place.

    ``DecisionTreeBuilder`` imports ``combine_instances`` by name, so rebinding it on the
    ``Helpers`` module alone would leave the stale reference in place. Both bindings are
    replaced.
    """
    import PBC4cip.core.DecisionTreeBuilder as decision_tree_builder
    import PBC4cip.core.Helpers as helpers

    helpers.combine_instances = _combine_instances
    helpers.convert_to_ndarray = _convert_to_ndarray
    decision_tree_builder.combine_instances = _combine_instances


try:  # pragma: no cover - exercised by environment, not by unit tests
    _apply_patch()
    from PBC4cip import PBC4cip as _PBC4cip

    PBC4CIP_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - any import-time failure disables the screener
    _PBC4cip = None  # type: ignore[assignment]
    PBC4CIP_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"


def get_pbc4cip_class() -> type:
    """Return the patched reference PBC4cip class.

    Raises:
        RuntimeError: if the reference implementation is unusable. The caller is expected
            to record a substitution and disclose it, not to fall back silently.
    """
    if not PBC4CIP_AVAILABLE or _PBC4cip is None:
        raise RuntimeError(
            "the reference PBC4cip implementation is unusable in this environment "
            f"({PBC4CIP_UNAVAILABLE_REASON}); a substitution must be recorded and "
            "disclosed in the manuscript rather than applied silently"
        )
    return _PBC4cip


def pbc4cip_substitution() -> Substitution | None:
    """Describe how PBC4cip was obtained, for the run manifest.

    Returns ``None`` when no adjustment was needed at all. When the reference
    implementation is used behind the compatibility patch, the substitution is recorded
    with ``affects_results=False``: the algorithm is unchanged, but the reader is entitled
    to know the installed package was not used verbatim.
    """
    if PBC4CIP_AVAILABLE:
        return Substitution(
            component="pbc4cip",
            used_instead="PBC4cip reference implementation with a compatibility patch",
            reason=_PATCH_DESCRIPTION,
            affects_results=False,
        )
    return Substitution(
        component="pbc4cip",
        used_instead="none - screener unavailable",
        reason=(
            "reference implementation could not be imported "
            f"({PBC4CIP_UNAVAILABLE_REASON})"
        ),
        affects_results=True,
    )


def decode_predictions(raw: Any, class_labels: list[Any]) -> np.ndarray:
    """Map PBC4cip's class-index output onto class labels.

    ``predict`` returns positions in ``fitted.dataset.Class[1]``. Callers that compare its
    output directly against string labels get zero accuracy with no error raised, so the
    mapping is centralised here.

    Args:
        raw: Whatever ``PBC4cip.predict`` returned.
        class_labels: Ordered class labels, i.e. ``fitted.dataset.Class[1]``.

    Returns:
        Array of class labels, one per sample.
    """
    indices = np.asarray(raw).ravel().astype(int)
    labels = np.asarray(class_labels, dtype=object)
    if indices.size and (indices.min() < 0 or indices.max() >= labels.size):
        raise ValueError(
            f"PBC4cip returned class index outside the label range 0..{labels.size - 1}"
        )
    return labels[indices]
