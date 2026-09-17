"""Extracting portfolios from a Pareto front, and measuring the front itself.

Three portfolios are taken from each front: the two extremes, and a **knee point** in place
of the original study's "average risk" pick. That pick was index 50 of a hundred solutions --
a position in a list, not a preference model, and not defensible as one. The knee is the
solution where the trade-off turns: moving in either direction costs more than it gains.

**Hypervolume is recorded for every front**, because the original generates a
hundred-solution front and then discards ninety-seven of them without ever asking whether
the search found a good front or a poor one whose extreme happened to pay off. Reporting it
also connects the finance application to standard multi-objective assessment practice.

**Raw hypervolume is not comparable across problem instances.** Different months, and
different selection sizes in the ablation, mean different assets, different fronts and
different objective ranges. Objectives are therefore min-max normalised **per instance**
before the indicator is computed, and the ablation compares attainment ratios against an
instance's own reference front rather than comparing absolute values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ExtractedPortfolio",
    "cap_binding",
    "normalise_objectives",
    "hypervolume",
    "extract_portfolios",
    "knee_index",
]


@dataclass(frozen=True, slots=True)
class ExtractedPortfolio:
    """One portfolio taken from a front."""

    allocator: str
    weights: np.ndarray
    expected_return: float
    sigma_ex_ante: float
    front_index: int


def normalise_objectives(objectives: np.ndarray) -> np.ndarray:
    """Min-max normalise each objective onto ``[0, 1]`` within this instance.

    A degenerate objective -- every solution identical on it -- maps to zero rather than
    dividing by a zero range.
    """
    points = np.atleast_2d(np.asarray(objectives, dtype=float))
    low = points.min(axis=0)
    high = points.max(axis=0)
    span = high - low
    out = np.zeros_like(points)
    usable = span > 1e-12
    out[:, usable] = (points[:, usable] - low[usable]) / span[usable]
    return out


def hypervolume(objectives: np.ndarray, reference: tuple[float, float]) -> float:
    """Hypervolume of a minimisation front, computed on normalised objectives.

    Args:
        objectives: ``(k, 2)`` objective values.
        reference: Reference point in normalised space, outside the unit square.

    Returns:
        The indicator value, or 0.0 for an empty front.
    """
    points = np.atleast_2d(np.asarray(objectives, dtype=float))
    if points.size == 0:
        return 0.0
    from pymoo.indicators.hv import HV

    return float(HV(ref_point=np.asarray(reference, dtype=float))(normalise_objectives(points)))


def knee_index(objectives: np.ndarray) -> int | None:
    """Index of the knee: the point furthest from the chord joining the extremes.

    Operationalised as maximum perpendicular distance to the line between the two extreme
    solutions of the normalised front. This is the standard construction and it is stable,
    which matters because a second-derivative estimate on a front of unevenly spaced points
    is not.

    Returns ``None`` when the front has fewer than three points, or when the extremes
    coincide so that no chord exists. The caller records the knee as absent rather than
    substituting an arbitrary solution.
    """
    points = np.atleast_2d(np.asarray(objectives, dtype=float))
    if len(points) < 3:
        return None
    normalised = normalise_objectives(points)
    order = np.argsort(normalised[:, 0])
    ordered = normalised[order]
    start, end = ordered[0], ordered[-1]
    chord = end - start
    length = float(np.linalg.norm(chord))
    if length < 1e-12:
        return None
    # Perpendicular distance from each point to the chord, in two dimensions.
    offsets = ordered - start
    distances = np.abs(chord[0] * offsets[:, 1] - chord[1] * offsets[:, 0]) / length
    return int(order[int(np.argmax(distances))])


def extract_portfolios(
    weights: np.ndarray, objectives: np.ndarray
) -> list[ExtractedPortfolio]:
    """Take the minimum-risk, knee and maximum-return portfolios from a front.

    The knee is omitted when the front cannot support one, rather than falling back to a
    midpoint -- which would reintroduce exactly the arbitrary pick it replaced.
    """
    W = np.atleast_2d(np.asarray(weights, dtype=float))
    F = np.atleast_2d(np.asarray(objectives, dtype=float))
    if len(F) == 0:
        return []

    def make(name: str, index: int) -> ExtractedPortfolio:
        return ExtractedPortfolio(
            allocator=name,
            weights=W[index],
            expected_return=float(-F[index, 0]),
            sigma_ex_ante=float(F[index, 1]),
            front_index=int(index),
        )

    out = [
        make("ga_min_risk", int(np.argmin(F[:, 1]))),
        make("ga_max_return", int(np.argmin(F[:, 0]))),
    ]
    knee = knee_index(F)
    if knee is not None:
        out.append(make("ga_knee", knee))
    return sorted(out, key=lambda p: p.allocator)


def cap_binding(unconstrained_weights: np.ndarray, cap: float, *, tolerance: float = 1e-6) -> bool:
    """Whether a weight cap actually constrained anything.

    Measured on this universe, the unconstrained optimum reaches a maximum weight of 0.69 on
    a five-asset selection and 0.048 on a ninety-asset one. A cap of 0.20 therefore binds on
    small selections and is inert on large ones -- which makes the cap factor's effect
    conditional on the screener's selection size rather than independent of it.

    Reporting a null cap effect without saying that the cap never bound in most cells would
    describe a design artefact as a finding. This flag is recorded per cell so the reported
    effect can be conditioned on the cells where the constraint was live.

    Args:
        unconstrained_weights: Weights from the corresponding uncapped front.
        cap: The cap that would be applied.

    Returns:
        True when any solution of the unconstrained front exceeds the cap.
    """
    if cap >= 1.0:
        return False
    return bool(np.max(np.asarray(unconstrained_weights, dtype=float)) > cap + tolerance)
