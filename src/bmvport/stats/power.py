"""Detectable-effect reporting for a short evaluation series.

With 18 evaluation blocks, a test that fails to reject is uninformative on its own: the
reader cannot tell whether the effect is absent or whether the study could never have seen
it. That distinction is the difference between a null result worth publishing and one worth
nothing, and it is settled by reporting the **minimum detectable effect** alongside every
non-rejection.

The claim a null result can then support is precise: *"an advantage of at least X basis
points per month would have been detected with 80% probability at this variance; none was
observed."* That is a bounded, defensible statement. "No significant difference" alone is
not, and referees are right to say so.

All functions here operate on the paired differences between two arms across evaluation
months, which is the design the blocked tests use.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "DetectableEffect",
    "minimum_detectable_effect",
    "achieved_power",
    "assess_non_rejection",
]


@dataclass(frozen=True, slots=True)
class DetectableEffect:
    """What the study could and could not have seen.

    Attributes:
        n_blocks: Paired observations, i.e. evaluation months.
        observed_sd: Standard deviation of the paired differences.
        observed_mean: Mean paired difference.
        mde: Smallest true difference detectable at ``alpha`` with probability ``power``.
        achieved_power: Probability this design would have detected the observed effect.
        alpha: Two-sided significance level.
        target_power: Power the MDE is computed for.
    """

    n_blocks: int
    observed_sd: float
    observed_mean: float
    mde: float
    achieved_power: float
    alpha: float
    target_power: float

    def interpretation(self) -> str:
        """One sentence a results table can carry verbatim."""
        return (
            f"an effect of at least {self.mde:.4g} would have been detected with "
            f"{self.target_power:.0%} probability across {self.n_blocks} blocks at the "
            f"observed dispersion; the observed difference was {self.observed_mean:.4g}"
        )


def minimum_detectable_effect(
    sd: float, n_blocks: int, *, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest true paired difference a two-sided test could detect.

    Uses the normal-approximation form with t critical values, which is the convention for
    reporting an MDE and is accurate enough at these sample sizes for its purpose: bounding
    what the design could see, not estimating an effect.

    Args:
        sd: Standard deviation of the paired differences.
        n_blocks: Number of paired observations.
        alpha: Two-sided significance level.
        power: Probability of detection the effect is sized for.

    Raises:
        ValueError: if fewer than two blocks, or if ``sd`` is negative. A single block
            admits no variance estimate and no test.
    """
    if n_blocks < 2:
        raise ValueError("a detectable effect needs at least two paired observations")
    if sd < 0:
        raise ValueError("sd must be non-negative")
    if sd == 0:
        return 0.0
    df = n_blocks - 1
    t_alpha = stats.t.ppf(1 - alpha / 2, df)
    t_beta = stats.t.ppf(power, df)
    return float((t_alpha + t_beta) * sd / np.sqrt(n_blocks))


def achieved_power(
    effect: float, sd: float, n_blocks: int, *, alpha: float = 0.05
) -> float:
    """Probability this design detects a true difference of ``effect``.

    Computed from the non-central t distribution rather than the normal approximation,
    because at 18 blocks the difference is not negligible and this number is reported.
    """
    if n_blocks < 2:
        raise ValueError("power needs at least two paired observations")
    if sd <= 0:
        return 1.0 if effect != 0 else 0.0
    df = n_blocks - 1
    ncp = float(effect) * np.sqrt(n_blocks) / float(sd)
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    upper = 1 - stats.nct.cdf(t_crit, df, ncp)
    lower = stats.nct.cdf(-t_crit, df, ncp)
    return float(np.clip(upper + lower, 0.0, 1.0))


def assess_non_rejection(
    differences: np.ndarray | list[float],
    *,
    alpha: float = 0.05,
    target_power: float = 0.80,
) -> DetectableEffect:
    """Characterise what a non-rejection does and does not rule out.

    Args:
        differences: Paired differences between two arms, one per evaluation block.
        alpha: Two-sided significance level used by the test that failed to reject.
        target_power: Power the reported minimum detectable effect is sized for.

    Returns:
        The detectable-effect record to store with the test result.

    Raises:
        ValueError: if fewer than two finite differences are supplied.
    """
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError(
            "at least two finite paired differences are required to characterise a "
            "non-rejection"
        )
    n_blocks = int(values.size)
    # Sample standard deviation: the population form would understate the uncertainty the
    # test actually faced.
    sd = float(values.std(ddof=1))
    mean = float(values.mean())
    return DetectableEffect(
        n_blocks=n_blocks,
        observed_sd=sd,
        observed_mean=mean,
        mde=minimum_detectable_effect(sd, n_blocks, alpha=alpha, power=target_power),
        achieved_power=achieved_power(mean, sd, n_blocks, alpha=alpha),
        alpha=alpha,
        target_power=target_power,
    )
