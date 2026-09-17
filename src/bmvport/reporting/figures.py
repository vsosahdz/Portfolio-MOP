"""Manuscript figures, generated from the stored results.

Each figure writes the data it was drawn from alongside the image, so a reader can check the
picture against the numbers rather than trusting the rendering. Nothing is drawn from an
intermediate stage.

Two conventions carry through, for the same reason they do in the tables: a performance
figure names its cost scenario inside the axes, and a benchmark series names its currency
basis. A return path without either is not interpretable, and legends drift from captions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

__all__ = [
    "figure_performance_by_factor",
    "figure_return_paths",
    "figure_attribution",
    "figure_pareto_front",
    "figure_hypervolume_dispersion",
    "figure_search_cost",
    "figure_critical_difference",
    "write_figures",
]

_PALETTE = ["#3B6FB6", "#C1553B", "#4E8F6A", "#8A6BAF", "#B5893A", "#6B7280"]


def _save(fig, path: Path, data: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    data.to_csv(path.with_suffix(".csv"), index=False)
    return path


def figure_performance_by_factor(
    results: pd.DataFrame, cost_bps: float, path: Path
) -> Path:
    """Distribution of monthly net returns by screener, with the baseline marked."""
    frame = results[
        (results["arm_kind"] == "proposed")
        & (results["cost_bps"] == cost_bps)
        & (~results["skip_reason"].astype(bool))
    ]
    series = {
        screener: group.groupby("month")["return_net"].median()
        for screener, group in frame.groupby("screener")
    }
    order = sorted(series, key=lambda s: -series[s].mean())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # matplotlib 3.11 renamed this argument; the study pins the version, so the current
    # spelling is used rather than a compatibility shim.
    ax.boxplot([series[s].to_numpy() * 100 for s in order], tick_labels=order, showmeans=True)
    ax.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--")
    ax.set_ylabel("monthly net return (%)")
    ax.set_title(f"Monthly net return by screener — {cost_bps:.0f} bps round trip")
    ax.tick_params(axis="x", rotation=30)
    data = pd.DataFrame({s: series[s] for s in order}).reset_index()
    return _save(fig, path, data)


def figure_return_paths(
    results: pd.DataFrame, cost_bps: float, path: Path
) -> Path:
    """Cumulative return paths against the reference series.

    Cumulative rather than compound: the same capital is committed each month, so the line
    is the sum of monthly returns and a single extraordinary month cannot dominate the shape.
    """
    evaluated = results[
        (results["cost_bps"] == cost_bps) & (~results["skip_reason"].astype(bool))
    ]
    lines: dict[str, pd.Series] = {}
    for screener, group in evaluated[evaluated["arm_kind"] == "proposed"].groupby("screener"):
        lines[f"proposed: {screener}"] = group.groupby("month")["return_net"].median().cumsum()
    for arm, group in evaluated[evaluated["arm_kind"] == "comparator"].groupby("allocator"):
        if arm in ("momentum_equal_weight", "bmv_ipc", "cetes_28d", "sp500_mxn"):
            label = arm + (" (in MXN)" if arm == "sp500_mxn" else "")
            label += " [uncosted]" if group["comparator_tier"].iloc[0] == "tier0" else ""
            lines[label] = group.groupby("month")["return_net"].median().cumsum()

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, series) in enumerate(sorted(lines.items())):
        style = "--" if label.startswith("proposed") else "-"
        ax.plot(series.index, series.to_numpy() * 100, style,
                color=_PALETTE[i % len(_PALETTE)], label=label, linewidth=1.6)
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.set_ylabel("cumulative net return (%)")
    ax.set_title(f"Cumulative net return — {cost_bps:.0f} bps round trip")
    ax.tick_params(axis="x", rotation=60)
    ax.legend(fontsize=7, ncol=2)
    data = pd.DataFrame(lines).reset_index()
    return _save(fig, path, data)


def figure_attribution(results: pd.DataFrame, cost_bps: float, path: Path) -> Path:
    """Market, selection, allocation and the currency tilt, shown as distinct components."""
    frame = results[
        (results["arm_kind"] == "proposed")
        & (results["cost_bps"] == cost_bps)
        & (~results["skip_reason"].astype(bool))
    ]
    rows = []
    for screener, group in frame.groupby("screener"):
        rows.append(
            {
                "screener": screener,
                "market": group["attribution_market"].mean() * 12 * 100,
                "selection": group["attribution_selection"].mean() * 12 * 100,
                "allocation": group["attribution_allocation"].mean() * 12 * 100,
                "fx tilt": group["attribution_fx_tilt"].mean() * 12 * 100,
            }
        )
    data = pd.DataFrame(rows).sort_values("selection")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width, x = 0.2, np.arange(len(data))
    for i, column in enumerate(["market", "selection", "allocation", "fx tilt"]):
        ax.bar(x + (i - 1.5) * width, data[column], width,
               label=column, color=_PALETTE[i])
    ax.axhline(0, color="#111827", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(data["screener"], rotation=30, ha="right")
    ax.set_ylabel("annualised contribution (%)")
    ax.set_title(f"Return attribution — {cost_bps:.0f} bps round trip")
    ax.legend(fontsize=8)
    return _save(fig, path, data)


def figure_pareto_front(
    weights: np.ndarray, objectives: np.ndarray, extracted: Sequence, path: Path
) -> Path:
    """One Pareto front with the three extracted portfolios marked."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(objectives[:, 1] * 100, -objectives[:, 0] * 100, s=18,
               color="#9CA3AF", label="front")
    for i, portfolio in enumerate(extracted):
        ax.scatter(portfolio.sigma_ex_ante * 100, portfolio.expected_return * 100,
                   s=90, marker="D", color=_PALETTE[i], label=portfolio.allocator, zorder=3)
    ax.set_xlabel("ex-ante risk, monthly (%)")
    ax.set_ylabel("ex-ante expected return, monthly (%)")
    ax.set_title("Pareto front with extracted portfolios")
    ax.legend(fontsize=8)
    data = pd.DataFrame({"sigma_ex_ante": objectives[:, 1], "expected_return": -objectives[:, 0]})
    return _save(fig, path, data)


def figure_hypervolume_dispersion(results: pd.DataFrame, path: Path) -> Path:
    """Across-seed hypervolume dispersion: a statement about optimiser stability.

    Reported separately from method comparison, because collapsing replicate noise into a
    method ranking is a category error.
    """
    frame = results[
        (results["arm_kind"] == "proposed") & (~results["skip_reason"].astype(bool))
        & results["hypervolume"].notna()
    ]
    grouped = frame.groupby(["labeling", "screener", "weight_cap", "month"])["hypervolume"]
    data = grouped.agg(["mean", "std", "count"]).reset_index().dropna()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(data["mean"], data["std"], s=10, alpha=0.35, color=_PALETTE[0])
    ax.set_xlabel("mean hypervolume across seeds")
    ax.set_ylabel("standard deviation across seeds")
    ax.set_title("Front-quality dispersion across independent seeds")
    return _save(fig, path, data)


def figure_search_cost(ablation: pd.DataFrame, path: Path) -> Path:
    """Wall-clock and attainment against the decision-variable count.

    The two panels are the ablation's engineering and tractability evidence. A flat
    attainment line is the informative outcome: it says the larger instances were solved as
    completely as the small ones, and therefore that the fixed-budget contrast is readable.
    """
    grouped = ablation.groupby("n_variables").agg(
        wall_clock=("wall_clock", "mean"),
        attainment=("attainment_at_base", "mean"),
        covariance_entries=("covariance_entries", "mean"),
    ).reset_index()

    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(grouped["n_variables"], grouped["wall_clock"], "o-", color=_PALETTE[0])
    left.set_xlabel("decision variables")
    left.set_ylabel("search wall-clock (s)")
    left.set_title("Search cost")
    right.plot(grouped["n_variables"], grouped["attainment"], "o-", color=_PALETTE[1])
    right.axhline(1.0, color="#6B7280", linewidth=0.8, linestyle="--")
    right.set_ylim(0, 1.15)
    right.set_xlabel("decision variables")
    right.set_ylabel("attainment at the main-grid budget")
    right.set_title("Tractability")
    fig.suptitle("Preselection: cost and tractability against problem size")
    return _save(fig, path, grouped)


def figure_critical_difference(
    results: pd.DataFrame, cost_bps: float, path: Path, *,
    headline_set_size: int = 10, alpha: float = 0.05,
) -> Path:
    """Critical-difference diagram over the reduced headline set.

    The reduction is not cosmetic. Nemenyi's interval widens as sqrt(k(k+1)/6N), so over
    eighteen blocks a diagram of all 216 cells has a critical difference of about 96 ranks
    and cannot separate anything -- it would look like a result while carrying none. The
    headline set is fixed in advance by the pre-registration: the best cell of each screener
    at the primary cost scenario, which is the comparison the screening claim is about.

    The critical difference and the block count are drawn on the axis, because a diagram
    without them invites reading adjacency as evidence of similarity when it is only a
    statement about the design's resolution.
    """
    from bmvport.stats.inference import critical_difference

    frame = results[
        (results["arm_kind"] == "proposed")
        & (results["cost_bps"] == cost_bps)
        & (~results["skip_reason"].astype(bool))
    ]
    per_cell = frame.groupby(["screener", "labeling", "allocator", "weight_cap", "month"])[
        "return_net"
    ].median().reset_index()
    best = (
        per_cell.groupby(["screener", "labeling", "allocator", "weight_cap"])["return_net"]
        .mean().reset_index().sort_values("return_net", ascending=False)
    )
    chosen = best.groupby("screener").head(1).head(headline_set_size)
    labels = [
        f"{r.screener}\n{r.allocator}, cap {r.weight_cap:g}" for r in chosen.itertuples()
    ]
    keys = [(r.screener, r.labeling, r.allocator, r.weight_cap) for r in chosen.itertuples()]

    columns = {}
    for label, (screener, labeling, allocator, cap) in zip(labels, keys):
        subset = per_cell[
            (per_cell["screener"] == screener) & (per_cell["labeling"] == labeling)
            & (per_cell["allocator"] == allocator) & (per_cell["weight_cap"] == cap)
        ]
        columns[label] = subset.set_index("month")["return_net"]
    matrix = pd.DataFrame(columns).dropna()
    # Rank 1 is the best month-wise outcome, so returns rank descending.
    ranks = matrix.rank(axis=1, ascending=False).mean().sort_values()
    n_blocks, n_levels = matrix.shape[0], matrix.shape[1]
    cd = critical_difference(n_levels, n_blocks, alpha)

    fig, ax = plt.subplots(figsize=(8, 0.55 * n_levels + 2.2))
    positions = np.arange(n_levels)[::-1]
    ax.scatter(ranks.to_numpy(), positions, s=70, color=_PALETTE[0], zorder=3)
    for position, (label, rank) in zip(positions, ranks.items()):
        ax.plot([rank - cd / 2, rank + cd / 2], [position, position],
                color=_PALETTE[5], linewidth=2.4, alpha=0.55, zorder=2)
        ax.text(rank, position + 0.28, f"{rank:.2f}", ha="center", fontsize=7)
    ax.set_yticks(positions)
    ax.set_yticklabels(ranks.index, fontsize=8)
    ax.set_xlabel("mean rank across evaluation months (1 = best)")
    ax.set_title(
        f"Critical-difference diagram — {cost_bps:.0f} bps\n"
        f"CD = {cd:.2f} ranks at alpha = {alpha:g}, {n_blocks} blocks, {n_levels} cells; "
        f"the bar spans one CD"
    )
    ax.set_xlim(0.5, n_levels + 0.5)
    ax.grid(axis="x", alpha=0.25)
    data = ranks.rename("mean_rank").reset_index().rename(columns={"index": "cell"})
    data["critical_difference"] = cd
    data["n_blocks"] = n_blocks
    data["separated_from_best"] = (ranks.to_numpy() - ranks.min()) > cd
    return _save(fig, path, data)


def write_figures(
    results: pd.DataFrame, cost_bps: float, directory: str | Path,
    ablation: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write every manuscript figure and the data behind it."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "performance": figure_performance_by_factor(
            results, cost_bps, directory / "fig_performance.png"),
        "return_paths": figure_return_paths(
            results, cost_bps, directory / "fig_return_paths.png"),
        "attribution": figure_attribution(
            results, cost_bps, directory / "fig_attribution.png"),
        "hypervolume": figure_hypervolume_dispersion(
            results, directory / "fig_hypervolume_dispersion.png"),
        "critical_difference": figure_critical_difference(
            results, cost_bps, directory / "fig_critical_difference.png"),
    }
    if ablation is not None and len(ablation):
        paths["search_cost"] = figure_search_cost(
            ablation, directory / "fig_search_cost.png")
    return paths
