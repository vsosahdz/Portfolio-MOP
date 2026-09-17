"""Command-line entry point.

Stages run in dependency order and each reads the previous stage's stored artefacts, so
they are individually invocable:

    bmvport --config configs/smoke.yaml stage universe
    bmvport --config configs/smoke.yaml stage features
    bmvport --config configs/default.yaml run-all

Stages not yet implemented fail with an explicit message naming the task group that builds
them. That is deliberate: a stage silently doing nothing would let a later stage read an
empty artefact and produce results that look complete.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Sequence

from .config import RunConfig, load_config
from .manifest import build_manifest, write_manifest
from .marketdata import rates as rates_stage
from .screening._pbc4cip_compat import pbc4cip_substitution


def _stage_rates(config: RunConfig) -> None:
    """Retrieve the Banxico series the evaluation and comparators depend on."""
    for label in ("cetes_28d", "fx_fix"):
        frame = rates_stage.fetch_series(
            label,
            config.windows.download_start,
            config.windows.download_end,
            config.paths.cache,
        )
        spec = rates_stage.SIE_SERIES[label]
        print(
            f"  {label:10s} {spec.series_id}  {len(frame):5d} obs  "
            f"{frame.index.min().date()}..{frame.index.max().date()}"
        )
    monthly = rates_stage.monthly_risk_free(
        list(config.windows.eval_months()), config.paths.cache
    )
    print(
        f"  risk-free: {len(monthly)} months, mean {monthly.mean() * 100:.4f}% per month"
    )


def _stage_quality(config: RunConfig) -> None:
    """Assess every cached ticker-month and persist the verdicts."""
    from .marketdata.prices import cache_path, load_prices
    from .marketdata.quality import assess_ticker_months, find_stale_runs, quality_report
    from .marketdata.universe import read_candidates

    assessments = []
    stale_runs = []
    for candidate in read_candidates(config.paths.universe):
        if not cache_path(config.paths.cache, candidate.provider_symbol).exists():
            continue
        frame = load_prices(config.paths.cache, candidate.provider_symbol)
        if frame.empty or "close" not in frame:
            continue
        assessments.extend(
            assess_ticker_months(
                frame,
                candidate.provider_symbol,
                config.universe,
                min_trading_days=config.features.min_trading_days_per_month,
            )
        )
        stale_runs.extend(
            find_stale_runs(
                frame["close"].dropna(),
                candidate.provider_symbol,
                min_sessions=config.universe.max_stale_run_days,
            )
        )
    months_path, runs_path = quality_report(assessments, stale_runs, config.paths.universe)
    rejected = sum(1 for a in assessments if not a.usable)
    first_eval = config.windows.eval_months()[0]
    in_eval = [a for a in assessments if a.month >= first_eval]
    rejected_in_eval = sum(1 for a in in_eval if not a.usable)
    print(
        f"  {len(assessments)} ticker-months assessed, {rejected} rejected "
        f"({rejected / max(len(assessments), 1):.1%}); {len(stale_runs)} stale runs"
    )
    # The headline rate spans every candidate, including SIC instruments that are quoted but
    # dormant and that the liquidity screen removes anyway. Reported alone it reads as a data
    # disaster; the figure that matters is the rate inside the evaluation window, and the
    # rate over the screened universe once the universe stage exists.
    print(
        f"  within the evaluation window ({first_eval} onward): {len(in_eval)} assessed, "
        f"{rejected_in_eval} rejected ({rejected_in_eval / max(len(in_eval), 1):.1%})"
    )
    print(
        "  note: the headline rate covers all candidates, most of them dormant SIC lines "
        "the liquidity screen excludes; see the per-instrument detail in the report"
    )
    print(f"  wrote {months_path.name}, {runs_path.name}")


def _stage_screen(config: RunConfig) -> None:
    """Admit instruments over the evaluation window, then screen liquidity per month."""
    from .marketdata.prices import cache_path, load_prices
    from .marketdata.screen import run_screen, write_screen_outputs
    from .marketdata.universe import read_candidates

    frames: dict[str, object] = {}
    listings: dict[str, str] = {}
    for candidate in read_candidates(config.paths.universe):
        if not cache_path(config.paths.cache, candidate.provider_symbol).exists():
            continue
        frame = load_prices(config.paths.cache, candidate.provider_symbol)
        if frame.empty or "close" not in frame:
            continue
        frames[candidate.provider_symbol] = frame
        listings[candidate.provider_symbol] = candidate.listing

    admissions, memberships = run_screen(frames, listings, config)
    admitted = sum(1 for a in admissions if a.admitted)
    print(f"  static admission over the evaluation window: {admitted}/{len(admissions)}")

    sizes = [m.size for m in memberships]
    flagged = [m.month for m in memberships if m.below_minimum]
    print(
        f"  monthly membership across {len(memberships)} months: "
        f"min {min(sizes)}, median {sorted(sizes)[len(sizes) // 2]}, max {max(sizes)}"
    )
    national = {s for s, kind in listings.items() if kind == "national"}
    last = memberships[-1]
    print(
        f"  {last.month}: {last.size} instruments "
        f"({len(set(last.members) & national)} national, "
        f"{last.size - len(set(last.members) & national)} SIC)"
    )
    if flagged:
        print(
            f"  WARNING: {len(flagged)} month(s) below the configured minimum of "
            f"{config.universe.min_universe_size}: {', '.join(flagged)}"
        )
    paths = write_screen_outputs(admissions, memberships, config.paths.universe)
    print("  wrote " + ", ".join(p.name for p in paths.values()))


def _stage_features(config: RunConfig) -> None:
    """Build the monthly feature matrix over instruments the screen admitted."""
    import csv

    from .features.monthly import build_feature_matrix, write_feature_dictionary
    from .marketdata.prices import cache_path, load_prices
    from .marketdata.quality import assess_ticker_months
    from .marketdata.universe import read_candidates

    admitted: set[str] = set()
    admission_file = config.paths.universe / "screen_admission.csv"
    if admission_file.exists():
        with admission_file.open(encoding="utf-8") as handle:
            admitted = {
                row["provider_symbol"]
                for row in csv.DictReader(handle)
                if row["admitted"] == "True"
            }
    if not admitted:
        raise RuntimeError(
            "no admitted instruments found; run `stage screen` before `stage features`"
        )

    frames = {}
    usable: dict[tuple[str, str], bool] = {}
    for candidate in read_candidates(config.paths.universe):
        symbol = candidate.provider_symbol
        if symbol not in admitted:
            continue
        if not cache_path(config.paths.cache, symbol).exists():
            continue
        frame = load_prices(config.paths.cache, symbol)
        if frame.empty or "close" not in frame:
            continue
        frames[symbol] = frame
        for assessment in assess_ticker_months(
            frame,
            symbol,
            config.universe,
            min_trading_days=config.features.min_trading_days_per_month,
        ):
            usable[(symbol, assessment.month)] = assessment.usable

    matrix, exclusions = build_feature_matrix(frames, config.features, usable=usable)
    config.paths.features.mkdir(parents=True, exist_ok=True)
    matrix_path = config.paths.features / "monthly_features.parquet"
    matrix.to_parquet(matrix_path)

    dictionary_path = write_feature_dictionary(
        config.features, config.paths.features / "feature_dictionary.json"
    )
    exclusions_path = config.paths.features / "feature_exclusions.csv"
    with exclusions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["provider_symbol", "month", "sessions", "reason"])
        for e in exclusions:
            writer.writerow([e.provider_symbol, e.month, e.sessions, e.reason])

    feature_columns = [
        c
        for c in matrix.columns
        if c
        not in ("provider_symbol", "month", "sessions", "monthly_return",
                "trailing_3m_mean_return", "next_month_return")
    ]
    months = sorted(set(matrix["month"]))
    print(f"  {len(matrix)} ticker-month observations from {len(frames)} instruments")
    print(f"  {len(feature_columns)} feature columns, {months[0]} .. {months[-1]}")
    print(f"  {len(exclusions)} ticker-months excluded")
    labelable = matrix.dropna(
        subset=["monthly_return", "trailing_3m_mean_return", "next_month_return"]
    )
    print(f"  {len(labelable)} fully labelable observations")
    print(f"  wrote {matrix_path.name}, {dictionary_path.name}, {exclusions_path.name}")


# Stage identifier -> (task group that builds it, callable or None).
# Callables are wired in as each group completes.
STAGES: dict[str, tuple[str, Callable[[RunConfig], None] | None]] = {
    "prices": ("group 2 (market data)", None),
    "universe": ("group 2 (market data)", None),
    "rates": ("group 2 (market data)", _stage_rates),
    "quality": ("group 2 (market data)", _stage_quality),
    "screen": ("group 2 (market data)", lambda cfg: _stage_screen(cfg)),
    "features": ("group 3 (feature engineering)", lambda cfg: _stage_features(cfg)),
    "screening": ("group 4 (stock screening)", None),
    "optimize": ("group 5 (portfolio optimisation)", None),
    "backtest": ("group 6 (backtest evaluation)", None),
    "benchmarks": ("group 7 (comparator suite)", None),
    "ablation": ("group 8 (screening ablation)", None),
    "stats": ("group 10 (statistical analysis)", None),
    "report": ("group 11 (reporting)", None),
}

STAGE_ORDER = tuple(STAGES)


class StageNotImplementedError(RuntimeError):
    """Raised when a requested stage has not been built yet."""


def _run_stage(name: str, config: RunConfig) -> None:
    group, fn = STAGES[name]
    if fn is None:
        raise StageNotImplementedError(
            f"stage {name!r} is not implemented yet; it is built by {group}"
        )
    fn(config)


def cmd_config(config: RunConfig, _args: argparse.Namespace) -> int:
    """Print the resolved configuration summary and its fingerprint."""
    months = config.windows.eval_months()
    print(f"config            {config.name}  (fingerprint {config.fingerprint()})")
    print(f"download window   {config.windows.download_start} .. {config.windows.download_end}")
    print(f"evaluation        {months[0]} .. {months[-1]}  ({len(months)} months)")
    print(f"labeling          {', '.join(config.screening.labeling_strategies)}")
    print(f"screeners         {', '.join(config.screening.screeners)}")
    print(f"weight caps       {', '.join(str(c) for c in config.optimization.weight_caps)}")
    print(f"factorial cells   {config.factorial_cell_count()}")
    print(f"portfolio-months  {config.factorial_cell_count() * len(months)}")
    print(f"NSGA-III runs     {config.nsga_run_count()}")
    print(f"eval budget/run   {config.optimization.evaluation_budget}")
    if config.ablation.enabled:
        extended = config.ablation.extended_budget(config.optimization.evaluation_budget)
        sizes = ", ".join(
            "no-preselection" if s is None else str(s)
            for s in config.ablation.selection_sizes
        )
        print(f"ablation sizes    {sizes}")
        print(f"ablation budget   {extended} evaluations, {config.ablation.seeds} seeds")
    print(f"cost scenarios    {config.evaluation.cost_scenarios_bps} bps "
          f"(primary {config.evaluation.primary_cost_scenario_bps})")
    return 0


def cmd_manifest(config: RunConfig, args: argparse.Namespace) -> int:
    """Write a run manifest for the resolved configuration."""
    substitutions = [s for s in (pbc4cip_substitution(),) if s is not None]
    # Record only that a token was configured, never its value: the manifest is meant to be
    # shared alongside the results.
    try:
        rates_stage.get_token()
        token_present = True
    except rates_stage.SIEError:
        token_present = False
    manifest = build_manifest(
        config,
        data_vintage=args.data_vintage,
        substitutions=substitutions,
        notes={
            "banxico_token_configured": token_present,
            "banxico_series": {
                label: spec.series_id for label, spec in rates_stage.SIE_SERIES.items()
            },
        },
    )
    path = write_manifest(manifest, config.paths.results)
    print(f"wrote {path}")
    for substitution in manifest.substitutions:
        flag = "AFFECTS RESULTS" if substitution.affects_results else "result-neutral"
        print(f"  substitution [{flag}] {substitution.component}: {substitution.reason}")
    return 0


def cmd_stage(config: RunConfig, args: argparse.Namespace) -> int:
    config.paths.ensure()
    _run_stage(args.name, config)
    return 0


def cmd_run_all(config: RunConfig, _args: argparse.Namespace) -> int:
    config.paths.ensure()
    for name in STAGE_ORDER:
        print(f"== {name}")
        _run_stage(name, config)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bmvport", description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="configuration file (default: configs/default.yaml)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="print the resolved configuration and exit")

    p_manifest = sub.add_parser("manifest", help="write a run manifest")
    p_manifest.add_argument(
        "--data-vintage",
        default=None,
        help="ISO date the price cache was retrieved",
    )

    p_stage = sub.add_parser("stage", help="run a single stage")
    p_stage.add_argument("name", choices=STAGE_ORDER)

    sub.add_parser("run-all", help="run every stage in dependency order")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    handlers: dict[str, Callable[[RunConfig, argparse.Namespace], int]] = {
        "config": cmd_config,
        "manifest": cmd_manifest,
        "stage": cmd_stage,
        "run-all": cmd_run_all,
    }
    try:
        return handlers[args.command](config, args)
    except StageNotImplementedError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
