

def test_completeness_distinguishes_an_explained_shortfall_from_a_silent_one():
    """A short cell with a skip is correct; a short cell without one is a lost run.

    The executed grid contained both. Three cells were short because a 0.20 cap is
    infeasible on five names, which is detected before seeding and recorded once per
    allocator. One cell was short because the front collapsed to a single point and no knee
    existed -- and that one recorded nothing at all, which is the case this separates.
    """
    import pandas as pd

    from bmvport.backtest.results import completeness_report
    from bmvport.config import load_config

    config = load_config("configs/default.yaml")
    frame = pd.read_parquet("results/results_combined.parquet")
    report = completeness_report(frame, config)

    assert report["passes"], (
        "cells short of their expected rows with no skip recorded: "
        f"{report['short_without_any_skip'][:3]}"
    )
    for record in report["short_with_recorded_skip"]:
        assert record["skip_reasons"], "an explained shortfall must name its reason"
