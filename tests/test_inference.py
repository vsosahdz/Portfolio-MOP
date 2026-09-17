

def test_mixed_model_cannot_report_p_values_without_the_dependence_check():
    """A fitted model must carry the ratio that says how far its p-values can be trusted.

    The executed grid put that ratio near ten. A future run where the model is refitted and
    the check quietly dropped would present nominal significance for effects the design
    cannot resolve, which is the exact misreading D32 exists to prevent.
    """
    import inspect

    from bmvport.stats import inference

    source = inspect.getsource(inference.mixed_model)
    assert "dependence_check" in source
    assert "inference_basis" in source
    returned = [line for line in source.splitlines() if "p_value" in line]
    assert returned, "the model reports p-values, so the check is mandatory"


def test_month_block_standard_error_uses_months_not_cell_months():
    """The unit of analysis is the evaluation month; cells within a month are not replicates."""
    import numpy as np
    import pandas as pd
    import pytest

    from bmvport.config import load_config
    from bmvport.stats.inference import month_block_standard_errors

    config = load_config("configs/default.yaml")
    rng = np.random.default_rng(0)
    months = [f"2025-{m:02d}" for m in range(1, 13)]
    # Twenty cells per month that move together: the dependence the mixed model misses.
    rows = []
    for month in months:
        shock = rng.normal(0, 0.05)
        for cell in range(20):
            rows.append({"month": month, "screener": "all_stocks",
                         "y": shock + rng.normal(0, 0.001)})
            rows.append({"month": month, "screener": "svm",
                         "y": shock - 0.004 + rng.normal(0, 0.001)})
    result = month_block_standard_errors(pd.DataFrame(rows), config, metric="y", n_boot=2000)

    assert result["available"]
    (contrast,) = result["contrasts"]
    assert contrast["n_blocks"] == len(months), "blocks are months, not cell-months"
    assert contrast["effect_monthly"] == pytest.approx(-0.004, abs=5e-4)


def test_serialised_test_result_keeps_the_verdict_fields():
    """asdict drops properties, and the dropped ones are what the test concluded."""
    import dataclasses

    from bmvport.stats.inference import TestResult

    result = TestResult(
        family="confirmatory", name="H1", effect=-0.0075, ci_low=-0.017, ci_high=0.0004,
        p_value=0.167, p_adjusted=0.502, minimum_detectable=0.0143, n_blocks=18,
        metric="return_net", cost_bps=50.0, note="",
    )
    record = result.to_record()

    assert set(dataclasses.asdict(result)) < set(record)
    for field in ("rejected", "below_resolution", "verdict"):
        assert field in record, f"{field} was dropped by serialisation"
    assert record["verdict"] == result.verdict
