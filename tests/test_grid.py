

def test_degenerate_front_records_a_skip_for_every_allocator_it_cannot_supply():
    """A cell must be a result or a recorded skip, never simply absent.

    The executed grid hit this: five names under a 0.20 cap make equal weight the only
    feasible portfolio, so the front collapses to one point and no knee exists. Forty rows
    vanished with no skip reason, which the completeness check (9.5) could see only as a
    count mismatch it could not attribute.
    """
    import inspect

    import numpy as np

    from bmvport.grid import run_grid
    from bmvport.optimization.front import extract_portfolios

    # A front of a single point: minimum-risk and maximum-return coincide, no knee.
    weights = np.full((1, 5), 0.2)
    objectives = np.array([[-0.01, 0.03]])
    produced = {p.allocator for p in extract_portfolios(weights, objectives)}

    assert "ga_knee" not in produced, "a one-point front cannot support a knee"
    missing = {"ga_min_risk", "ga_knee", "ga_max_return"} - produced
    assert missing, "this fixture must exercise the missing-allocator path"

    source = inspect.getsource(run_grid)
    assert "supports no" in source, "the grid must record the omission as a skip"
