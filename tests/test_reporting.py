

def test_appendix_always_emits_the_four_required_limitations():
    """A limitation that can be switched off is one that will be, in the run that needs it."""
    import json

    import pandas as pd

    from bmvport.config import load_config
    from bmvport.reporting.appendix import build_appendix

    config = load_config("configs/default.yaml")
    results = pd.read_parquet("results/results_combined.parquet")
    manifest = json.loads(open("results/manifest.json", encoding="utf-8").read())
    appendix = build_appendix(results, config, manifest)

    names = {item["name"] for item in appendix["limitations"]}
    assert names == {
        "Survivorship exposure", "Evaluation block count",
        "Post-hoc transaction costs", "Flat spread assumption",
    }
    for item in appendix["limitations"]:
        assert len(item["statement"]) > 120, f"{item['name']} is stated too thinly to judge"


def test_appendix_takes_substitutions_from_the_manifest_only():
    """One source of truth: a restated substitution can disagree with what actually ran."""
    import inspect

    from bmvport.reporting import appendix

    source = inspect.getsource(appendix.build_appendix)
    assert 'manifest.get("substitutions"' in source
    assert "pbc4cip" not in source.lower(), "no substitution may be hard-coded here"


def test_appendix_has_no_missing_required_item():
    """Every item the specification names must resolve to a value, not to None.

    A `None` in the appendix is worse than an omission: it reads as "recorded" while
    carrying nothing, and the first draft of this generator produced four of them by
    guessing configuration field names that did not exist.
    """
    import json

    import pandas as pd

    from bmvport.config import load_config
    from bmvport.reporting.appendix import build_appendix

    config = load_config("configs/default.yaml")
    results = pd.read_parquet("results/results_combined.parquet")
    manifest = json.loads(open("results/manifest.json", encoding="utf-8").read())
    appendix = build_appendix(results, config, manifest)

    empty = []
    for section in ("data", "universe", "features", "screening", "optimisation",
                    "evaluation", "environment"):
        for key, value in appendix[section].items():
            if value is None:
                empty.append(f"{section}.{key}")
            elif isinstance(value, dict):
                empty += [f"{section}.{key}.{k}" for k, v in value.items() if v is None]
    assert not empty, f"appendix items resolved to None: {empty}"


def test_no_manuscript_section_contains_a_hand_typed_number():
    """Task 12.8 as a test rather than a proofread.

    Every quantity a section quotes must come from the executed run, so that a rerun which
    changes a number changes the sentence built from it. Only citation years and the names
    of instruments and methods that happen to contain digits are exempt, and that exemption
    is an explicit list rather than a relaxed pattern.
    """
    from bmvport.reporting.manuscript import template_digit_violations

    violations = template_digit_violations()
    assert not violations, f"numbers typed into section templates: {violations}"


def test_every_section_placeholder_resolves():
    """An unresolved placeholder would ship as literal braces in the manuscript."""
    import pytest

    from bmvport.reporting.manuscript import (SECTIONS, render_sections,
                                               required_facts)

    with pytest.raises(KeyError):
        render_sections({})

    needed = required_facts()
    rendered = render_sections({k: "X" for k in needed})
    # Braces alone are not evidence of a stray placeholder: the method section carries a
    # displayed equation whose braces belong to the mathematics.
    for name, text in rendered.items():
        leftover = [key for key in needed if "{" + key + "}" in text]
        assert not leftover, f"{name} kept placeholders: {leftover}"


def test_detectable_effect_never_annualises_a_ratio_metric():
    """A Sharpe ratio is not a monthly rate and must not be multiplied by twelve.

    The cap hypothesis is tested on the Sharpe ratio. Pooling its detectable effect with the
    return-metric tests and annualising the lot produced a reported bound of 542 percent --
    the same class of error the RETURN_METRICS guard exists to prevent, reintroduced one
    layer up by grouping tests that do not share a unit.
    """
    import json

    import pandas as pd

    from bmvport.config import load_config
    from bmvport.reporting.manuscript import collect_facts
    from bmvport.stats.inference import critical_difference

    config = load_config("configs/default.yaml")
    results = pd.read_parquet("results/results_combined.parquet")
    mixed = json.loads(open("results/mixed_model.json", encoding="utf-8").read())
    artefacts = {
        "ablation": pd.read_csv("results/ablation_search_space.csv"),
        "reachability": pd.read_csv("results/front_reachability_sweep.csv"),
        "tests": json.loads(open("results/tests.json", encoding="utf-8").read()),
        "friedman": pd.read_csv("results/friedman.csv"),
        "shrinkage": json.loads(
            open("results/shrinkage_sensitivity.json", encoding="utf-8").read()),
        "mixed_model": mixed, "se_ratios": [5.0, 9.0],
        "n_cell_months": mixed["n_observations"],
        "critical_difference": critical_difference(6, 18), "headline_cells": 6,
    }
    facts = collect_facts(results, config, artefacts)

    high = float(facts["mde_high"].rstrip("%"))
    assert high < 100.0, f"detectable effect of {high}% means a ratio metric was annualised"


def test_manifest_fingerprint_matches_the_shipped_configuration():
    """Results produced under a different configuration are not results this package makes.

    Seeds derive from the configuration fingerprint, so a drifted configuration reseeds the
    genetic algorithm and the stored table stops being reproducible from the shipped config.
    It drifted once during development without anything noticing, because the numbers stayed
    close enough that no analysis looked wrong (D35).
    """
    import json

    from bmvport.config import load_config

    config = load_config("configs/default.yaml")
    manifest = json.loads(open("results/manifest.json", encoding="utf-8").read())

    assert manifest["config_fingerprint"] == config.fingerprint(), (
        "the stored results predate the shipped configuration; rerun scripts/run_grid.py"
    )


def test_rendered_manuscript_respects_the_venue_length_limits():
    """Checked on rendered text: a substituted fact changes the length after the prose.

    Expert Systems with Applications decides scope and substance within a five-day window
    before review, so an over-length abstract or highlight is a desk rejection rather than a
    revision request.
    """
    import json

    import pandas as pd

    from bmvport.config import load_config
    from bmvport.reporting.manuscript import (collect_facts, render_sections,
                                              venue_limit_violations)
    from bmvport.stats.inference import critical_difference

    config = load_config("configs/default.yaml")
    results = pd.read_parquet("results/results_combined.parquet")
    mixed = json.loads(open("results/mixed_model.json", encoding="utf-8").read())
    artefacts = {
        "ablation": pd.read_csv("results/ablation_search_space.csv"),
        "reachability": pd.read_csv("results/front_reachability_sweep.csv"),
        "tests": json.loads(open("results/tests.json", encoding="utf-8").read()),
        "friedman": pd.read_csv("results/friedman.csv"),
        "shrinkage": json.loads(
            open("results/shrinkage_sensitivity.json", encoding="utf-8").read()),
        "mixed_model": mixed, "se_ratios": [5.5, 9.2],
        "n_cell_months": mixed["n_observations"],
        "critical_difference": critical_difference(6, 18), "headline_cells": 6,
    }
    rendered = render_sections(collect_facts(results, config, artefacts))

    assert not venue_limit_violations(rendered)


def test_front_matter_author_order_comes_from_the_packaging_metadata():
    """One source of truth: the paper's order cannot drift from the artefact's."""
    from bmvport.reporting.front_matter import read_authors

    authors = read_authors()

    assert authors[0].startswith("Víctor"), "first author is also the corresponding author"
    assert len(authors) == 3


def test_front_matter_marks_what_nobody_has_supplied_rather_than_guessing():
    """A guessed CRediT statement assigns real credit for real work to real people.

    It is also the field a reviewer is most likely to check against the acknowledgements, so
    a plausible default is worse than a marker that announces itself.
    """
    from bmvport.reporting.front_matter import (build_front_matter, read_authors,
                                                unresolved_items)

    front_matter = build_front_matter(read_authors())
    pending = unresolved_items(front_matter)

    assert any("CRediT" in item for item in pending)
    assert any("ORCID" in item or "orcid" in item for item in pending)
    assert "data availability" in pending, (
        "an unnamed repository is a desk-rejection trigger and must be listed as pending"
    )


def test_front_matter_rejects_a_role_outside_the_credit_vocabulary():
    """CRediT is a fixed vocabulary; an invented role is not a contribution statement."""
    import pytest

    from bmvport.reporting.front_matter import build_front_matter, read_authors

    authors = read_authors()
    with pytest.raises(ValueError, match="CRediT vocabulary"):
        build_front_matter(authors, credit={authors[0]: ["Thesis supervision"]})
    with pytest.raises(ValueError, match="not an author"):
        build_front_matter(authors, credit={"Someone Else": ["Methodology"]})


def test_every_section_renders_with_literal_braces_intact():
    """LaTeX braces and str.format share a delimiter, and format wins.

    The method section carries a displayed equation. Written naively its braces are read as
    format fields and rendering raises before any check on content can run, so every brace
    that belongs to the mathematics has to be doubled in the template.
    """
    import re

    from bmvport.reporting.manuscript import (SECTIONS, render_sections,
                                               required_facts)

    rendered = render_sections({key: "X" for key in required_facts()})

    assert "\\frac{" in rendered["method"], "the equation lost its braces"
    assert "{X}" not in rendered["method"], "a fact was wrapped in stray braces"


def test_typeset_table_is_narrow_enough_and_reads_as_a_table():
    """The first typeset draft ran off the page and lost its rightmost columns in silence.

    LaTeX does not warn when a tabular overflows into the margin, so the check is on the
    source: a constant column is width spent on nothing, a machine name renders with visible
    gaps where its underscores were escaped, and a raw fraction contradicts a prose sentence
    that quotes the same quantity as a percentage.
    """
    import pandas as pd

    from bmvport.config import load_config
    from bmvport.reporting.tables import performance_table, to_latex

    config = load_config("configs/default.yaml")
    results = pd.read_parquet("results/results_combined.parquet")
    table = performance_table(results, config.evaluation.primary_cost_scenario_bps)
    latex = to_latex(table)

    assert "cost_bps" not in latex and "50.0000" not in latex, "a constant column survived"
    assert r"\_" not in latex, "a machine name reached the typeset heading"
    assert r"12.0\%" in latex, "returns must be typeset as percentages, as the prose quotes them"
    header = [line for line in latex.splitlines() if "Screening arm" in line][0]
    assert header.count("&") <= 7, f"too many columns to fit the text block: {header}"
    assert latex.splitlines()[1].startswith(r"\begin{tabular}{l"), (
        "the label column must be left-aligned"
    )


def test_no_table_prints_a_bare_nan():
    """A literal NaN in a typeset table reads as a mistake even where it is a true absence."""
    import json

    import pandas as pd

    from bmvport.config import load_config
    from bmvport.reporting.tables import comparator_table, to_latex

    config = load_config("configs/default.yaml")
    results = pd.read_parquet("results/results_combined.parquet")
    tests = json.loads(open("results/tests.json", encoding="utf-8").read())

    class _Test:
        def __init__(self, record):
            self.__dict__.update(record)

    latex = to_latex(comparator_table(
        results, [_Test(r) for r in tests], config.evaluation.primary_cost_scenario_bps))

    assert "NaN" not in latex and "nan" not in latex
