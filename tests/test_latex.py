"""Guards on the LaTeX assembly.

Two of these encode defects that reached a compiled PDF during development: a per-cent sign
that would have commented out the rest of its line, and citations written as prose that
never reached the bibliography. Both produce a document that compiles and is wrong, which is
the failure mode worth testing for.
"""

from __future__ import annotations

import pytest

from bmvport.reporting.latex import CITATIONS, markdown_to_latex


def test_percent_signs_are_escaped():
    """Every fact this study quotes is a percentage; an unescaped one truncates its line."""
    latex = markdown_to_latex("The control returns 12.0% annualised at 50 bps.\n")

    assert r"12.0\%" in latex
    assert "12.0%" not in latex.replace(r"12.0\%", "")


def test_citations_become_commands_rather_than_staying_prose():
    """A citation left as text vanishes from the bibliography without any error."""
    for literal, _ in CITATIONS:
        latex = markdown_to_latex(f"As shown {literal} this holds.\n")
        assert r"\citep{" in latex, f"{literal} stayed as prose"
        assert literal not in latex


def test_maths_survives_escaping():
    """Mathematics is already LaTeX; escaping it would destroy it."""
    latex = markdown_to_latex("Coverage is $C_j(F)$ for objective $j$.\n")

    assert "$C_j(F)$" in latex
    assert r"\textbackslash" not in latex


def test_display_equation_becomes_an_equation_environment():
    latex = markdown_to_latex("text\n\n$$ x = y. $$\n\nmore\n")

    assert r"\begin{equation}" in latex and r"\end{equation}" in latex


def test_converter_refuses_a_construct_it_was_not_built_for():
    """Guessing at an unsupported construct puts literal markup in a typeset paper."""
    with pytest.raises(ValueError, match="will not guess"):
        markdown_to_latex("| a | b |\n")
    with pytest.raises(ValueError, match="will not guess"):
        markdown_to_latex("```python\nx = 1\n```\n")


def test_only_the_section_heading_takes_the_label():
    """A label on a subsection makes a cross-reference resolve to the wrong place."""
    latex = markdown_to_latex("## Results\n\ntext\n\n### Detail\n\nmore\n", label="results")

    assert latex.count(r"\label{sec:results}") == 1
    assert r"\section{Results}\label{sec:results}" in latex


def test_compiled_pdf_has_no_unresolved_reference_or_citation():
    """The compiled artefact, not the source: a broken ref shows as ?? on the page."""
    pypdf = pytest.importorskip("pypdf")
    from pathlib import Path

    pdf_path = Path("paper/main.pdf")
    if not pdf_path.exists():
        pytest.skip("paper/main.pdf not built; run scripts/build_paper.py")

    text = "\n".join(page.extract_text() for page in pypdf.PdfReader(pdf_path).pages)

    assert "??" not in text, "a cross-reference did not resolve"
    assert "[?]" not in text, "a citation did not resolve"
    assert "References" in text, "the bibliography is missing"


def test_no_content_runs_past_the_text_block():
    """Ink crossing the margin is invisible on screen and plain in print.

    Nothing else in this pipeline can see it. The table source looks correct, the extracted
    text is complete, and only the log records the overflow. An earlier draft ran 97pt past
    the margin while appearing, on screen, to fit.

    The one box left is elsarticle's own output routine, not content, so the threshold
    admits it by magnitude rather than by exempting the check.
    """
    from pathlib import Path

    from bmvport.reporting.latex import overfull_boxes

    log = Path("paper/main.log")
    if not log.exists():
        pytest.skip("paper/main.log not present; run scripts/build_paper.py")

    serious = [box for box in overfull_boxes(log) if box[0] >= 5.0]
    assert not serious, f"content runs past the text block: {serious[:5]}"


def test_document_tree_places_every_generated_section():
    """A section written and never placed is absent from the paper without any error."""
    from pathlib import Path

    from bmvport.reporting.latex import DOCUMENT

    placed = {name for _, names in DOCUMENT for name in names}
    front_matter = {"abstract", "highlights", "keywords"}
    written = {
        path.name[len("section_"):-len(".md")]
        for path in Path("paper").glob("section_*.md")
    }
    if not written:
        pytest.skip("no generated sections present")

    assert not written - placed - front_matter, (
        f"generated but never placed in the document: {written - placed - front_matter}"
    )
