"""Assemble the LaTeX manuscript from the generated sections, tables and figures.

The sections are authored as Markdown because that is what a co-author reads and comments
on, and converted here rather than maintained twice. The converter is deliberately small and
deliberately strict: it handles the constructs the sections actually use and raises on
anything else, so an unsupported construct fails at build time instead of reaching the PDF as
literal asterisks.

Escaping is the part that matters. Every fact this study quotes is a percentage, and an
unescaped per-cent sign comments out the rest of the line in LaTeX -- silently, producing a
document that compiles and is wrong. Escaping runs before any markup is interpreted, and the
few sequences the converter itself emits are protected from it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

__all__ = ["markdown_to_latex", "build_document", "write_latex", "compile_pdf",
           "overfull_boxes", "SECTION_ORDER", "DOCUMENT", "FIGURES"]

# The document tree, following the structure of the companion UAV study from the same group
# so a reader of both meets the same shape twice. Each entry is a top-level section and the
# generated sections that sit under it as subsections; a section listed alone becomes a
# top-level section in its own right.
#
# The consolidation matters beyond consistency. Contributions belong in the introduction
# rather than standing alone, empirical material reads as one argument rather than four
# sections that each restate the setup, and limitations sit with the results they qualify
# instead of after the conclusions, where a reader has already stopped weighing them.
DOCUMENT: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Introduction", ("introduction", "contributions")),
    ("Background", ("related_work",)),
    ("The proposal", ("method",)),
    ("Experiments, results and discussion",
     ("experimental_setup", "results", "screening_justification", "threats", "limitations")),
    ("Conclusions and future work", ("conclusions", "future_work")),
)

# Emitted after the tree above, from the run manifest rather than from a prose section: the
# parameters that fix the run, in one place a reader can check the code against.
PARAMETERS_SECTION = "Parameters and provenance"

SECTION_ORDER: tuple[str, ...] = tuple(
    name for _, names in DOCUMENT for name in names
)

# Figures in the main text, with their captions and the label the prose can reference.
FIGURES: tuple[tuple[str, str, str], ...] = (
    ("fig_search_cost", "search-cost",
     "Search cost and attainment against problem size, by ablation level, with standard "
     "deviations across seeds. The attainment axis is not zero-based. The rising attainment "
     "is the reading this paper argues is not interpretable on its own."),
    ("fig_pareto_front", "pareto",
     "One Pareto front with the three extracted portfolios marked, at the median evaluation "
     "month under the no-screening control. The three profiles are nearly the same "
     "portfolio, which is the collapse reported in Section~\\ref{sec:experiments} "
     "made concrete."),
    ("fig_performance", "performance",
     "Distribution of monthly net return by screening arm at the primary cost scenario."),
    ("fig_attribution", "attribution",
     "Return attribution by screening arm. The selection component is negative for every "
     "arm and exactly zero for the no-screening control, which validates the decomposition."),
    ("fig_critical_difference", "critical-difference",
     "Critical-difference diagram over the pre-registered headline set. The bar spans one "
     "critical difference; no pair is separated."),
)

TABLES: tuple[tuple[str, str, str], ...] = (
    ("table_performance", "performance",
     "Per-screener performance at the primary cost scenario."),
    ("table_comparators", "comparators",
     "Every arm ranked against the comparator confrontation. No comparison resolves: "
     "the detectable bound exceeds every difference, as the notes record."),
    ("table_attribution", "attribution",
     "Return decomposition into market, selection, allocation and currency tilt."),
    ("table_profiles", "profiles",
     "The three investor profiles, each against its designated metric."),
)

# Citations are written as plain text in the Markdown so a co-author reads a sentence rather
# than a command, and are turned into citation commands here. The mapping is explicit: an
# unmapped citation stays as literal text, which a test catches, rather than silently
# vanishing from the bibliography.
CITATIONS: tuple[tuple[str, str], ...] = (
    ("(Markowitz, 1952)", r"\citep{markowitz1952}"),
    ("(Zitzler and Thiele, 1999)", r"\citep{zitzler1999}"),
    ("(Fonseca and Fleming, 1996)", r"\citep{fonseca1996}"),
    ("(Deb and Jain, 2014)", r"\citep{deb2014}"),
    ("(Dem\u0161ar, 2006)", r"\citep{demsar2006}"),
)

_ESCAPES = (
    ("\\", r"\textbackslash{}"),
    ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
    ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
    ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
)


def _escape(text: str) -> str:
    for character, replacement in _ESCAPES:
        text = text.replace(character, replacement)
    return text


def _inline(text: str) -> str:
    """Convert inline markup on a line of prose, escaping everything else first.

    Mathematics is lifted out before escaping and put back afterwards: it is already LaTeX
    and escaping it would destroy it.
    """
    maths: list[str] = []

    def stash(match: re.Match) -> str:
        maths.append(match.group(0))
        return f"\x00{len(maths) - 1}\x00"

    text = re.sub(r"\$\$.+?\$\$|\$[^$]+\$", stash, text)
    # Citation commands are stashed like mathematics: they are LaTeX already and escaping
    # would turn the backslash into \textbackslash.
    for literal, command in CITATIONS:
        if literal in text:
            maths.append(command)
            text = text.replace(literal, f"\x00{len(maths) - 1}\x00")
    text = _escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\\emph{\1}", text)
    text = re.sub(r"`([^`]+)`", r"\\texttt{\1}", text)
    # Section cross-references survive escaping as a deliberate exception.
    text = text.replace(r"Section\textasciitilde{}\textbackslash{}ref\{", "Section~\\ref{")
    text = re.sub(r"Section~\\ref\{([^}]*)\\\}", r"Section~\\ref{\1}", text)

    def restore(match: re.Match) -> str:
        return maths[int(match.group(1))]

    return re.sub(r"\x00(\d+)\x00", restore, text)


def _first_heading(markdown: str) -> str | None:
    """Title of a generated section, for deciding whether it repeats its parent."""
    for line in markdown.splitlines():
        if line.strip().startswith("#"):
            return line.strip().lstrip("#").strip()
    return None


def markdown_to_latex(
    markdown: str, *, top_level: str = "section", label: str | None = None,
    drop_first_heading: bool = False,
) -> str:
    """Convert one generated section. Raises on any construct it was not built for."""
    depth = {"section": 0, "subsection": 1}[top_level]
    levels = ("section", "subsection", "subsubsection", "paragraph")
    out: list[str] = []
    paragraph: list[str] = []
    in_list = False

    def flush() -> None:
        if paragraph:
            out.append(_inline(" ".join(paragraph)))
            out.append("")
            paragraph.clear()

    for raw in markdown.splitlines():
        line = raw.rstrip()
        stripped = line.strip()

        if not stripped:
            flush()
            if in_list:
                out.append(r"\end{itemize}")
                out.append("")
                in_list = False
            continue

        if stripped.startswith("#"):
            flush()
            if in_list:
                out.append(r"\end{itemize}")
                in_list = False
            hashes = len(stripped) - len(stripped.lstrip("#"))
            title = stripped.lstrip("#").strip()
            if drop_first_heading and hashes == 2:
                drop_first_heading = False
                continue
            command = levels[min(depth + hashes - 2, len(levels) - 1)]
            heading = f"\\{command}{{{_inline(title)}}}"
            # Only the top-level heading of a section carries its label, so a cross-reference
            # resolves to the section rather than to whichever subsection came last.
            if label and command == levels[depth]:
                heading += f"\\label{{sec:{label}}}"
                label = None
            out.append(heading)
            out.append("")
            continue

        if stripped.startswith("- "):
            flush()
            if not in_list:
                out.append(r"\begin{itemize}")
                in_list = True
            out.append(f"  \\item {_inline(stripped[2:])}")
            continue

        if stripped.startswith("$$") and stripped.endswith("$$"):
            flush()
            body = stripped[2:-2].strip()
            out.append(r"\begin{equation}")
            out.append(f"  {body}")
            out.append(r"\end{equation}")
            out.append("")
            continue

        if stripped.startswith(("|", "```", ">")):
            raise ValueError(
                f"the converter does not handle this construct and will not guess: {stripped[:60]!r}"
            )

        paragraph.append(stripped)

    flush()
    if in_list:
        out.append(r"\end{itemize}")
    return "\n".join(out).strip() + "\n"


def build_document(
    sections: Mapping[str, str], *, title: str, authors: Sequence[Mapping[str, str]],
    keywords: str, figure_dir: str = "figures", table_dir: str = "tables",
) -> str:
    """Assemble the full elsarticle document."""
    author_block = []
    for index, author in enumerate(authors):
        mark = "[label1]" if index == 0 else "[label1]"
        corresponding = "\\corref{cor1}" if index == 0 else ""
        author_block.append(f"\\author{mark}{{{_escape(author['name'])}{corresponding}}}")
    author_block.append(r"\cortext[cor1]{Corresponding author.}")
    author_block.append(
        f"\\affiliation[label1]{{organization={{{_escape(authors[0]['affiliation'])}}}}}"
    )

    abstract = markdown_to_latex(sections["abstract"]).replace("\\section{Abstract}\n", "")
    # elsarticle's highlights environment is itself a list, so the bullets go in as bare
    # \item lines; wrapping them in an itemize raises "perhaps a missing \item".
    highlights = "\n".join(
        f"  \\item {_inline(line.strip()[2:])}"
        for line in sections["highlights"].splitlines()
        if line.strip().startswith("- ")
    )

    body = []
    for heading, names in DOCUMENT:
        present = [name for name in names if name in sections]
        if not present:
            continue
        slug = heading.lower().split(",")[0].split(" and ")[0].replace(" ", "-")
        body.append("% " + "=" * 74)
        body.append(f"\\section{{{_escape(heading)}}}\\label{{sec:{slug}}}")
        body.append("")
        for name in present:
            body.append(f"% --- {name} " + "-" * (68 - len(name)))
            # Each generated section's own heading demotes one level: it was written as a
            # section and now sits as a subsection beneath the heading above.
            #
            # Unless it would only repeat its parent. A lone child, or one whose title
            # matches the section it sits under, produces "1 Introduction / 1.1
            # Introduction" -- a level of numbering that carries no information and reads as
            # a mistake. Its own subsections still promote correctly when it is dropped.
            own_title = _first_heading(sections[name])
            drop_heading = len(present) == 1 or (
                own_title and own_title.lower() == heading.lower())
            converted = markdown_to_latex(
                sections[name], top_level="subsection",
                label=None if drop_heading else name.replace("_", "-"),
                drop_first_heading=drop_heading)
            if name == "results":
                converted += "\n" + _float_block(figure_dir, table_dir)
            body.append(converted)
            body.append("")

    body.append("% " + "=" * 74)
    body.append(f"\\section{{{_escape(PARAMETERS_SECTION)}}}\\label{{sec:parameters}}")
    body.append("")
    body.append(f"\\input{{{table_dir}/parameters}}")
    body.append("")

    keyword_line = "; ".join(
        _escape(k.strip()) for k in keywords.replace("## Keywords", "").split(";") if k.strip()
    )

    # Substituted by explicit marker, not str.format: the template is LaTeX and almost
    # every line contains braces that format would read as fields.
    document = TEMPLATE
    for marker, value in (
        ("<<TITLE>>", _escape(title)),
        ("<<AUTHORS>>", "\n".join(author_block)),
        ("<<ABSTRACT>>", abstract.strip()),
        ("<<HIGHLIGHTS>>", highlights.strip()),
        ("<<KEYWORDS>>", keyword_line),
        ("<<BODY>>", "\n".join(body).strip()),
    ):
        assert marker in document, f"template lost its {marker} marker"
        document = document.replace(marker, value)
    leftover = re.findall(r"<<[A-Z_]+>>", document)
    assert not leftover, f"unsubstituted markers: {leftover}"
    return document


def _float_block(figure_dir: str, table_dir: str) -> str:
    """Figures and tables, placed after the results section rather than inline.

    Elsevier typesets floats itself, so their position in the source is not the position on
    the page; grouping them keeps the prose readable in the source.
    """
    blocks = []
    for stem, label, caption in FIGURES:
        blocks.append(
            "\\begin{figure}[htbp]\n  \\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{{figure_dir}/{stem}.png}}\n"
            f"  \\caption{{{caption}}}\n  \\label{{fig:{label}}}\n\\end{{figure}}"
        )
    for stem, label, caption in TABLES:
        blocks.append(
            "\\begin{table}[htbp]\n  \\centering\n  \\small\n"
            f"  \\caption{{{caption}}}\n  \\label{{tab:{label}}}\n"
            f"  \\input{{{table_dir}/{stem}.tex}}\n\\end{{table}}"
        )
    return "\n\n".join(blocks) + "\n"


TEMPLATE = r"""\documentclass[preprint,11pt,authoryear]{elsarticle}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{tabularx}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}

% A tall float may share a page with text rather than claiming a page of its own.
\renewcommand{\topfraction}{0.92}
\renewcommand{\bottomfraction}{0.7}
\renewcommand{\textfraction}{0.08}
\renewcommand{\floatpagefraction}{0.85}

% Let TeX relax a line rather than push it into the margin. Without this a handful of
% paragraphs overflow by a few points, which is invisible on screen and visible in print.
\emergencystretch=3em

\journal{Expert Systems with Applications}

\begin{document}

\begin{frontmatter}

\title{<<TITLE>>}

<<AUTHORS>>

\begin{abstract}
<<ABSTRACT>>
\end{abstract}

\begin{highlights}
<<HIGHLIGHTS>>
\end{highlights}

\begin{keyword}
<<KEYWORDS>>
\end{keyword}

\end{frontmatter}

<<BODY>>

\bibliographystyle{elsarticle-harv}
\bibliography{references}

\end{document}
"""


def write_latex(
    sections: Mapping[str, str], directory: str | Path, *, title: str,
    authors: Sequence[Mapping[str, str]],
) -> Path:
    """Write main.tex beside the tables and figures it includes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    document = build_document(
        sections, title=title, authors=authors, keywords=sections.get("keywords", ""))
    path = directory / "main.tex"
    path.write_text(document, encoding="utf-8")
    return path


def overfull_boxes(log_path: str | Path) -> list[tuple[float, str]]:
    """Overfull boxes from a LaTeX log, worst first.

    A box that runs past the text block is invisible on screen at normal zoom and plain in
    print. Nothing else in this pipeline can see it: the table source looks correct, the PDF
    text extracts correctly, and only the log records that the ink crossed the margin.
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    found = [
        (float(points), context.strip())
        for points, context in re.findall(
            r"Overfull \\hbox \(([0-9.]+)pt too wide\)([^\n]*)", text)
    ]
    return sorted(found, reverse=True)


def compile_pdf(tex_path: str | Path, *, engine: str = "tectonic") -> Path:
    """Compile to PDF, surfacing the log rather than a bare exit code.

    A LaTeX failure is almost always one line in a long log, and a caller that only sees
    "returned 1" has to go find it.
    """
    tex_path = Path(tex_path)
    result = subprocess.run(
        [engine, "--keep-logs", "--print", str(tex_path.name)],
        cwd=tex_path.parent, capture_output=True, text=True,
    )
    if result.returncode != 0:
        errors = [
            line for line in (result.stderr + result.stdout).splitlines()
            if line.startswith("!") or "Error" in line or "error:" in line
        ]
        raise RuntimeError(
            "LaTeX compilation failed:\n" + "\n".join(errors[:20] or
                                                      result.stderr.splitlines()[-20:])
        )
    return tex_path.with_suffix(".pdf")
