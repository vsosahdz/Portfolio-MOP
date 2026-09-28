"""Assemble and compile the LaTeX manuscript from the generated sections.

Runs after regenerate_reports.py, which produces the sections, tables and figures this
reads. Nothing here writes prose or numbers; it converts and typesets what is already
generated, so a change in the results reaches the PDF without anyone editing the PDF's
source.

Writes paper/main.tex and, when a LaTeX engine is available, paper/main.pdf.
"""
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from bmvport.reporting.front_matter import build_front_matter, read_authors  # noqa: E402
from bmvport.reporting.latex import compile_pdf, overfull_boxes, write_latex  # noqa: E402

TITLE = ("Self-referential performance assessment hides search degradation: "
         "a diagnostic, and a portfolio-selection case study")

sections = {
    path.name[len("section_"):-len(".md")]: path.read_text(encoding="utf-8")
    for path in sorted(Path("paper").glob("section_*.md"))
}
if not sections:
    sys.exit("no sections found; run scripts/regenerate_reports.py first")

front_matter = build_front_matter(read_authors())
tex = write_latex(sections, "paper", title=TITLE, authors=front_matter["authors"])
print(f"wrote {tex} from {len(sections)} sections")

engine = next((e for e in ("tectonic", "latexmk", "pdflatex") if shutil.which(e)), None)
if engine is None:
    sys.exit("no LaTeX engine found; main.tex is written and can be compiled elsewhere")
print(f"compiling with {engine}")
pdf = compile_pdf(tex, engine=engine)
print(f"wrote {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")

# A box past the margin is invisible on screen and plain in print, and only the log sees it.
boxes = overfull_boxes(pdf.with_suffix(".log"))
serious = [b for b in boxes if b[0] >= 1.0]
print(f"overfull boxes: {len(boxes)} total, {len(serious)} at or above 1pt")
for points, context in serious[:5]:
    print(f"  {points:6.2f}pt {context[:70]}")
