"""Submission front matter: authorship, contributions, declarations and data availability.

Generated rather than written for one reason that matters and one that is convenience. The
author list is read from the packaging metadata, so the order cannot drift between the paper
and the artefact a reviewer downloads. And the declarations a venue requires are easy to
half-complete; emitting them together makes an omission visible.

Items that only a person can settle -- affiliations, ORCID identifiers, funding, and who did
what -- are emitted as explicit unresolved markers rather than as plausible defaults. A
guessed CRediT statement assigns credit for real work to real people, and a placeholder that
looks finished is worse than one that announces itself.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any, Mapping

__all__ = ["read_authors", "build_front_matter", "render_front_matter",
           "write_front_matter", "unresolved_items", "MARKER", "REPOSITORY_URL"]

MARKER = "«TO BE SUPPLIED»"

# The roles a CRediT statement draws from. Kept here so a drafted assignment can be checked
# against the vocabulary rather than invented per paper.
CREDIT_ROLES = (
    "Conceptualization", "Data curation", "Formal analysis", "Funding acquisition",
    "Investigation", "Methodology", "Project administration", "Resources", "Software",
    "Supervision", "Validation", "Visualization", "Writing – original draft",
    "Writing – review & editing",
)


def read_authors(pyproject: str | Path = "pyproject.toml") -> list[str]:
    """Author names in submission order, from the packaging metadata."""
    data = tomllib.loads(Path(pyproject).read_text(encoding="utf-8"))
    return [author["name"] for author in data["project"]["authors"]]


def build_front_matter(
    authors: list[str], *, repository_url: str | None = None,
    credit: Mapping[str, list[str]] | None = None,
    affiliations: Mapping[str, str] | None = None,
    orcids: Mapping[str, str] | None = None,
    funding: str | None = None,
) -> dict[str, Any]:
    """Assemble the front matter, marking every item nobody has supplied yet."""
    credit = credit or {}
    for author, roles in credit.items():
        unknown = sorted(set(roles) - set(CREDIT_ROLES))
        if unknown:
            raise ValueError(f"{author}: roles outside the CRediT vocabulary: {unknown}")
        if author not in authors:
            raise ValueError(f"CRediT names {author!r}, who is not an author")

    return {
        "authors": [
            {
                "name": name,
                "position": "first author, corresponding" if index == 0
                else f"author {index + 1}",
                "affiliation": (affiliations or {}).get(name, MARKER),
                "orcid": (orcids or {}).get(name, MARKER),
                "credit": credit.get(name, [MARKER]),
            }
            for index, name in enumerate(authors)
        ],
        "data_availability": (
            f"The code and the cached dataset of record are publicly available at "
            f"{repository_url}. Every table, figure and quoted number in this article "
            f"regenerates from that repository with the network unavailable."
            if repository_url else
            f"{MARKER} — a named public repository is required. Expert Systems with "
            f"Applications treats 'available on request' as grounds for desk rejection."
        ),
        "competing_interests": (
            "The authors declare that they have no known competing financial interests or "
            "personal relationships that could have appeared to influence the work reported "
            "in this article."
        ),
        "funding": funding or MARKER,
        "ethics": (
            "This study analyses publicly available market price series and published "
            "reference rates. It involves no human participants, no animal subjects and no "
            "personal data."
        ),
        "generative_ai": (
            f"{MARKER} — declare any use of generative AI in the writing process, per the "
            f"publisher's policy."
        ),
    }


def unresolved_items(front_matter: Mapping[str, Any]) -> list[str]:
    """Everything still waiting on a person, named so the list can be worked through."""
    pending = []
    for author in front_matter["authors"]:
        for field in ("affiliation", "orcid"):
            if author[field] == MARKER:
                pending.append(f"{author['name']}: {field}")
        if MARKER in author["credit"]:
            pending.append(f"{author['name']}: CRediT roles")
    for field in ("data_availability", "funding", "generative_ai"):
        if MARKER in str(front_matter[field]):
            pending.append(field.replace("_", " "))
    return pending


def render_front_matter(front_matter: Mapping[str, Any]) -> str:
    """Render as Markdown for the submission system and the cover letter."""
    lines = ["# Submission front matter", "", "## Authors", ""]
    for author in front_matter["authors"]:
        lines.append(f"**{author['name']}** — {author['position']}")
        lines.append(f"- Affiliation: {author['affiliation']}")
        lines.append(f"- ORCID: {author['orcid']}")
        lines.append(f"- CRediT: {', '.join(author['credit'])}")
        lines.append("")

    for title, key in (
        ("Data availability", "data_availability"),
        ("Declaration of competing interests", "competing_interests"),
        ("Funding", "funding"),
        ("Ethics", "ethics"),
        ("Generative AI", "generative_ai"),
    ):
        lines += [f"## {title}", "", str(front_matter[key]), ""]

    pending = unresolved_items(front_matter)
    lines += ["## Still to be supplied", ""]
    lines += [f"- {item}" for item in pending] if pending else ["- Nothing outstanding."]
    return "\n".join(lines) + "\n"


REPOSITORY_URL = "https://github.com/vsosahdz/Portfolio-MOP"


def write_front_matter(
    directory: str | Path, **kwargs: Any
) -> Path:
    """Write the front matter beside the manuscript sections."""
    kwargs.setdefault("repository_url", REPOSITORY_URL)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    front_matter = build_front_matter(read_authors(), **kwargs)
    path = directory / "submission_front_matter.md"
    path.write_text(render_front_matter(front_matter), encoding="utf-8")
    return path
