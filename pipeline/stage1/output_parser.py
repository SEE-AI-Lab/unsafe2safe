"""Parse structured outputs produced by the Stage 1 prompt templates."""

from __future__ import annotations

import re
from typing import Optional


_SECTION_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?SECTION:\s*"
    r"([A-Z][A-Z0-9_]*)(?:\*\*)?\s*:?[ \t]*$",
    re.IGNORECASE,
)
_KNOWN_SECTION_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?"
    r"([A-Z][A-Z0-9_]*)(?:\*\*)?\s*:?[ \t]*$",
    re.IGNORECASE,
)
_FLAG_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?PRIVACY_FLAG\s*:\s*"
    r"(TRUE|FALSE)\b",
    re.IGNORECASE,
)

_SECTION_ALIASES = {
    "FULL_VISUAL_DESCRIPTION": "PRIVATE_CAPTION",
    "PRIVACY_PRESERVING_DESCRIPTION": "PUBLIC_CAPTION",
}


def _section_name(line: str) -> Optional[str]:
    """Return a recognized section name without treating body text as one."""
    normalized = line.replace("**", "").strip()
    match = _SECTION_RE.match(normalized)
    if match:
        return match.group(1).upper()

    match = _KNOWN_SECTION_RE.match(normalized)
    if not match:
        return None
    name = match.group(1).upper()
    known = {"PRIVACY_FLAG", "PRIVACY_REVIEW", *(_SECTION_ALIASES.keys())}
    return name if name in known else None


def extract_privacy_flag(text: str) -> Optional[bool]:
    """Return the first explicit privacy flag, or ``None`` if absent."""
    awaiting_value = False
    for line in text.replace("<think>", "").replace("</think>", "").splitlines():
        normalized = line.replace("**", "").strip()
        match = _FLAG_RE.match(normalized)
        if match:
            return match.group(1).upper() == "TRUE"
        if _section_name(normalized) == "PRIVACY_FLAG":
            awaiting_value = True
        elif awaiting_value and normalized.upper() in {"TRUE", "FALSE"}:
            return normalized.upper() == "TRUE"
    return None


def split_sections(text: str) -> dict[str, str]:
    """Return section headings and their text from a structured model response."""
    sections: dict[str, list[str]] = {}
    current: Optional[str] = None

    for line in text.replace("<think>", "").replace("</think>", "").splitlines():
        name = _section_name(line)
        if name and name != "PRIVACY_FLAG":
            current = name
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)

    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def parse_structured_output(text: str) -> dict[str, object]:
    """Parse a Stage 1 response into a flag and named caption sections."""
    sections = split_sections(text)
    for source, target in _SECTION_ALIASES.items():
        if target not in sections and source in sections:
            sections[target] = sections[source]

    flag = extract_privacy_flag(text)
    if flag is not None:
        sections["PRIVACY_FLAG"] = flag
    return sections
