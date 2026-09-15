"""Parse structured outputs produced by the Stage 1 prompt templates."""

from __future__ import annotations

import re

_SECTION_ALIASES = {
    "FULL_VISUAL_DESCRIPTION": "PRIVATE_CAPTION",
    "PRIVACY_PRESERVING_DESCRIPTION": "PUBLIC_CAPTION",
}


def _section_name(line: str):
    match = re.match(r"SECTION:\s*([A-Z][A-Z0-9_]*)\s*$", line.strip(), re.IGNORECASE)
    return match.group(1).upper() if match else None


def extract_privacy_flag(text: str):
    """Return the explicit privacy flag, or ``None`` when it is absent."""
    match = re.search(r"PRIVACY_FLAG\s*:\s*(TRUE|FALSE)", text, re.IGNORECASE)
    return match.group(1).upper() == "TRUE" if match else None


def split_sections(text: str) -> dict[str, str]:
    """Return section headings and their text from a structured model response."""
    sections: dict[str, list[str]] = {}
    current = None

    for line in text.splitlines():
        name = _section_name(line)
        if name:
            current = None if name == "PRIVACY_FLAG" else name
            if current:
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
