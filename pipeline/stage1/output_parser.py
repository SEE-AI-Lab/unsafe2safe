"""Parse the structured sections requested by the Stage 1 prompts."""

from __future__ import annotations

import re


def parse_structured_output(text: str) -> dict[str, object]:
    """Return caption sections and the privacy flag from one model response."""
    sections = {}
    current = None
    for line in text.splitlines():
        heading = line.strip()
        if heading.upper().startswith("SECTION:"):
            current = heading.split(":", 1)[1].strip().upper()
            if current == "PRIVACY_FLAG":
                current = None
            else:
                sections.setdefault(current, [])
            continue
        if current:
            sections[current].append(line)

    sections = {name: "\n".join(lines).strip() for name, lines in sections.items()}
    # Keep the names consumed by the later Stage 1 profiles.
    for source, target in (("FULL_VISUAL_DESCRIPTION", "PRIVATE_CAPTION"), ("PRIVACY_PRESERVING_DESCRIPTION", "PUBLIC_CAPTION")):
        if source in sections:
            sections[target] = sections[source]
    match = re.search(r"PRIVACY_FLAG\s*:\s*(TRUE|FALSE)", text, re.IGNORECASE)
    if match:
        sections["PRIVACY_FLAG"] = match.group(1).upper() == "TRUE"
    return sections
