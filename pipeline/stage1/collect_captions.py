"""Collect Stage 1 JSON outputs into a CSV table."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from pipeline.stage1.output_parser import parse_structured_output


def collect_captions(captions_dir, output_path, *, parse_structured=False, output_column="caption"):
    """Read generated captions into a CSV; do not merge metadata."""
    root = Path(captions_dir)
    rows = []
    for caption_path in sorted(root.rglob("*_caption.json")):
        with caption_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        caption = payload["caption"]

        relative = caption_path.relative_to(root)
        image_name = relative.name.removesuffix("_caption.json") + ".jpg"
        row = {"file": str(relative.with_name(image_name)), output_column: caption}
        if parse_structured:
            row.update(parse_structured_output(caption))
        rows.append(row)

    preferred = ["file", "caption", "PRIVACY_FLAG", "PRIVACY_REVIEW", "PRIVATE_CAPTION", "PUBLIC_CAPTION"]
    available = {key for row in rows for key in row}
    fieldnames = [key for key in preferred if key in available] + sorted(available.difference(preferred))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows
