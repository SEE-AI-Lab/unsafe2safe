"""Collect Stage 1 JSON outputs into a CSV table."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def collect_captions(captions_dir, *, output_path=None, metadata_path=None, filename_suffix="_caption.json", image_suffix=".jpg", parse_structured=False, output_column="caption"):
    """Read generated captions, optionally merge metadata, and write a CSV."""
    root = Path(captions_dir)
    rows = []
    for caption_path in sorted(root.rglob(f"*{filename_suffix}")):
        with caption_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        caption = payload.get("caption", "")

        relative = caption_path.relative_to(root)
        image_name = relative.name[: -len(filename_suffix)] + image_suffix
        row = {"file": str(relative.with_name(image_name)), output_column: caption}
        if parse_structured:
            from pipeline.stage1.output_parser import parse_structured_output

            row.update(parse_structured_output(caption))
        rows.append(row)

    if metadata_path:
        captions = {row["file"]: row for row in rows}
        with Path(metadata_path).open(encoding="utf-8", newline="") as handle:
            rows = []
            for row in csv.DictReader(handle):
                if row["file"] in captions:
                    row.update(captions[row["file"]])
                    rows.append(row)

    if output_path:
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
