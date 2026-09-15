"""Collect Stage 1 JSON outputs into CSV or JSONL records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect_captions(captions_dir, *, filename_suffix="_caption.json", image_suffix=".jpg", parse_structured=False, output_column="caption"):
    """Read generated captions and return rows keyed by relative image path."""
    root = Path(captions_dir)
    rows = []
    for caption_path in sorted(root.rglob(f"*{filename_suffix}")):
        with caption_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        caption = payload.get("caption", "")
        if not isinstance(caption, str):
            raise ValueError(f"Caption must be text: {caption_path}")

        relative = caption_path.relative_to(root)
        image_name = relative.name[: -len(filename_suffix)] + image_suffix
        row = {"file": str(relative.with_name(image_name)), output_column: caption}
        if parse_structured:
            from pipeline.stage1.output_parser import parse_structured_output

            row.update(parse_structured_output(caption))
        rows.append(row)
    return rows


def _fieldnames(rows):
    preferred = ["file", "caption", "PRIVACY_FLAG", "PRIVACY_REVIEW", "PRIVATE_CAPTION", "PUBLIC_CAPTION"]
    available = {key for row in rows for key in row}
    return [key for key in preferred if key in available] + sorted(available.difference(preferred))


def write_caption_table(rows, output_path):
    """Write caption rows to CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames(rows))
        writer.writeheader()
        writer.writerows(rows)


def merge_with_metadata(rows, metadata_path):
    """Return metadata rows that have a matching generated caption."""
    captions = {row["file"]: row for row in rows}
    with Path(metadata_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "file" not in reader.fieldnames:
            raise ValueError("Metadata CSV must contain a 'file' column")
        merged = []
        for row in reader:
            if row["file"] in captions:
                row.update(captions[row["file"]])
                merged.append(row)
    return merged


def main():
    parser = argparse.ArgumentParser(description="Collect generated Stage 1 captions")
    parser.add_argument("--captions-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, help="Optional CSV with a file column")
    parser.add_argument("--parse-structured", action="store_true", help="Add parsed privacy and caption fields")
    parser.add_argument("--output-column", default="caption", help="Column name for the generated text")
    args = parser.parse_args()

    rows = collect_captions(args.captions_dir, parse_structured=args.parse_structured, output_column=args.output_column)
    if args.metadata:
        rows = merge_with_metadata(rows, args.metadata)
    write_caption_table(rows, args.output)


if __name__ == "__main__":
    main()
