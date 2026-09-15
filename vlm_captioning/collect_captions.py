"""Collect Stage 1 JSON outputs into a simple CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect_captions(captions_dir, *, filename_suffix="_caption.json", image_suffix=".jpg"):
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
        rows.append({"file": str(relative.with_name(image_name)), "caption": caption})
    return rows


def write_caption_table(rows, output_path):
    """Write caption rows to CSV."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "caption"])
        writer.writeheader()
        writer.writerows(rows)


def merge_with_metadata(rows, metadata_path, output_path):
    """Keep metadata rows that have a matching generated caption."""
    captions = {row["file"]: row["caption"] for row in rows}
    with Path(metadata_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "file" not in reader.fieldnames:
            raise ValueError("Metadata CSV must contain a 'file' column")
        fieldnames = list(reader.fieldnames)
        if "caption" not in fieldnames:
            fieldnames.append("caption")
        merged = []
        for row in reader:
            if row["file"] in captions:
                row["caption"] = captions[row["file"]]
                merged.append(row)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged)


def main():
    parser = argparse.ArgumentParser(description="Collect generated Stage 1 captions")
    parser.add_argument("--captions-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata", type=Path, help="Optional CSV with a file column")
    args = parser.parse_args()

    rows = collect_captions(args.captions_dir)
    if args.metadata:
        merge_with_metadata(rows, args.metadata, args.output)
    else:
        write_caption_table(rows, args.output)


if __name__ == "__main__":
    main()
