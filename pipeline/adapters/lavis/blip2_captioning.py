"""Make local LAVIS annotations for the Unsafe2Safe BLIP-2 run.

This adapter routes each annotation to a safe or original image and removes
private images without a safe copy before the external LAVIS package reads it.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    """Read one LAVIS unified annotation list."""
    with path.open() as handle:
        annotations = json.load(handle)
    if not isinstance(annotations, list):
        raise ValueError(f"{path} must contain a JSON list")
    return annotations


def _load_files(path: Path | None) -> set[str]:
    """Read source-relative image names from a manifest's ``file`` column."""
    if path is None:
        return set()
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or "file" not in rows.fieldnames:
            raise ValueError(f"{path} must contain a 'file' column")
        return {row["file"].strip() for row in rows if row.get("file", "").strip()}


def _image_path(root: Path, name: str) -> str:
    path = Path(name).expanduser()
    return str((path if path.is_absolute() else root / path).resolve())


def prepare_annotations(
    annotation_path: Path,
    output_path: Path,
    *,
    original_root: Path,
    safe_root: Path | None = None,
    safe_manifest: Path | None = None,
    private_manifest: Path | None = None,
    check_files: bool = False,
) -> tuple[int, int]:
    """Write one routed LAVIS annotation file and return (written, dropped).

    A safe manifest lists the same relative image names as the annotations.
    A private manifest marks names that must be dropped when they are absent
    from the safe manifest; without it, unmatched images use ``original_root``.
    """
    annotations = _load_annotations(annotation_path)
    safe_files = _load_files(safe_manifest)
    private_files = _load_files(private_manifest)
    if safe_files and safe_root is None:
        raise ValueError("--safe-root is required with --safe-manifest")

    routed: list[dict[str, Any]] = []
    dropped = 0
    for annotation in annotations:
        name = annotation["image"]
        if name in safe_files:
            # Absolute paths let vanilla LAVIS read both image roots in one run.
            assert safe_root is not None
            image = _image_path(safe_root, name)
        elif name in private_files:
            dropped += 1
            continue
        else:
            image = _image_path(original_root, name)

        if check_files and not Path(image).is_file():
            raise FileNotFoundError(image)
        item = dict(annotation)
        item["image"] = image
        routed.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(routed, handle, indent=2)
        handle.write("\n")
    return len(routed), dropped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-annotations", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path)
    parser.add_argument("--safe-manifest", type=Path)
    parser.add_argument("--private-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--check-files", action="store_true", help="fail on a missing image"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for split in ("train", "val", "test"):
        count, dropped = prepare_annotations(
            getattr(args, f"{split}_annotations"),
            args.output_dir / f"{split}.json",
            original_root=args.original_root,
            safe_root=args.safe_root,
            safe_manifest=args.safe_manifest,
            private_manifest=args.private_manifest,
            check_files=args.check_files,
        )
        print(f"{split}: wrote {count} annotations; dropped {dropped} private items")


if __name__ == "__main__":
    main()
