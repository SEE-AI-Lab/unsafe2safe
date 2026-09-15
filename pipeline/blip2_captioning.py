"""Prepare LAVIS-compatible annotations for Unsafe2Safe BLIP-2 training.

The historical LAVIS copy selected an anonymized image when one existed and
otherwise used the original image, while dropping private images that had no
anonymized counterpart.  This module keeps that project-owned mapping outside
the separately installed LAVIS dependency.  It writes ordinary LAVIS unified
annotation lists with absolute image paths; the generated files belong in a
local experiment directory, not in this repository.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    """Load a LAVIS unified annotation list."""
    with path.open() as handle:
        data = json.load(handle)

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        raise ValueError(
            f"{path} must contain a JSON list of LAVIS annotation objects"
        )
    if any("image" not in item for item in data):
        raise ValueError(f"{path} contains an annotation without an 'image' field")
    return data


def _load_manifest(
    path: Path | None,
    source_column: str = "file",
    target_column: str | None = None,
) -> dict[str, str]:
    """Return source-relative image names and their replacement names."""
    if path is None:
        return {}

    with path.open(newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or source_column not in rows.fieldnames:
            raise ValueError(f"{path} must contain a '{source_column}' column")
        if target_column and target_column not in rows.fieldnames:
            raise ValueError(f"{path} must contain a '{target_column}' column")

        mapping: dict[str, str] = {}
        for row in rows:
            source = (row.get(source_column) or "").strip()
            if not source:
                continue
            target = (row.get(target_column) if target_column else source) or ""
            target = target.strip()
            if not target:
                raise ValueError(f"{path} has an empty replacement for '{source}'")
            if source in mapping and mapping[source] != target:
                raise ValueError(f"{path} maps '{source}' to multiple files")
            mapping[source] = target
        return mapping


def _resolve(root: Path, image_name: str) -> str:
    image_path = Path(image_name).expanduser()
    if not image_path.is_absolute():
        image_path = root / image_path
    return str(image_path.resolve())


def _iter_prepared(
    annotations: Iterable[dict[str, Any]],
    *,
    original_root: Path,
    safe_root: Path | None,
    safe_files: dict[str, str],
    private_files: set[str],
    drop_missing_private: bool,
    check_files: bool,
) -> tuple[list[dict[str, Any]], int]:
    prepared: list[dict[str, Any]] = []
    dropped = 0

    for annotation in annotations:
        source_name = str(annotation["image"])
        if source_name in safe_files:
            if safe_root is None:
                raise ValueError("--safe-root is required when --safe-manifest is used")
            image_path = _resolve(safe_root, safe_files[source_name])
        elif source_name in private_files:
            if not drop_missing_private:
                raise FileNotFoundError(
                    f"no anonymized counterpart listed for private image '{source_name}'"
                )
            dropped += 1
            continue
        else:
            image_path = _resolve(original_root, source_name)

        if check_files and not Path(image_path).is_file():
            raise FileNotFoundError(image_path)

        item = dict(annotation)
        item["image"] = image_path
        prepared.append(item)

    return prepared, dropped


def prepare_annotations(
    annotation_path: Path,
    output_path: Path,
    *,
    original_root: Path,
    safe_root: Path | None = None,
    safe_manifest: Path | None = None,
    private_manifest: Path | None = None,
    manifest_source_column: str = "file",
    manifest_target_column: str | None = None,
    drop_missing_private: bool = True,
    check_files: bool = False,
) -> tuple[int, int]:
    """Create one portable LAVIS annotation list.

    ``safe_manifest`` lists anonymized images keyed by the source annotation
    name.  ``private_manifest`` lists images that must not fall back to the
    original when their anonymized counterpart is missing.
    """
    annotations = _load_annotations(annotation_path)
    safe_files = _load_manifest(
        safe_manifest,
        source_column=manifest_source_column,
        target_column=manifest_target_column,
    )
    private_files = set(
        _load_manifest(
            private_manifest,
            source_column=manifest_source_column,
        )
    )
    prepared, dropped = _iter_prepared(
        annotations,
        original_root=original_root,
        safe_root=safe_root,
        safe_files=safe_files,
        private_files=private_files,
        drop_missing_private=drop_missing_private,
        check_files=check_files,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(prepared, handle, indent=2)
        handle.write("\n")
    return len(prepared), dropped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-annotations", type=Path, required=True)
    parser.add_argument("--val-annotations", type=Path, required=True)
    parser.add_argument("--test-annotations", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path)
    parser.add_argument("--safe-manifest", type=Path)
    parser.add_argument("--private-manifest", type=Path)
    parser.add_argument("--manifest-source-column", default="file")
    parser.add_argument("--manifest-target-column")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--keep-missing-private",
        action="store_true",
        help="fall back to the original for private images without a safe file",
    )
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="fail if any resolved image path does not exist",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    for split, annotation_path in (
        ("train", args.train_annotations),
        ("val", args.val_annotations),
        ("test", args.test_annotations),
    ):
        count, dropped = prepare_annotations(
            annotation_path,
            args.output_dir / f"{split}.json",
            original_root=args.original_root,
            safe_root=args.safe_root,
            safe_manifest=args.safe_manifest,
            private_manifest=args.private_manifest,
            manifest_source_column=args.manifest_source_column,
            manifest_target_column=args.manifest_target_column,
            drop_missing_private=not args.keep_missing_private,
            check_files=args.check_files,
        )
        print(f"{split}: wrote {count} annotations; dropped {dropped} private items")


if __name__ == "__main__":
    main()
