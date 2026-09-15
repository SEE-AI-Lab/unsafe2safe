"""CSV-backed datasets for downstream ImageMAE classification."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def _is_truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "unsafe", "private"}


def _label_sort_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def build_class_mapping(rows: Iterable[Mapping[str, str]], label_column: str) -> dict[str, int]:
    labels = {str(row[label_column]) for row in rows}
    return {label: index for index, label in enumerate(sorted(labels, key=_label_sort_key))}


def build_transform(is_train: bool, input_size: int = 224) -> transforms.Compose:
    """Use ImageNet normalization expected by the official MAE checkpoints."""
    if is_train:
        image_transforms = [
            transforms.RandomResizedCrop(
                input_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        resize_size = int(input_size / 0.875)
        image_transforms = [
            transforms.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
        ]
    image_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(image_transforms)


class ImageMAEClassificationDataset(Dataset):
    """Read original or generated images selected by a privacy flag."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        *,
        image_root: str | Path,
        generated_root: str | Path | None,
        class_to_idx: Mapping[str, int],
        transform: Callable | None,
        path_column: str = "file",
        label_column: str = "class",
        privacy_column: str = "PRIVACY_FLAG",
    ) -> None:
        self.rows = list(rows)
        self.image_root = Path(image_root)
        self.generated_root = Path(generated_root) if generated_root else None
        self.class_to_idx = dict(class_to_idx)
        self.transform = transform
        self.path_column = path_column
        self.label_column = label_column
        self.privacy_column = privacy_column
        required = {path_column, label_column}
        missing = sorted(column for column in required if any(column not in row for row in self.rows))
        if missing:
            raise KeyError(f"Manifest is missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.rows)

    def _image_path(self, row: Mapping[str, str]) -> Path:
        relative_path = Path(row[self.path_column])
        if _is_truthy(row.get(self.privacy_column, "false")):
            if self.generated_root is None:
                raise ValueError("generated_root is required for rows marked private/unsafe")
            return self.generated_root / relative_path
        return self.image_root / relative_path

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        row = self.rows[index]
        with Image.open(self._image_path(row)) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, self.class_to_idx[str(row[self.label_column])]


def split_manifest(
    rows: Sequence[Mapping[str, str]], *, split_column: str, split_name: str
) -> list[Mapping[str, str]]:
    if any(split_column not in row for row in rows):
        raise KeyError(f"Manifest is missing split column {split_column!r}")
    selected = [row for row in rows if str(row[split_column]).strip() == split_name]
    if not selected:
        raise ValueError(f"No rows found for {split_column}={split_name!r}")
    return selected
