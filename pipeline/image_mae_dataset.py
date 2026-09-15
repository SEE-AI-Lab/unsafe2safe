"""Project dataset definition for downstream ImageMAE classification.

This module is independent of the diffusion dataset and does not import
ImageMAE. The upstream MAE checkout can consume the returned
``(image_tensor, class_id)`` samples through a small external training wrapper.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV containing at least ``file``, ``class``, and ``split``."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    return rows


def split_rows(rows: list[dict[str, str]], split: str, split_column: str = "split"):
    """Select one reproducible train/validation/test split from the manifest."""
    selected = [row for row in rows if row.get(split_column, "").strip() == split]
    if not selected:
        raise ValueError(f"No rows found for {split_column}={split!r}")
    return selected


def build_transform(is_train: bool, image_size: int = 224):
    """Build the ImageNet preprocessing expected by standard MAE checkpoints."""
    if is_train:
        steps = [
            transforms.RandomResizedCrop(
                image_size, scale=(0.08, 1.0), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.RandomHorizontalFlip(),
        ]
    else:
        steps = [
            transforms.Resize(int(image_size / 0.875), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(steps)


def _is_private(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "unsafe", "private"}


class ImageMAEDataset(Dataset):
    """Return original or edited images and their downstream class labels.

    Rows marked private/unsafe are resolved under ``edited_root``. All other
    rows use ``image_root``. Both roots must share the relative paths in the
    manifest's ``file`` column.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        image_root: str | Path,
        edited_root: str | Path | None = None,
        transform=None,
        *,
        class_column: str = "class",
        file_column: str = "file",
        privacy_column: str = "PRIVACY_FLAG",
    ):
        self.rows = rows
        self.image_root = Path(image_root)
        self.edited_root = Path(edited_root) if edited_root else None
        self.transform = transform
        self.file_column = file_column
        self.class_column = class_column
        self.privacy_column = privacy_column

        labels = sorted({row[class_column] for row in rows})
        self.class_to_idx = {label: index for index, label in enumerate(labels)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        relative_path = Path(row[self.file_column])
        if _is_private(row.get(self.privacy_column, "false")):
            if self.edited_root is None:
                raise ValueError("edited_root is required for private/unsafe rows")
            image_path = self.edited_root / relative_path
        else:
            image_path = self.image_root / relative_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, self.class_to_idx[row[self.class_column]]
