"""Dataset definition for ImageMAE downstream evaluation.

This module does not import ImageMAE or the diffusion pipeline. It only maps
the project manifest to the ``(image, class_id)`` samples expected by a
standard external MAE classifier.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a manifest containing ``file``, ``class``, and ``split`` columns."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def build_transform(is_train: bool, image_size: int = 224):
    """Build the ImageNet preprocessing used by standard MAE checkpoints."""
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
    """Load original images or edited counterparts for downstream labels."""

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
        class_to_idx: Mapping[str, int] | None = None,
        is_train: bool = False,
    ):
        self.rows = rows
        self.image_root = Path(image_root)
        self.edited_root = Path(edited_root) if edited_root else None
        self.transform = transform
        self.class_column = class_column
        self.file_column = file_column
        self.privacy_column = privacy_column
        labels = {row[class_column] for row in rows}
        if class_to_idx is None:
            labels = sorted(labels)
            self.class_to_idx = {label: index for index, label in enumerate(labels)}
        else:
            self.class_to_idx = dict(class_to_idx)
        self.transform = transform or build_transform(is_train=is_train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        relative_path = Path(row[self.file_column])
        if _is_private(row.get(self.privacy_column, "false")):
            image_path = self.edited_root / relative_path
        else:
            image_path = self.image_root / relative_path

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)
        return image, self.class_to_idx[row[self.class_column]]
