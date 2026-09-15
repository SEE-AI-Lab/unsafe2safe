from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset

def _image_tensor(image: Image.Image) -> torch.Tensor:
    return 2 * torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255 - 1


class EditDataset(Dataset):
    """Load aligned unsafe/public image pairs and their two text conditions."""

    def __init__(self, path: str, target_path: str, csv_path: str, split: str = "train", train_fraction: float = 0.95, crop_res: int = 256, flip_prob: float = 0.0, file_column: str = "file", public_caption_column: str = "caption_public", edit_caption_column: str = "caption_edit"):
        self.file_column = file_column
        self.public_caption_column = public_caption_column
        self.edit_caption_column = edit_caption_column

        df = pd.read_csv(csv_path)

        # Use the COCO filename split used by the released manifests.
        train_df = df[
            df[file_column].astype(str).str.contains("train2014", na=False)
        ].reset_index(drop=True)
        # Shuffle once so train/validation membership is reproducible.
        train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        train_cutoff = int(train_fraction * len(train_df))

        selected_df = {"train": train_df[:train_cutoff], "val": train_df[train_cutoff:]}[split]

        self.rows = selected_df.to_dict(orient="records")
        self.root_dir = Path(path)
        self.target_dir = Path(target_path)
        self.crop_res = crop_res
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        entry = self.rows[i]
        relative_path = entry[self.file_column]
        image_path = self.root_dir / relative_path
        caption_public = str(entry[self.public_caption_column])
        caption_edit = str(entry[self.edit_caption_column])
        target_image_path = self.target_dir / relative_path

        # Both images use the fixed 256px training resolution from the paper.
        with Image.open(image_path) as image:
            image_0 = _image_tensor(image.convert("RGB").resize((self.crop_res, self.crop_res), Image.Resampling.LANCZOS))
        with Image.open(target_image_path) as image:
            image_1 = _image_tensor(image.convert("RGB").resize((self.crop_res, self.crop_res), Image.Resampling.LANCZOS))

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        # Crop and flip the concatenated pair so both images keep identical
        # geometry after augmentation.
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(
            image_private=image_0,
            image_public=image_1,
            caption_public=caption_public,
            caption_edit=caption_edit,
        )
