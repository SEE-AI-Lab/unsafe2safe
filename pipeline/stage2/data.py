from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from pipeline.data_prep.filter_dataset import filter_by_clip_similarity


def _image_path(root: Path, relative_path: str | Path) -> Path:
    """Resolve a manifest path without allowing it to escape its image root."""
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"manifest path must stay relative to the image root: {relative}")
    root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest path must stay relative to the image root: {relative}") from exc
    return resolved


def _image_tensor(image: Image.Image) -> torch.Tensor:
    return rearrange(2 * torch.tensor(np.array(image)).float() / 255 - 1, "h w c -> c h w")


class EditDataset(Dataset):
    """Load aligned unsafe/public image pairs and their two text conditions."""

    def __init__(
        self,
        path: str,
        target_path: str,
        csv_path: str,
        clip_score_path=None,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        file_column: str = "file",
        public_caption_column: str = "caption_public",
        edit_caption_column: str = "caption_edit",
        score_file_column: str = "filename",
        score_public_column: str = "clip_orig",
        score_edit_column: str = "clip_edit",
        clip_threshold: float = 0.7,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of: train, val, test")
        if any(value < 0 for value in splits) or not math.isclose(sum(splits), 1.0):
            raise ValueError("splits must sum to 1")
        if splits[0] + splits[1] == 0:
            raise ValueError("train and validation split fractions cannot both be zero")

        self.file_column = file_column
        self.public_caption_column = public_caption_column
        self.edit_caption_column = edit_caption_column

        df = pd.read_csv(csv_path)
        required = {file_column, public_caption_column, edit_caption_column}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")
        if clip_score_path is not None:
            df_clip_score = pd.read_csv(clip_score_path)
            score_columns = {score_file_column, score_public_column, score_edit_column}
            missing = sorted(score_columns - set(df_clip_score.columns))
            if missing:
                raise ValueError(f"CLIP score CSV is missing columns: {missing}")
            df = df_clip_score.merge(df, left_on=score_file_column, right_on=file_column, how="inner")
            df = filter_by_clip_similarity(
                df,
                original_column=score_public_column,
                edited_column=score_edit_column,
                threshold=clip_threshold,
            )

        # Use the COCO filename split used by the released manifests.
        train_df = df[
            df[file_column].astype(str).str.contains("train2014", na=False)
        ].reset_index(drop=True)
        test_df = df[
            df[file_column].astype(str).str.contains("val2014", na=False)
        ].reset_index(drop=True)

        # Shuffle once so train/validation membership is reproducible.
        train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        train_fraction = splits[0] / (splits[0] + splits[1])
        train_cutoff = int(train_fraction * len(train_df))

        if split == "train":
            selected_df = train_df[:train_cutoff]
        elif split == "val":
            selected_df = train_df[train_cutoff:]
        elif split == "test":
            selected_df = test_df

        self.rows = selected_df.to_dict(orient="records")
        self.root_dir = Path(path)
        self.target_dir = Path(target_path)
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        entry = self.rows[i]
        relative_path = entry[self.file_column]
        image_path = _image_path(self.root_dir, relative_path)
        caption_public = str(entry[self.public_caption_column])
        caption_edit = str(entry[self.edit_caption_column])
        target_image_path = _image_path(self.target_dir, relative_path)

        resize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        with Image.open(image_path) as image:
            image_0 = _image_tensor(image.convert("RGB").resize((resize_res, resize_res), Image.Resampling.LANCZOS))
        with Image.open(target_image_path) as image:
            image_1 = _image_tensor(image.convert("RGB").resize((resize_res, resize_res), Image.Resampling.LANCZOS))

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
