from __future__ import annotations

import json
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
            df = df_clip_score.merge(
                df, left_on="filename", right_on=file_column, how="inner"
            )
            df = df[(df["clip_edit"] / df["clip_orig"]) > 0.7]

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

        self.seeds = selected_df.to_dict(orient="records")
        self.root_dir = Path(path)
        self.target_dir = Path(target_path)
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        entry = self.seeds[i]
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


class EditDatasetEval(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError("split must be one of: train, val, test")
        if any(value < 0 for value in splits) or not math.isclose(sum(splits), 1.0):
            raise ValueError("splits must sum to 1")
        self.path = path
        self.res = res

        with open(Path(self.path, "seeds.json")) as f:
            self.seeds = json.load(f)

        split_0, split_1 = {
            "train": (0.0, splits[0]),
            "val": (splits[0], splits[0] + splits[1]),
            "test": (splits[0] + splits[1], 1.0),
        }[split]

        idx_0 = math.floor(split_0 * len(self.seeds))
        idx_1 = math.floor(split_1 * len(self.seeds))
        self.seeds = self.seeds[idx_0:idx_1]

    def __len__(self) -> int:
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        name, seeds = self.seeds[i]
        prompt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        with open(prompt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)
            edit = prompt["edit"]
            input_prompt = prompt["input"]
            output_prompt = prompt["output"]

        with Image.open(prompt_dir.joinpath(f"{seed}_0.jpg")) as image:
            image_0 = _image_tensor(image.convert("RGB").resize((self.res, self.res), Image.Resampling.LANCZOS))

        return dict(image_0=image_0, input_prompt=input_prompt, edit=edit, output_prompt=output_prompt)
