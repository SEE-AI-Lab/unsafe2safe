from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset


class EditDataset(Dataset):
    def __init__(
        self,
        path: str,
        target_path: str,
        csv_path: str,
        clip_score_path,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1

        df = pd.read_csv(csv_path)
        if clip_score_path is not None:
            df_clip_score = pd.read_csv(clip_score_path)
            df = df_clip_score.merge(df, left_on="filename", right_on=df["file"], how="left")
            df = df[(df.clip_edit/df.clip_orig)>0.7]

        # Filter by filename.
        train_df = df[df["file"].str.contains("train2014", na=False)].reset_index(drop=True)
        test_df = df[df["file"].str.contains("val2014", na=False)].reset_index(drop=True)

        # Deterministic shuffle for train/val split
        train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        val_cutoff = int(0.25 * len(train_df))

        if split == "train":
            selected_df = train_df[val_cutoff:]
        elif split == "val":
            selected_df = train_df[:val_cutoff]
        elif split == "test":
            selected_df = test_df

        self.seeds = selected_df.to_dict(orient="records")
        self.root_dir = Path(path)  # root folder for images
        self.target_dir = Path(target_path)
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

    def __len__(self):
        return len(self.seeds)

    def __getitem__(self, i: int) -> dict[str, Any]:
        entry = self.seeds[i]
        image_path = self.root_dir / entry["file"]
        caption_c1 = entry["c1"]
        caption_edit = entry["caption"]
        target_image_path = self.target_dir / entry["file"]

        image_0 = Image.open(image_path).convert("RGB")
        image_1 = Image.open(target_image_path).convert("RGB")
        resize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
        image_0 = image_0.resize((resize_res, resize_res), Image.Resampling.LANCZOS)
        image_1 = image_1.resize((resize_res, resize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")
        image_1 = rearrange(2 * torch.tensor(np.array(image_1)).float() / 255 - 1, "h w c -> c h w")

        crop = torchvision.transforms.RandomCrop(self.crop_res)
        flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
        image_0, image_1 = flip(crop(torch.cat((image_0, image_1)))).chunk(2)

        return dict(image_private=image_0, caption_private=caption_c1, caption_edit=caption_edit, image_public=image_1)


class EditDatasetEval(Dataset):
    def __init__(
        self,
        path: str,
        split: str = "train",
        splits: tuple[float, float, float] = (0.9, 0.05, 0.05),
        res: int = 256,
    ):
        assert split in ("train", "val", "test")
        assert sum(splits) == 1
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
        propt_dir = Path(self.path, name)
        seed = seeds[torch.randint(0, len(seeds), ()).item()]
        with open(propt_dir.joinpath("prompt.json")) as fp:
            prompt = json.load(fp)
            edit = prompt["edit"]
            input_prompt = prompt["input"]
            output_prompt = prompt["output"]

        image_0 = Image.open(propt_dir.joinpath(f"{seed}_0.jpg"))

        reize_res = torch.randint(self.res, self.res + 1, ()).item()
        image_0 = image_0.resize((reize_res, reize_res), Image.Resampling.LANCZOS)

        image_0 = rearrange(2 * torch.tensor(np.array(image_0)).float() / 255 - 1, "h w c -> c h w")

        return dict(image_0=image_0, input_prompt=input_prompt, edit=edit, output_prompt=output_prompt)
