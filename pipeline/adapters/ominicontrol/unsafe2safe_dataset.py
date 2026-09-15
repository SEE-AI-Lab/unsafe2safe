from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPTokenizer

from pipeline.filter_dataset import filter_by_clip_similarity


class Unsafe2SafeDataset(Dataset):
    """Return Unsafe2Safe pairs in the format expected by OminiControl.

    OminiControl's standard subject task receives a target image, an image
    condition, and a text description. For Unsafe2Safe, the safe image is the
    target, the unsafe image is the condition, and the edit instruction is the
    text description. A zero position delta disables spatial displacement.

    The pair CSV must contain ``file`` and ``caption``. ``file`` is shared by
    the unsafe image under ``image_root`` and the safe image under
    ``target_root``.
    """

    def __init__(
        self,
        csv_path: str | Path,
        image_root: str | Path,
        target_root: str | Path,
        *,
        split: str = "train",
        train_fraction: float = 0.75,
        clip_score_path: str | Path | None = None,
        clip_threshold: float = 0.7,
        image_column: str = "file",
        caption_column: str = "caption",
        score_filename_column: str = "filename",
        score_edit_column: str = "clip_edit",
        score_original_column: str = "clip_orig",
        image_size: tuple[int, int] = (512, 512),
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        tokenizer_name: str = "openai/clip-vit-base-patch32",
        max_caption_tokens: int = 73,
    ) -> None:
        frame = pd.read_csv(csv_path)
        if clip_score_path is not None:
            scores = pd.read_csv(clip_score_path)
            frame = scores.merge(
                frame,
                left_on=score_filename_column,
                right_on=image_column,
                how="inner",
            )
            frame = filter_by_clip_similarity(
                frame,
                original_column=score_original_column,
                edited_column=score_edit_column,
                threshold=clip_threshold,
            )

        # The paper's split is a deterministic 75/25 split of train2014;
        # val2014 is reserved for the test split.
        train_frame = frame[
            frame[image_column].astype(str).str.contains("train2014", na=False)
        ].reset_index(drop=True)
        test_frame = frame[
            frame[image_column].astype(str).str.contains("val2014", na=False)
        ].reset_index(drop=True)

        train_frame = train_frame.sample(frac=1.0, random_state=42).reset_index(drop=True)
        val_cutoff = int((1 - train_fraction) * len(train_frame))
        selected = {"train": train_frame.iloc[val_cutoff:], "val": train_frame.iloc[:val_cutoff], "test": test_frame}[split]

        self.rows = selected.to_dict(orient="records")
        self.image_root = Path(image_root)
        self.target_root = Path(target_root)
        self.image_column = image_column
        self.caption_column = caption_column
        self.image_size = tuple(image_size)
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.to_tensor = T.ToTensor()
        self.max_caption_tokens = max_caption_tokens
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.rows)

    def _caption(self, value: Any) -> str:
        caption = str(value)
        token_ids = self.tokenizer(caption, max_length=self.max_caption_tokens, truncation=True)["input_ids"]
        caption = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return "" if random.random() < self.drop_text_prob else caption

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        relative_path = Path(str(row[self.image_column]))
        with Image.open(self.image_root / relative_path) as image:
            condition = image.convert("RGB")
        with Image.open(self.target_root / relative_path) as image:
            target = image.convert("RGB")

        condition = condition.resize(self.image_size, Image.Resampling.LANCZOS)
        target = target.resize(self.image_size, Image.Resampling.LANCZOS)
        if random.random() < self.drop_image_prob:
            condition = Image.new("RGB", self.image_size, (0, 0, 0))

        # These are the names consumed by OminiControl's subject trainer:
        # safe target, unsafe image condition, edit instruction, and no offset.
        return {
            "image": self.to_tensor(target),
            "condition_0": self.to_tensor(condition),
            "condition_type_0": "subject",
            "position_delta_0": np.array([0, 0], dtype=np.int32),
            "description": self._caption(row[self.caption_column]),
        }
