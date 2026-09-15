"""Shared data helpers for the Qwen3-VL OK-VQA scripts."""

import json
from pathlib import Path


SYSTEM_MESSAGE = (
    "You are a Vision Language Model answering questions about images. "
    "Use the image to identify relevant entities and visual cues, and use "
    "general world knowledge to answer the question. Provide a concise, "
    "factual answer without speculation."
)


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def coco_image_path(image_id: int, split: str) -> str:
    return f"{split}2014/COCO_{split}2014_{int(image_id):012d}.jpg"
