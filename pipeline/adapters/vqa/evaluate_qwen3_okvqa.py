"""Generate OK-VQA predictions from a Qwen3-VL adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from pipeline.adapters.vqa.common import (
    DEFAULT_MODEL_ID,
    SYSTEM_MESSAGE,
    coco_image_path,
    load_json,
)


def format_example(question_id: int, image_path: Path, question: str) -> dict:
    return {
        "id": question_id,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question},
                ],
            },
        ],
    }


def build_examples(questions: list[dict], image_root: Path) -> list[dict]:
    return [
        format_example(
            question["question_id"],
            image_root / coco_image_path(question["image_id"], "val"),
            question["question"],
        )
        for question in questions
    ]


def generate_batch(model, processor, examples: list[dict], max_new_tokens: int) -> list[str]:
    messages = [example["messages"] for example in examples]
    texts = [
        processor.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
        for item in messages
    ]
    image_inputs = [process_vision_info(item)[0][0] for item in messages]
    model_inputs = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    generated = model.generate(**model_inputs, max_new_tokens=max_new_tokens)
    prompt_length = model_inputs.input_ids.shape[1]
    return [
        output.strip()
        for output in processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_json(args.questions)["questions"]
    examples = build_examples(questions, args.image_root)
    model_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if args.cache_dir:
        model_kwargs["cache_dir"] = str(args.cache_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        padding_side="left",
        cache_dir=args.cache_dir,
    )
    model.load_adapter(args.adapter)

    predictions = {}
    for index in tqdm(range(0, len(examples), args.batch_size), desc="Generating answers"):
        batch = examples[index : index + args.batch_size]
        outputs = generate_batch(model, processor, batch, args.max_new_tokens)
        predictions.update({item["id"]: output for item, output in zip(batch, outputs)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(predictions, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
