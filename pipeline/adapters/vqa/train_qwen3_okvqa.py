"""Fine-tune Qwen3-VL on OK-VQA with Unsafe2Safe image routing."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

from pipeline.adapters.vqa.common import (
    SYSTEM_MESSAGE,
    coco_image_path,
    load_json,
)


def read_manifest(path: Path | None) -> set[str]:
    if path is None:
        return set()
    frame = pd.read_csv(path)
    if "file" not in frame.columns:
        raise ValueError(f"{path} must contain a 'file' column")
    return set(frame["file"].dropna().astype(str))


def format_example(question_id: int, image_path: Path, question: str, answer: str):
    return {
        "id": question_id,
        "images": [str(image_path)],
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": question},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ],
    }


def build_examples(
    questions: list[dict],
    answers_by_id: dict[int, dict],
    image_root: Path,
    safe_root: Path | None,
    private_files: set[str],
    safe_files: set[str],
) -> list[dict]:
    examples = []
    for question in tqdm(questions, desc="Building VQA examples"):
        question_id = question["question_id"]
        annotation = answers_by_id.get(question_id)
        if annotation is None:
            continue

        filename = coco_image_path(question["image_id"], "train")
        if filename in private_files:
            if filename not in safe_files or safe_root is None:
                continue
            image_path = safe_root / filename
        else:
            image_path = image_root / filename

        counts = Counter(answer["answer"] for answer in annotation["answers"])
        answer = counts.most_common(1)[0][0]
        examples.append(format_example(question_id, image_path, question["question"], answer))
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path)
    parser.add_argument("--safe-manifest", type=Path)
    parser.add_argument("--private-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.validation_fraction < 1:
        raise ValueError("--validation-fraction must be between 0 and 1")

    questions = load_json(args.questions)["questions"]
    annotations = load_json(args.annotations)["annotations"]
    answers_by_id = {item["question_id"]: item for item in annotations}
    random.Random(args.seed).shuffle(questions)
    split = int((1 - args.validation_fraction) * len(questions))
    private_files = read_manifest(args.private_manifest)
    safe_files = read_manifest(args.safe_manifest)
    train_examples = build_examples(
        questions[:split],
        answers_by_id,
        args.image_root,
        args.safe_root,
        private_files,
        safe_files,
    )
    eval_examples = build_examples(
        questions[split:],
        answers_by_id,
        args.image_root,
        args.safe_root,
        private_files,
        safe_files,
    )

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
        "quantization_config": quantization,
    }
    if args.cache_dir:
        model_kwargs["cache_dir"] = str(args.cache_dir)
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=None,
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        logging_steps=10,
        eval_steps=10,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=20,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        push_to_hub=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=eval_examples,
        peft_config=LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=8,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
        processing_class=processor,
    )
    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
