"""Compare raw and anonymized images with an InternVL privacy judge."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

from pipeline.stage1.internvl_common import load_internvl_model_and_tokenizer, preprocess_image
from pipeline.stage1.qwen_common import write_caption_json


def run_pair_batch(model, tokenizer, raw_paths, anonymized_paths, prompt, *, system_prompt, image_size=448, max_new_tokens=512, device="cuda"):
    pixel_values = []
    for raw_path, anonymized_path in zip(raw_paths, anonymized_paths):
        pixel_values.extend([preprocess_image(raw_path, image_size=image_size), preprocess_image(anonymized_path, image_size=image_size)])
    pixel_values = torch.stack(pixel_values).to(device=device, dtype=torch.bfloat16)
    questions = []
    for _ in raw_paths:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Image-1: <image>\nImage-2: <image>\n" + prompt}]
        questions.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    generation_config = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.eos_token_id}
    return model.batch_chat(tokenizer, pixel_values, num_patches_list=[2] * len(raw_paths), questions=questions, generation_config=generation_config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, help="CSV containing one relative file path per image pair")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--anonymized-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", default="prompts/intern_image_flagging-compare.txt")
    parser.add_argument("--model-id", default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--system-prompt", default="You are a vision-language evaluator for privacy-preserving image anonymization.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    files = [Path(str(value)) for value in pd.read_csv(args.input_csv)["file"]]
    prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    model, tokenizer = load_internvl_model_and_tokenizer(args.model_id, device=args.device)
    output_dir = Path(args.output_dir)

    for start in tqdm(range(0, len(files), args.batch_size), desc="compare raw/anonymized"):
        batch = files[start : start + args.batch_size]
        outputs = run_pair_batch(model, tokenizer, [Path(args.raw_root) / file for file in batch], [Path(args.anonymized_root) / file for file in batch], prompt, system_prompt=args.system_prompt, image_size=args.image_size, max_new_tokens=args.max_new_tokens, device=args.device)
        for file, output in zip(batch, outputs):
            write_caption_json(output_dir / f"{file.with_suffix('')}_caption.json", output)


if __name__ == "__main__":
    main()
