import json
import os
from pathlib import Path

from transformers import pipeline


def write_caption_json(path, caption):
    # Keep a stable {"caption": "..."} payload across all generators.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"caption": caption}, handle, ensure_ascii=False)


def build_text_generator(
    model_id,
    *,
    hf_home=None,
    torch_dtype="auto",
    device_map="cuda",
):
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    return pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )


def _extract_assistant_text(result):
    return result["generated_text"][-1]["content"].strip()


def run_text_batch(generator, messages_batch, *, max_new_tokens=512, batch_size=32):
    generator.tokenizer.padding_side = "left"
    generator.tokenizer.pad_token = generator.tokenizer.eos_token

    results = generator(messages_batch, max_new_tokens=max_new_tokens, batch_size=batch_size)
    return [_extract_assistant_text(result) for result in results]
