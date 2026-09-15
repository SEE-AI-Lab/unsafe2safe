import json
import os
from pathlib import Path

from qwen_vl_utils import process_vision_info
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
    """Normalize the chat pipeline's version-dependent output shape."""
    generated = result.get("generated_text", result) if isinstance(result, dict) else result
    if isinstance(generated, str):
        return generated.strip()
    if isinstance(generated, list):
        for message in reversed(generated):
            if isinstance(message, dict) and message.get("role") == "assistant":
                return str(message.get("content", "")).strip()
        if generated and isinstance(generated[-1], dict):
            return str(generated[-1].get("content", "")).strip()
    raise ValueError(f"Unsupported text-generation output: {type(result).__name__}")


def run_text_batch(generator, messages_batch, *, max_new_tokens=512, batch_size=32):
    generator.tokenizer.padding_side = "left"
    if generator.tokenizer.pad_token is None:
        generator.tokenizer.pad_token = generator.tokenizer.eos_token

    results = generator(messages_batch, max_new_tokens=max_new_tokens, batch_size=batch_size)
    return [_extract_assistant_text(result) for result in results]


def run_vl_batch(model, processor, messages_batch, *, device="cuda", max_new_tokens=1024):
    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages_batch
    ]

    image_inputs, _ = process_vision_info(messages_batch)
    inputs = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True).to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    # Remove echoed prompt tokens so decode returns only generated completion.
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
