import json
import os
from qwen_vl_utils import process_vision_info
from transformers import pipeline


def write_caption_json(path, caption):
    # Keep a stable {"caption": "..."} payload across all generators.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"caption": caption}, f)


def build_text_generator(
    model_id,
    *,
    hf_home=None,
    torch_dtype="auto",
    device_map="cuda",
):
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    return pipeline("text-generation", model_id, torch_dtype=torch_dtype, device_map=device_map)


def run_text_batch(generator, messages_batch, *, max_new_tokens=512, batch_size=32):
    generator.tokenizer.padding_side = "left"
    if generator.tokenizer.pad_token is None:
        generator.tokenizer.pad_token = generator.tokenizer.eos_token

    results = generator(messages_batch, max_new_tokens=max_new_tokens, batch_size=batch_size)
    # HF chat pipeline returns full message history; we only keep latest assistant text.
    return [result[0]["generated_text"][-1]["content"].strip() for result in results]


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
