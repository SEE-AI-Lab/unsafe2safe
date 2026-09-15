import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


def load_internvl_model_and_tokenizer(model_path, cache_dir=".cache/huggingface", device="cuda"):
    model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def preprocess_image(image_path, image_size=448):
    # Match InternVL expected normalization and fixed square input size.
    with Image.open(image_path) as image_file:
        image = image_file.convert("RGB")
    transform = T.Compose(
        [
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform(image)


def run_internvl_batch(
    model,
    tokenizer,
    image_paths,
    image_classes,
    prompt,
    *,
    system_prompt,
    image_size=448,
    max_new_tokens=1024,
    do_sample=False,
    pad_token_id=None,
    format_with_class=True,
    device="cuda",
):
    # InternVL batch_chat expects a single stacked tensor for all images.
    pixel_values = [preprocess_image(p, image_size=image_size) for p in image_paths]
    pixel_values = torch.stack(pixel_values).to(device=device, dtype=torch.bfloat16)
    questions = []
    for image_class in image_classes:
        # Some prompts use {image_class}; others are fixed text blocks.
        user_text = prompt.format(image_class=image_class) if format_with_class else prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "<image>\n" + user_text},
        ]
        text_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        questions.append(text_prompt)
    # One visual input per sample in this path.
    num_patches_list = [1] * len(image_paths)
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample)
    if pad_token_id is not None:
        generation_config["pad_token_id"] = pad_token_id
    responses = model.batch_chat(
        tokenizer,
        pixel_values,
        num_patches_list=num_patches_list,
        questions=questions,
        generation_config=generation_config,
    )
    return responses
