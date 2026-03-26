import json
import os
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


def write_caption_json(path, caption):
    # Shared output format used by downstream parsing code.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump({'caption': caption}, f)


def load_internvl_model_and_tokenizer(model_path, *, cache_dir=
    '/gpudata3/minh', device='cuda', torch_dtype=torch.bfloat16,
    use_flash_attn=True):
    model = AutoModel.from_pretrained(model_path, cache_dir=cache_dir,
        torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_flash_attn=
        use_flash_attn, trust_remote_code=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=
        cache_dir, trust_remote_code=True, use_fast=False)
    return model, tokenizer


def preprocess_image(image_path, image_size=448):
    # Match InternVL expected normalization and fixed square input size.
    image = Image.open(image_path).convert('RGB')
    transform = T.Compose([T.Resize((image_size, image_size), interpolation
        =InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(mean=(0.485,
        0.456, 0.406), std=(0.229, 0.224, 0.225))])
    return transform(image)


def run_internvl_batch(model, tokenizer, image_paths, image_classes, prompt,
    *, system_prompt, image_size=448, max_new_tokens=1024, do_sample=False,
    pad_token_id=None, format_with_class=True):
    # InternVL batch_chat expects a single stacked tensor for all images.
    pixel_values = [preprocess_image(p, image_size=image_size) for p in
        image_paths]
    pixel_values = torch.stack(pixel_values).to(torch.bfloat16).cuda()
    questions = []
    for image_class in image_classes:
        # Some prompts use {image_class}; others are fixed text blocks.
        user_text = prompt.format(image_class=image_class
            ) if format_with_class else prompt
        messages = [{'role': 'system', 'content': system_prompt}, {'role':
            'user', 'content': '<image>\n ' + user_text}]
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=
            False, add_generation_prompt=True)
        questions.append(text_prompt)
    # One visual input per sample in this path.
    num_patches_list = [1] * len(image_paths)
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample
        )
    if pad_token_id is not None:
        generation_config['pad_token_id'] = pad_token_id
    responses = model.batch_chat(tokenizer, pixel_values, num_patches_list=
        num_patches_list, questions=questions, generation_config=
        generation_config)
    return responses


def run_internvl_pair_batch(model, tokenizer, left_image_paths,
    right_image_paths, prompt, *, system_prompt, image_size=448,
    max_new_tokens=1024, do_sample=False, pad_token_id=None):
    assert len(left_image_paths) == len(right_image_paths)
    pixel_values_list = []
    num_patches_list = []
    # Pair mode packs two images per sample (left/right) for comparison prompts.
    for left_path, right_path in zip(left_image_paths, right_image_paths):
        left_img = preprocess_image(left_path, image_size)
        right_img = preprocess_image(right_path, image_size)
        pixel_values_list.extend([left_img, right_img])
        num_patches_list.append(2)
    pixel_values = torch.stack(pixel_values_list).to(torch.bfloat16).cuda()
    questions = []
    for _ in range(len(left_image_paths)):
        messages = [{'role': 'system', 'content': system_prompt}, {'role':
            'user', 'content': """Image-1: <image>
Image-2: <image>
""" +
            prompt}]
        text_prompt = tokenizer.apply_chat_template(messages, tokenize=
            False, add_generation_prompt=True)
        questions.append(text_prompt)
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=do_sample
        )
    if pad_token_id is not None:
        generation_config['pad_token_id'] = pad_token_id
    responses = model.batch_chat(tokenizer, pixel_values, num_patches_list=
        num_patches_list, questions=questions, generation_config=
        generation_config)
    return responses
