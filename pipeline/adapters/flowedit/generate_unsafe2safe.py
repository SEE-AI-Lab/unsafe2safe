"""Generate Unsafe2Safe images with an external FlowEdit checkout.

The public project owns the CSV-to-condition mapping and portable I/O. The
FlowEdit sampler remains in the separately installed upstream repository.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _configure_external(root):
    """Import samplers from a clean FlowEdit checkout without modifying it."""

    checkout = Path(root or os.environ["FLOWEDIT_ROOT"]).expanduser().resolve()
    sys.path.insert(0, str(checkout))
    from FlowEdit_utils import FlowEditFLUX, FlowEditSD3  # type: ignore

    return FlowEditSD3, FlowEditFLUX


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dtype(name: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return values[name.lower()]


def _load_pipeline(model_type: str, model_id: str, dtype: torch.dtype):
    from diffusers import FluxPipeline, StableDiffusion3Pipeline

    pipeline_class = {"SD3": StableDiffusion3Pipeline, "FLUX": FluxPipeline}[model_type.upper()]
    return pipeline_class.from_pretrained(model_id, torch_dtype=dtype)


def _prepare_image(image: Image.Image, max_resolution: int) -> Image.Image:
    image = image.convert("RGB")
    width = image.width - image.width % 16
    height = image.height - image.height % 16
    image = image.crop((0, 0, width, height))
    if max(image.width, image.height) > max_resolution:
        scale = min(max_resolution / image.width, max_resolution / image.height)
        width = max(16, int(image.width * scale) // 16 * 16)
        height = max(16, int(image.height * scale) // 16 * 16)
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    return image


def _encode_source(pipe, image: Image.Image, device: torch.device) -> torch.Tensor:
    dtype = pipe.vae.dtype
    image_src = pipe.image_processor.preprocess(image).to(device=device, dtype=dtype)
    with torch.autocast("cuda", dtype=dtype), torch.inference_mode():
        encoded = pipe.vae.encode(image_src).latent_dist.mode()
    return (encoded - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor


def _sample(sampler, pipe, scheduler, latent, source_prompt: str, target_prompt: str, config: Dict[str, Any]):
    return sampler(
        pipe,
        scheduler,
        latent,
        source_prompt,
        target_prompt,
        str(config.get("negative_prompt", "")),
        int(config["steps"]),
        int(config["n_avg"]),
        float(config["src_guidance_scale"]),
        float(config["tar_guidance_scale"]),
        int(config["n_min"]),
        int(config["n_max"]),
    )


def _decode(pipe, latent: torch.Tensor) -> Image.Image:
    denormalized = latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    with torch.autocast("cuda", dtype=pipe.vae.dtype), torch.inference_mode():
        decoded = pipe.vae.decode(denormalized, return_dict=False)[0]
    return pipe.image_processor.postprocess(decoded)[0]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.example.yaml")))
    parser.add_argument("--flowedit-root", help="Clean upstream FlowEdit checkout")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-column", required=True)
    parser.add_argument("--device")
    parser.add_argument("--model-id")
    parser.add_argument("--file-column")
    parser.add_argument("--exclude-file-prefix")
    parser.add_argument("--condition", required=True, metavar="COLUMN")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(Path(args.config).expanduser())
    if args.device:
        config["device"] = args.device
    if args.model_id:
        config["model_id"] = args.model_id
    if args.file_column:
        config["file_column"] = args.file_column
    if args.exclude_file_prefix:
        config["exclude_file_prefix"] = args.exclude_file_prefix

    device = torch.device(str(config.get("device", "cuda")))
    model_type = str(config.get("model_type", "SD3")).upper()
    flowedit_sd3, flowedit_flux = _configure_external(args.flowedit_root)
    pipe = _load_pipeline(
        model_type,
        str(config["model_id"]),
        _dtype(str(config.get("dtype", "float16"))),
    ).to(device)
    scheduler = pipe.scheduler
    sampler = flowedit_sd3 if model_type == "SD3" else flowedit_flux

    frame = pd.read_csv(args.input_csv)
    file_column = str(config.get("file_column", "file"))
    source_column = str(args.source_column)
    condition_column = str(args.condition)
    prefix = str(config.get("exclude_file_prefix", "val"))
    frame = frame[~frame[file_column].astype(str).str.startswith(prefix)]
    frame = frame.sample(frac=1.0, random_state=int(config.get("seed", 42))).reset_index(drop=True)
    if args.limit is not None:
        frame = frame.head(args.limit)

    image_root = Path(args.image_root).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(int(config.get("seed", 42)))
    generated = 0
    skipped = 0
    for row in frame.to_dict(orient="records"):
        relative = Path(str(row[file_column]))
        input_path = image_root / relative
        output_path = output_dir / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        with Image.open(input_path) as source_image:
            image = _prepare_image(source_image, int(config.get("max_resolution", 1536)))
        latent = _encode_source(pipe, image, device)
        edited = _sample(sampler, pipe, scheduler, latent, str(row[source_column]), str(row[condition_column]), config)
        _decode(pipe, edited).save(output_path)
        generated += 1
    print(f"generated {generated}, skipped {skipped}")


if __name__ == "__main__":
    main()
