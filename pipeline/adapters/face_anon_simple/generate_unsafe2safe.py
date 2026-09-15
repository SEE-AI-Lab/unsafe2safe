"""Batch-anonymize images with an external Face Anon Simple checkout."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_upstream(root: str | None):
    checkout = Path(root or os.environ["FACE_ANON_SIMPLE_ROOT"]).expanduser().resolve()
    if not (checkout / "src/diffusers").is_dir():
        raise ValueError(f"Not a Face Anon Simple checkout: {checkout}")
    sys.path.insert(0, str(checkout))

    from diffusers import AutoencoderKL, DDPMScheduler
    from src.diffusers.models.referencenet.referencenet_unet_2d_condition import ReferenceNetModel
    from src.diffusers.models.referencenet.unet_2d_condition import UNet2DConditionModel
    from src.diffusers.pipelines.referencenet.pipeline_referencenet import StableDiffusionReferenceNetPipeline
    from utils.anonymize_faces_in_image import anonymize_faces_in_image

    return locals()


def load_pipeline(upstream, args):
    from transformers import CLIPImageProcessor, CLIPVisionModel

    kwargs = {"use_safetensors": True}
    if args.cache_dir:
        kwargs["cache_dir"] = str(Path(args.cache_dir).expanduser())

    pipe = upstream["StableDiffusionReferenceNetPipeline"](
        unet=upstream["UNet2DConditionModel"].from_pretrained(
            args.face_model_id, subfolder="unet", **kwargs
        ),
        referencenet=upstream["ReferenceNetModel"].from_pretrained(
            args.face_model_id, subfolder="referencenet", **kwargs
        ),
        conditioning_referencenet=upstream["ReferenceNetModel"].from_pretrained(
            args.face_model_id, subfolder="conditioning_referencenet", **kwargs
        ),
        vae=upstream["AutoencoderKL"].from_pretrained(
            args.base_model_id, subfolder="vae", **kwargs
        ),
        feature_extractor=CLIPImageProcessor.from_pretrained(
            args.clip_model_id, cache_dir=args.cache_dir
        ),
        image_encoder=CLIPVisionModel.from_pretrained(
            args.clip_model_id, cache_dir=args.cache_dir
        ),
        scheduler=upstream["DDPMScheduler"].from_pretrained(
            args.base_model_id, subfolder="scheduler", **kwargs
        ),
    )
    return pipe.to(args.device)


def anonymize(image, detector, pipe, upstream, generator, args):
    image = image.convert("RGB")
    landmarks = detector.get_landmarks(np.asarray(image)[:, :, :3])
    if landmarks is None or len(landmarks) == 0:
        return image
    return upstream["anonymize_faces_in_image"](
        image=image,
        face_alignment=detector,
        pipe=pipe,
        generator=generator,
        face_image_size=args.face_image_size,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        anonymization_degree=args.anonymization_degree,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-anon-root", help="Clean upstream checkout")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--face-model-id", default="hkung/face-anon-simple")
    parser.add_argument("--base-model-id", default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--face-detector", default="sfd", choices=("sfd", "dlib"))
    parser.add_argument("--face-image-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--anonymization-degree", type=float, default=1.25)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)

    upstream = load_upstream(args.face_anon_root)
    pipe = load_pipeline(upstream, args)

    import face_alignment

    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        face_detector=args.face_detector,
    )
    generator = (
        torch.Generator(device=args.device).manual_seed(args.seed)
        if args.seed is not None
        else None
    )
    paths = sorted(
        path
        for path in input_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    if args.limit is not None:
        paths = paths[: args.limit]

    generated = skipped = 0
    for input_path in paths:
        output_path = output_root / input_path.relative_to(input_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        with Image.open(input_path) as image:
            result = anonymize(image, detector, pipe, upstream, generator, args)
        result.save(output_path)
        generated += 1

    print(f"Generated {generated}; skipped {skipped} existing output(s).")


if __name__ == "__main__":
    main()
