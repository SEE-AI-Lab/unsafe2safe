"""Batch-anonymize images with an external Face Anon Simple checkout.

The ReferenceNet implementation and face-alignment utilities remain in the
separately installed upstream repository. This module owns only the
Unsafe2Safe batch contract and its no-face handling.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _configure_external(root: str | None) -> dict[str, Any]:
    """Load upstream modules from a clean Face Anon Simple checkout."""

    configured = root or os.environ.get("FACE_ANON_SIMPLE_ROOT")
    if not configured:
        raise ImportError(
            "Set FACE_ANON_SIMPLE_ROOT to a clean face_anon_simple checkout; "
            "upstream source is intentionally not vendored in Unsafe2Safe."
        )

    checkout = Path(configured).expanduser().resolve()
    required = (
        checkout / "src" / "diffusers" / "models" / "referencenet",
        checkout / "src" / "diffusers" / "pipelines" / "referencenet",
        checkout / "utils" / "extractor.py",
        checkout / "utils" / "merger.py",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ImportError(
            f"FACE_ANON_SIMPLE_ROOT is not a complete upstream checkout: {checkout}; "
            f"missing {', '.join(missing)}"
        )

    sys.path.insert(0, str(checkout))

    from diffusers import AutoencoderKL, DDPMScheduler
    from src.diffusers.models.referencenet.referencenet_unet_2d_condition import (
        ReferenceNetModel,
    )
    from src.diffusers.models.referencenet.unet_2d_condition import (
        UNet2DConditionModel,
    )
    from src.diffusers.pipelines.referencenet.pipeline_referencenet import (
        StableDiffusionReferenceNetPipeline,
    )
    from utils.extractor import extract_faces
    from utils.merger import paste_foreground_onto_background

    return {
        "AutoencoderKL": AutoencoderKL,
        "DDPMScheduler": DDPMScheduler,
        "ReferenceNetModel": ReferenceNetModel,
        "UNet2DConditionModel": UNet2DConditionModel,
        "StableDiffusionReferenceNetPipeline": StableDiffusionReferenceNetPipeline,
        "extract_faces": extract_faces,
        "paste_foreground_onto_background": paste_foreground_onto_background,
        "checkout": checkout,
    }


def _load_pipeline(
    upstream: dict[str, Any],
    face_model_id: str,
    base_model_id: str,
    clip_model_id: str,
    cache_dir: str | None,
    device: str,
) -> Any:
    """Construct the upstream ReferenceNet pipeline with explicit model IDs."""

    from transformers import CLIPImageProcessor, CLIPVisionModel

    load_kwargs: dict[str, Any] = {"use_safetensors": True}
    if cache_dir:
        load_kwargs["cache_dir"] = str(Path(cache_dir).expanduser())

    unet = upstream["UNet2DConditionModel"].from_pretrained(
        face_model_id, subfolder="unet", **load_kwargs
    )
    referencenet = upstream["ReferenceNetModel"].from_pretrained(
        face_model_id, subfolder="referencenet", **load_kwargs
    )
    conditioning_referencenet = upstream["ReferenceNetModel"].from_pretrained(
        face_model_id, subfolder="conditioning_referencenet", **load_kwargs
    )
    vae = upstream["AutoencoderKL"].from_pretrained(
        base_model_id, subfolder="vae", **load_kwargs
    )
    scheduler = upstream["DDPMScheduler"].from_pretrained(
        base_model_id, subfolder="scheduler", **load_kwargs
    )
    feature_extractor = CLIPImageProcessor.from_pretrained(
        clip_model_id, cache_dir=cache_dir
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        clip_model_id, cache_dir=cache_dir
    )

    pipe = upstream["StableDiffusionReferenceNetPipeline"](
        unet=unet,
        referencenet=referencenet,
        conditioning_referencenet=conditioning_referencenet,
        vae=vae,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        scheduler=scheduler,
    )
    return pipe.to(device)


class _CachedLandmarks:
    """Present one detector result to upstream ``extract_faces``.

    The upstream extractor calls ``get_landmarks`` itself. Caching the first
    result avoids running the face detector twice while allowing this adapter
    to handle the upstream ``None`` result without modifying upstream code.
    """

    def __init__(self, landmarks: Any) -> None:
        self._landmarks = landmarks

    def get_landmarks(self, _image: np.ndarray) -> Any:
        return self._landmarks


def anonymize_faces_in_image(
    image: Image.Image,
    detector: Any,
    pipe: Any,
    extract_faces: Callable[..., tuple[list[Image.Image] | None, list[Any] | None]],
    paste_foreground_onto_background: Callable[..., Image.Image],
    *,
    generator: torch.Generator | None = None,
    face_image_size: int = 512,
    num_inference_steps: int = 25,
    guidance_scale: float = 4.0,
    anonymization_degree: float = 1.25,
) -> Image.Image:
    """Anonymize every detected face, leaving no-face images unchanged."""

    rgb_image = image.convert("RGB")
    landmarks = detector.get_landmarks(np.asarray(rgb_image)[:, :, :3])
    if landmarks is None or len(landmarks) == 0:
        return rgb_image

    face_images, image_to_face_matrices = extract_faces(
        _CachedLandmarks(landmarks),
        rgb_image,
        face_image_size,
    )
    if not face_images or not image_to_face_matrices:
        return rgb_image

    anonymized = rgb_image
    for face_image, image_to_face_matrix in zip(
        face_images, image_to_face_matrices
    ):
        anonymized_face = pipe(
            source_image=face_image,
            conditioning_image=face_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            anonymization_degree=anonymization_degree,
            width=face_image_size,
            height=face_image_size,
        ).images[0]
        anonymized = paste_foreground_onto_background(
            anonymized_face,
            anonymized,
            image_to_face_matrix,
        )

    return anonymized


def _iter_images(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-anon-root", help="Clean upstream face_anon_simple checkout")
    parser.add_argument("--input-root", required=True, help="Root containing source images")
    parser.add_argument("--output-root", required=True, help="Root for anonymized images")
    parser.add_argument("--face-model-id", default="hkung/face-anon-simple")
    parser.add_argument("--base-model-id", default="stabilityai/stable-diffusion-2-1")
    parser.add_argument("--clip-model-id", default="openai/clip-vit-large-patch14")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--face-detector", default="sfd", choices=("sfd", "dlib"))
    parser.add_argument("--face-image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--anonymization-degree", type=float, default=1.25)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.face_image_size <= 0 or args.num_inference_steps <= 0:
        raise ValueError("face image size and inference steps must be positive")
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    upstream = _configure_external(args.face_anon_root)
    pipe = _load_pipeline(
        upstream,
        args.face_model_id,
        args.base_model_id,
        args.clip_model_id,
        args.cache_dir,
        args.device,
    )

    import face_alignment

    detector = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        face_detector=args.face_detector,
    )
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    paths = list(_iter_images(input_root))
    if args.limit is not None:
        paths = paths[: args.limit]

    generated = skipped = 0
    for input_path in paths:
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue

        with Image.open(input_path) as source:
            anonymized = anonymize_faces_in_image(
                source,
                detector,
                pipe,
                upstream["extract_faces"],
                upstream["paste_foreground_onto_background"],
                generator=generator,
                face_image_size=args.face_image_size,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                anonymization_degree=args.anonymization_degree,
            )
        anonymized.save(output_path)
        generated += 1

    print(f"Generated {generated} image(s); skipped {skipped} existing output(s).")


if __name__ == "__main__":
    main()
