from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from PIL import Image


def main() -> None:
    """Generate safe images from unsafe inputs with an OminiControl adapter."""

    parser = argparse.ArgumentParser(description="Generate Unsafe2Safe images with OminiControl")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base-model", default="black-forest-labs/FLUX.1-dev")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Import the separately installed upstream OminiControl implementation;
    # this repository provides only the Unsafe2Safe data-to-condition mapping.
    from diffusers import FluxPipeline
    from omini.pipeline.flux_omini import Condition, generate, seed_everything

    seed_everything(args.seed)
    pipe = FluxPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    pipe.load_lora_weights(
        args.checkpoint,
        weight_name="default.safetensors",
        adapter_name="subject",
        local_files_only=True,
    )

    frame = pd.read_csv(args.input_csv)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for row in frame.to_dict(orient="records"):
        relative_path = Path(str(row["file"]))
        output_path = output_dir / relative_path
        # Allow an interrupted generation run to resume without overwriting outputs.
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            continue

        source = Image.open(image_root / relative_path).convert("RGB")
        source = source.resize((args.image_size, args.image_size), Image.Resampling.LANCZOS)
        condition = Condition(source, "subject", position_delta=(0, 0))
        with torch.no_grad():
            result = generate(
                pipe,
                prompt=str(row["caption"]),
                conditions=[condition],
                num_inference_steps=args.steps,
                height=args.image_size,
                width=args.image_size,
            ).images[0]
        result.save(output_path)


if __name__ == "__main__":
    main()
