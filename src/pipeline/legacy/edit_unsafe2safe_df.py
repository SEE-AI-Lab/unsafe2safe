"""Legacy batch Unsafe2Safe editor for older copied diffusion configs.

Use the external InstructPix2Pix adapter for the minimal training path. This
CLI remains only for reproducing earlier inference experiments and expects
the legacy ``stable_diffusion/`` layout described in the README.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
import os
from pathlib import Path
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import pandas as pd
from tqdm import tqdm

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


def load_model_from_config(config, ckpt, vae_ckpt=None, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    if vae_ckpt is not None:
        print(f"Loading VAE from {vae_ckpt}")
        vae_sd = torch.load(vae_ckpt, map_location="cpu")["state_dict"]
        sd = {
            k: vae_sd[k[len("first_stage_model.") :]] if k.startswith("first_stage_model.") else v
            for k, v in sd.items()
        }
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

def center_crop_square(img, target_res):
    side = min(img.size)
    img = ImageOps.fit(img, (side, side), centering=(0.5, 0.5))
    return img.resize((target_res, target_res), Image.Resampling.LANCZOS)

def preprocess_image(path, resolution, device):
    """Load and resize image to match model resolution requirements."""
    with Image.open(path) as image:
        img = center_crop_square(image.convert("RGB"), target_res=resolution)
    tensor = 2 * torch.tensor(np.array(img)).float() / 255 - 1
    tensor = rearrange(tensor, "h w c -> 1 c h w").to(device)
    return tensor


def relative_path(value):
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"CSV file paths must stay relative: {path}")
    return path


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/train.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str, help="CSV containing image and text columns")
    parser.add_argument("--image-root", required=True, type=str, help="Directory containing source images")
    parser.add_argument("--output", required=True, type=str, help="Directory to save edited images")
    parser.add_argument("--file-column", default="file")
    parser.add_argument("--public-caption-column", default="caption_public")
    parser.add_argument("--edit-caption-column", default="caption_edit")
    parser.add_argument("--batch-size", default=16, type=int)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()

    os.makedirs(args.output, exist_ok=True)
    csv_name = os.path.splitext(os.path.basename(args.input))[0]
    ckpt_parts = os.path.normpath(args.ckpt).split(os.sep)
    ckpt_tail = os.path.join(*ckpt_parts[-3:]) if len(ckpt_parts) >= 3 else os.path.basename(args.ckpt)
    dest_dir = os.path.join(args.output, f"{csv_name}_{ckpt_tail}")
    print(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    required_columns = {
        args.file_column,
        args.public_caption_column,
        args.edit_caption_column,
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"CSV is missing columns: {missing_columns}")
    df = df[
        df[args.file_column].astype(str).str.contains("val2014", na=False)
    ].reset_index(drop=True)
    device = model.device

    for i in tqdm(range(0, len(df), args.batch_size), desc="Editing images"):
        batch_df = df.iloc[i:i + args.batch_size]

        file_paths, tensors, captions_public, captions_edit = [], [], [], []
        for _, row in batch_df.iterrows():
            img_path = relative_path(row[args.file_column])
            public_caption = row[args.public_caption_column]
            edit_caption = row[args.edit_caption_column]
            file_paths.append(img_path)
            tensors.append(preprocess_image(Path(args.image_root) / img_path, args.resolution, device))
            captions_public.append(public_caption)
            captions_edit.append(edit_caption)

        tensors = torch.cat(tensors, dim=0)

        z_priv = model.encode_first_stage(tensors).mode().detach()
        c_public_embed = model.get_learned_conditioning(captions_public).detach()
        c_edit_embed = model.cond_stage_model.encode_text_pooled(captions_edit).detach()

        cond = {
            "c_concat": [z_priv],
            "c_crossattn": [c_public_embed],
        }
        cond["c_crossattn"].extend([c_edit_embed])

        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            z, _ = model.sample_log(
                cond=cond,
                batch_size=len(batch_df),
                ddim=False,
                ddim_steps=args.steps,
                eta=1.0,
            )

            x = model.decode_first_stage(z)

        x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        x = (255.0 * rearrange(x, "b c h w -> b h w c")).type(torch.uint8).cpu().numpy()

        for arr, fname in zip(x, file_paths):
            out_path = Path(dest_dir) / fname
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(out_path)
