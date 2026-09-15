"""Legacy InstructPix2Pix batch editor.

This script keeps the original single-context CFG sampling path for existing
experiments. New Unsafe2Safe training uses ``unsafe2safe_model.py`` and the
external-checkout import boundary instead.
"""

from __future__ import annotations

import random
import sys
from argparse import ArgumentParser
import os
import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast
import pandas as pd
from tqdm import tqdm

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        B = z.shape[0]

        # --- Repeat z and sigma 3x along batch to align with the 3 branches ---
        # Pattern: [B, ...] -> [3B, ...]
        cfg_z = einops.repeat(z, 'b c h w -> (b r) c h w', r=3)

        # Make sigma broadcastable to [B, ...] if it isn't already (e.g., scalar/shape [1])
        if not hasattr(sigma, 'shape') or sigma.shape == torch.Size([]):
            sigma = sigma.expand(B)  # scalar -> [B]
        elif sigma.shape[0] == 1 and B > 1:
            sigma = sigma.expand(B, *sigma.shape[1:])  # [1, ...] -> [B, ...]

        cfg_sigma = einops.repeat(sigma, 'b ... -> (b r) ...', r=3)

        # --- Build concatenated conditioning (3 branches stacked along batch) ---
        #  Order must match how we will chunk the outputs: [cond, img_cond, uncond]
        #  - text branch:   cond["c_crossattn"][0], cond["c_concat"][0]
        #  - image branch:  uncond["c_crossattn"][0], cond["c_concat"][0]
        #  - uncond branch: uncond["c_crossattn"][0], uncond["c_concat"][0]
        c_txt_cond   = cond["c_crossattn"][0]     # [B, ...]
        c_img_latent = cond["c_concat"][0]        # [B, C, H, W]
        c_txt_null   = uncond["c_crossattn"][0]   # [B, ...]
        c_img_null   = uncond["c_concat"][0]      # [B, C, H, W]

        cfg_cond = {
            "c_crossattn": [torch.cat([c_txt_cond, c_txt_null, c_txt_null], dim=0)],  # [3B, ...]
            "c_concat":    [torch.cat([c_img_latent, c_img_latent, c_img_null], dim=0)],  # [3B, C, H, W]
        }

        # --- Single inner forward over the 3B batch, then split back into 3 chunks of size B ---
        out = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond)  # [3B, C, H, W]
        out_cond, out_img_cond, out_uncond = out.chunk(3, dim=0)  # each [B, C, H, W]
        #print(out_uncond[0], out_cond[0], out_img_cond[0])
        # --- Combine with text/image guidance scales (per batch) ---
        guided = (
            out_uncond
            + text_cfg_scale  * (out_cond    - out_img_cond)
            + image_cfg_scale * (out_img_cond - out_uncond)
        )
        print(guided[0])
        return guided


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
    side = min(img.size)  # pick the shorter side
    img = ImageOps.fit(img, (side, side), centering=(0.5, 0.5))  # crop square
    return img.resize((target_res, target_res), Image.Resampling.LANCZOS)

def pad_square_white(img, target_res):
    s = max(img.size)
    new = Image.new("RGB", (s, s), (255, 255, 255))
    new.paste(img, ((s - img.width)//2, (s - img.height)//2))
    return new.resize((target_res, target_res), Image.Resampling.LANCZOS)

def preprocess_image(path, resolution, device):
    """Load and resize image to match model resolution requirements."""
    img = Image.open(path).convert("RGB")
    #w, h = img.size
    #factor = resolution / max(w, h)
    #factor = math.ceil(min(w, h) * factor / 64) * 64 / min(w, h)
    #w = int((w * factor) // 64) * 64
    #h = int((h * factor) // 64) * 64
    #img = ImageOps.fit(img, (w, h), method=Image.Resampling.LANCZOS)
    img = center_crop_square(img, target_res=resolution)
    tensor = 2 * torch.tensor(np.array(img)).float() / 255 - 1
    tensor = rearrange(tensor, "h w c -> 1 c h w").to(device)
    return img, tensor


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/train.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str, help="CSV with path, edit_caption")
    parser.add_argument("--image-root", required=True, type=str, help="Directory containing source images")
    parser.add_argument("--output", required=True, type=str, help="Directory to save edited images")
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # Load config/model
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    null_token = model.get_learned_conditioning([""])

    os.makedirs(args.output, exist_ok=True)
    # ----------------------------------------------------------------------
    # Construct destination folder name
    # ----------------------------------------------------------------------
    # 1. CSV file base name
    csv_name = os.path.splitext(os.path.basename(args.input))[0]

    # 2. Last 3 directories of checkpoint path
    ckpt_parts = os.path.normpath(args.ckpt).split(os.sep)
    ckpt_tail = os.path.join(*ckpt_parts[-3:]) if len(ckpt_parts) >= 3 else os.path.basename(args.ckpt)

    # Final destination folder
    dest_dir = os.path.join(args.output, f"{csv_name}_{ckpt_tail}")
    print(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    df = pd.read_csv(args.input)
    device = model.device

    # Iterate in batches
    for i in tqdm(range(0, len(df), args.batch_size), desc="Editing images"):
        batch_df = df.iloc[i:i + args.batch_size]

        images, tensors, captions = [], [], []
        for _, row in batch_df.iterrows():
            img_path, caption = row["file"], row["PRIVACY_PRESERVING_DESCRIPTION"]
            img, tensor = preprocess_image(os.path.join(args.image_root, img_path), args.resolution, device)
            #print(tensor.shape)
            images.append((img, img_path))
            tensors.append(tensor)
            captions.append(caption)

        tensors = torch.cat(tensors, dim=0)

        # Build conditioning
        cond = {
            "c_crossattn": [model.get_learned_conditioning(captions)],
            "c_concat": [model.encode_first_stage(tensors).mode()],
        }
        uncond = {
            "c_crossattn": [null_token.repeat(len(captions), 1, 1)],
            "c_concat": [torch.zeros_like(cond["c_concat"][0])],
        }

        sigmas = model_wrap.get_sigmas(args.steps)
        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "text_cfg_scale": args.cfg_text,
            "image_cfg_scale": args.cfg_image,
        }

        seed = random.randint(0, 100000) if args.seed is None else args.seed
        torch.manual_seed(seed)

        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        print(cond["c_concat"][0].shape)
        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            #z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas, extra_args=extra_args)
            z, _ = model.sample_log(
                cond=cond,
                batch_size=args.batch_size,
                ddim=False,
                ddim_steps=200,
                eta=1.0,
            )

            x = model.decode_first_stage(z)

        # Save results
        x = torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
        x = (255.0 * rearrange(x, "b c h w -> b h w c")).type(torch.uint8).cpu().numpy()

        for arr, (_, fname) in zip(x, images):
            out_path = os.path.join(dest_dir, fname)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            Image.fromarray(arr).save(out_path)
