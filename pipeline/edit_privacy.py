"""Legacy single-image privacy editor.

It is retained for compatibility with older checkpoints and configs. It does
not implement the two-context Unsafe2Safe training path.
"""

from __future__ import annotations

import math
import random
import sys
from argparse import ArgumentParser

import einops
import k_diffusion as K
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image, ImageOps
from torch import autocast

sys.path.append("./stable_diffusion")

from stable_diffusion.ldm.util import instantiate_from_config


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)


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


def main():
    parser = ArgumentParser()
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--config", default="configs/generate.yaml", type=str)
    parser.add_argument("--ckpt", default="checkpoints/instruct-pix2pix-00-22000.ckpt", type=str)
    parser.add_argument("--vae-ckpt", default=None, type=str)
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--edit", required=True, type=str)
    parser.add_argument("--cfg-text", default=7.5, type=float)
    parser.add_argument("--cfg-image", default=1.5, type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, args.ckpt, args.vae_ckpt)
    model.eval().cuda()
    model_wrap = K.external.CompVisDenoiser(model)
    model_wrap_cfg = CFGDenoiser(model_wrap)
    null_token = model.get_learned_conditioning([""])

    seed = random.randint(0, 100000) if args.seed is None else args.seed
    input_image = Image.open(args.input).convert("RGB")
    #width, height = input_image.size
    #factor = args.resolution / max(width, height)
    #factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    #width = int((width * factor) // 64) * 64
    #height = int((height * factor) // 64) * 64
    #input_image = ImageOps.fit(input_image, (width, height), method=Image.Resampling.LANCZOS)
    def center_crop_square(img, target_res):
        side = min(img.size)  # pick the shorter side
        img = ImageOps.fit(img, (side, side), centering=(0.5, 0.5))  # crop square
        return img.resize((target_res, target_res), Image.Resampling.LANCZOS)
    def pad_square_white(img, target_res):
        s = max(img.size)
        new = Image.new("RGB", (s, s), (255, 255, 255))
        new.paste(img, ((s - img.width)//2, (s - img.height)//2))
        return new.resize((target_res, target_res), Image.Resampling.LANCZOS)

    input_image = center_crop_square(input_image, target_res=args.resolution)
    with torch.no_grad():
        if args.edit == "":
            input_image.save(args.output)
            return

        with torch.no_grad(), autocast("cuda"), model.ema_scope():
            c = {}
            c["c_crossattn"] = [model.get_learned_conditioning([args.edit])]

            # Image conditioning (concat)
            #print(input_image.shape)
            input_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1  # [0,255] -> [-1,1]
            input_tensor = rearrange(input_tensor, "h w c -> 1 c h w").to(model.device)
            c["c_concat"] = [model.encode_first_stage(input_tensor).mode()]
            print(input_tensor.shape, model.encode_first_stage(input_tensor).mode().shape)
            samples_z, z_denoise_row = model.sample_log(
                cond=c,
                batch_size=1,
                ddim=False,
                ddim_steps=200,
                eta=1.0,
            )

            # Decode latent → image in pixel space
            x = model.decode_first_stage(samples_z)
        torch.save(x, args.output.replace("jpg", "pt"))
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        edited_image = Image.fromarray(x.type(torch.uint8).cpu().numpy())
    edited_image.save(args.output)


if __name__ == "__main__":
    main()
