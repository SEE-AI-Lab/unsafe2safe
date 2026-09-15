"""Compute the paper's SSIM and LPIPS cheating scores for image pairs."""

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def load_pair(original_path, edited_path):
    """Load an image pair and resize the edited image to the original size."""
    with Image.open(original_path) as original_file, Image.open(edited_path) as edited_file:
        original = original_file.convert("RGB")
        edited = edited_file.convert("RGB").resize(original.size, Image.Resampling.BICUBIC)
    return np.asarray(original), np.asarray(edited)


def compute_ssim(original_path, edited_path) -> float:
    """Return structural similarity between an original and edited image."""
    original, edited = load_pair(original_path, edited_path)
    return float(structural_similarity(original, edited, channel_axis=2, data_range=255))


def compute_lpips(original_path, edited_path, device="cpu") -> float:
    """Return VGG-16 LPIPS distance between an original and edited image."""
    original, edited = load_pair(original_path, edited_path)
    original_tensor = torch.from_numpy(original).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0)
    edited_tensor = torch.from_numpy(edited).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0)
    model = lpips.LPIPS(net="vgg").to(device).eval()
    with torch.no_grad():
        return float(model(original_tensor.to(device), edited_tensor.to(device)).item())
