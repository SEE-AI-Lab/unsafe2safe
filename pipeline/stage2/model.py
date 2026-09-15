"""Unsafe2Safe training wrapper over an external InstructPix2Pix checkout."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from einops import rearrange

from .external import configure_external

# Configure the external import boundary before importing any ``ldm`` module.
configure_external()

from ldm.models.diffusion.ddpm_edit import LatentDiffusion  # noqa: E402
from ldm.modules.diffusionmodules import openaimodel  # noqa: E402

from .attention import SafeSpatialTransformer  # noqa: E402

# UNetModel resolves SpatialTransformer from this module at construction time.
# Replacing the symbol in memory keeps the external checkout's files pristine.
openaimodel.SpatialTransformer = SafeSpatialTransformer


class Unsafe2Safe(LatentDiffusion):
    """Train an editor from an unsafe image to a public/edited target image.

    The UNet receives two token-level text conditions: the public caption as a
    semantic anchor and the identity-neutral edit instruction. Both are
    produced by the frozen CLIP token encoder, yielding compatible
    ``[batch, 77, 768]`` tensors.
    """

    @torch.no_grad()
    def get_input(
        self,
        batch: Dict[str, Any],
        k: str = "image_public",
        return_first_stage_outputs: bool = False,
        bs: Optional[int] = None,
        uncond: float = 0.05,
        **_: Any,
    ) -> List[Any]:
        source = batch["image_private"]
        target = batch["image_public"]
        public_text = batch["caption_public"]
        edit_text = batch["caption_edit"]

        if bs is not None:
            source, target = source[:bs], target[:bs]
            public_text, edit_text = public_text[:bs], edit_text[:bs]

        source = source.to(self.device)
        target = target.to(self.device)
        source_latent = self.get_first_stage_encoding(self.encode_first_stage(source)).detach()
        target_latent = self.get_first_stage_encoding(self.encode_first_stage(target)).detach()

        public_embed = self.get_learned_conditioning(public_text).detach()
        edit_embed = self.get_learned_conditioning(edit_text).detach()
        null_embed = self.get_learned_conditioning([""] * source.shape[0])

        random = torch.rand(source.shape[0], device=self.device)
        text_mask = rearrange(random < 2 * uncond, "b -> b 1 1")
        image_mask = 1 - rearrange(
            ((random >= uncond) & (random < 3 * uncond)).float(),
            "b -> b 1 1 1",
        )
        public_embed = torch.where(text_mask, null_embed, public_embed)
        edit_embed = torch.where(text_mask, null_embed, edit_embed)

        # Safe attention receives the tuple as (public caption, edit text).
        # The source image remains the image condition for the hybrid UNet.
        cond = {
            "c_concat": [image_mask * source_latent],
            "c_crossattn": [public_embed, edit_embed],
        }
        outputs: List[Any] = [target_latent, cond]
        if return_first_stage_outputs:
            outputs.extend([target, self.decode_first_stage(target_latent)])
        return outputs

    def shared_step(self, batch: Dict[str, Any], **_: Any):
        target_latent, cond = self.get_input(batch, self.first_stage_key)
        return self(target_latent, cond)

    def forward(self, x, c, *args, **kwargs):
        timestep = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, timestep, *args, **kwargs)

    def apply_model(self, x_noisy, timestep, cond, return_ids=False):
        contexts = cond["c_crossattn"]
        x_input = torch.cat([x_noisy] + cond["c_concat"], dim=1)
        # Call the UNet directly: the upstream wrapper assumes one context and
        # would concatenate the two embeddings before SafeCrossAttention sees them.
        return self.model.diffusion_model(x_input, timestep, context=tuple(contexts))
