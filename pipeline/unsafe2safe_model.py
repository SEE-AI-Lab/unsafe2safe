"""Project-specific training wrapper for a compatible diffusion base model."""

import torch
from einops import rearrange

from ldm.models.diffusion.ddpm_edit import LatentDiffusion


class Unsafe2Safe(LatentDiffusion):
    """Train a diffusion editor with private and public caption conditioning."""

    def __init__(self, *args, conditioning_key=None, load_ema=True, **kwargs):
        super().__init__(conditioning_key=conditioning_key, *args, load_ema=load_ema, **kwargs)

    def instantiate_unet(self):
        """Keep the base UNet fixed and train only the Safe Attention parameters."""
        privacy_params = ("to_k_priv", "to_k_pub", "to_v_priv", "to_v_pub", "map_fuse", "attn2.to_out", "priv_filter")
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = any(name_part in name for name_part in privacy_params)

    def get_input(self, batch, bs=None, return_first_stage_outputs=False, return_original_cond=False, uncond=0.05):
        image_private = batch["image_private"][:bs].to(self.device) if bs is not None else batch["image_private"].to(self.device)
        image_public = batch["image_public"][:bs].to(self.device) if bs is not None else batch["image_public"].to(self.device)
        private_captions = batch["caption_private"][:bs] if bs is not None else batch["caption_private"]
        edit_captions = batch["caption_edit"][:bs] if bs is not None else batch["caption_edit"]

        z_private = self.encode_first_stage(image_private).mode().detach()
        z_public = self.get_first_stage_encoding(self.encode_first_stage(image_public)).detach()
        private_embed = self.get_learned_conditioning(private_captions).detach()
        edit_embed = self.cond_stage_model.encode_text_pooled(edit_captions).detach()

        random_private = torch.rand(z_private.size(0), device=self.device)
        private_prompt_mask = rearrange(random_private < 2 * uncond, "n -> n 1 1")
        image_mask = 1 - rearrange(((random_private >= uncond) & (random_private < 3 * uncond)).float(), "n -> n 1 1 1")
        random_edit = torch.rand(z_private.size(0), device=self.device)
        edit_prompt_mask = rearrange(random_edit < 2 * uncond, "n -> n 1")

        null_private = self.get_learned_conditioning([""])
        null_edit = self.cond_stage_model.encode_text_pooled([""])
        private_embed = torch.where(private_prompt_mask, null_private, private_embed)
        edit_embed = torch.where(edit_prompt_mask, null_edit, edit_embed)
        private_condition = {"c_concat": [image_mask * z_private], "c_crossattn": [private_embed]}

        output = [z_public, private_condition, edit_embed]
        if return_first_stage_outputs:
            output.append(self.decode_first_stage(z_public))
        if return_original_cond:
            output.extend([image_private, image_public])
        return output

    def shared_step(self, batch):
        z_public, private_condition, edit_embed = self.get_input(batch)
        return self(z_public, private_condition, edit_embed)

    def forward(self, x, condition, edit_embed, *args, **kwargs):
        timestep = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, condition, timestep, edit_embed, *args, **kwargs)

    def p_losses(self, x_start, condition, timestep, edit_embed, noise=None):
        noise = torch.randn_like(x_start) if noise is None else noise
        x_noisy = self.q_sample(x_start=x_start, t=timestep, noise=noise)
        condition = dict(condition)
        condition["c_crossattn"] = list(condition["c_crossattn"]) + [edit_embed]
        model_output, _ = self.apply_model(x_noisy, timestep, condition)

        target = x_start if self.parameterization == "x0" else noise
        if self.parameterization not in ("x0", "eps"):
            raise NotImplementedError()
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        prefix = "train" if self.training else "val"
        loss_dict = {f"{prefix}/loss_simple": loss_simple.mean()}
        logvar = self.logvar.to(self.device)[timestep]
        loss = loss_simple / torch.exp(logvar) + logvar
        if self.learn_logvar:
            loss_dict.update({f"{prefix}/loss_gamma": loss.mean(), "logvar": self.logvar.data.mean()})
        loss = self.l_simple_weight * loss.mean()
        loss_vlb = (self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3)) * self.lvlb_weights[timestep]).mean()
        loss_dict[f"{prefix}/loss_vlb"] = loss_vlb
        loss += self.original_elbo_weight * loss_vlb
        loss_dict[f"{prefix}/loss"] = loss
        return loss, loss_dict

    def p_sample_loop(self, condition, shape, *args, c_edit_crossattn=None, **kwargs):
        if c_edit_crossattn is not None:
            condition = dict(condition)
            condition["c_crossattn"] = list(condition["c_crossattn"]) + [c_edit_crossattn]
        return super().p_sample_loop(condition, shape, *args, **kwargs)
