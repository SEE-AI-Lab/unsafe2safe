"""Safe Cross Attention adapters for the external InstructPix2Pix UNet.

The base checkout is imported at runtime. No files inside that checkout are
modified: the Unsafe2Safe model replaces the UNet's transformer class in
memory before the external config instantiates it.
"""

from __future__ import annotations

import torch
from torch import einsum, nn
from einops import rearrange, repeat

from .instruct_pix2pix import configure_external

configure_external()

from ldm.modules import attention as base_attention  # noqa: E402
from ldm.modules.diffusionmodules.util import checkpoint  # noqa: E402


class SafeCrossAttention(nn.Module):
    """Cross attention with public-caption and edit-instruction contexts.

    ``to_q``, ``to_k``, ``to_v``, and ``to_out`` retain the upstream names so
    an InstructPix2Pix/MagicBrush checkpoint can initialize the edit branch.
    The public branch is an auxiliary residual path initialized at zero.
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim
        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_public = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_public = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )
        self.map_fuse = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.public_scale = nn.Parameter(torch.zeros(1))

    def _standard(self, x, context, mask=None):
        h = self.heads
        q = self.to_q(x)
        k, v = self.to_k(context), self.to_v(context)
        q, k, v = map(
            lambda tensor: rearrange(tensor, "b n (h d) -> (b h) n d", h=h),
            (q, k, v),
        )
        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale
        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)
        out = einsum("b i j, b j d -> b i d", sim.softmax(dim=-1), v)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)

    def forward(self, x, context=None, context_edit=None, context_public=None, mask=None):
        if context_edit is None:
            context_edit = x if context is None else context
        if context_public is None:
            return self._standard(x, context_edit, mask=mask)

        h = self.heads
        q = self.to_q(x)
        k_edit, v_edit = self.to_k(context_edit), self.to_v(context_edit)
        k_public = self.to_k_public(context_public)
        v_public = self.to_v_public(context_public)
        q, k_edit, v_edit, k_public, v_public = map(
            lambda tensor: rearrange(tensor, "b n (h d) -> (b h) n d", h=h),
            (q, k_edit, v_edit, k_public, v_public),
        )

        sim_edit = einsum("b i d, b j d -> b i j", q, k_edit) * self.scale
        sim_public = einsum("b i d, b j d -> b i j", q, k_public) * self.scale
        if mask is not None:
            mask = rearrange(mask, "b ... -> b (...)")
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            fill_value = -torch.finfo(sim_edit.dtype).max
            sim_edit.masked_fill_(~mask, fill_value)
            sim_public.masked_fill_(~mask, fill_value)

        attn_edit = sim_edit.softmax(dim=-1)
        attn_public = sim_public.softmax(dim=-1)
        map_features = torch.stack(
            (attn_edit.amax(dim=-1), attn_public.amax(dim=-1)), dim=-1
        )
        public_gate = torch.sigmoid(self.map_fuse(map_features))
        attn_public = attn_public * (1.0 + public_gate)
        attn_public = attn_public / (attn_public.sum(dim=-1, keepdim=True) + 1e-6)

        out_edit = einsum("b i j, b j d -> b i d", attn_edit, v_edit)
        out_public = einsum("b i j, b j d -> b i d", attn_public, v_public)
        out = out_edit + self.public_scale * out_public
        out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return self.to_out(out)


class SafeBasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0.0, context_dim=None, gated_ff=True, checkpoint_enabled=True):
        super().__init__()
        self.attn1 = base_attention.CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.attn2 = SafeCrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
        )
        self.ff = base_attention.FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint_enabled

    def forward(self, x, context=None):
        # The legacy external checkpoint helper expects tensor inputs only.
        if isinstance(context, (tuple, list)):
            return self._forward(x, context)
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        if isinstance(context, (tuple, list)):
            if len(context) != 2:
                raise ValueError("Safe attention context must be (public, edit)")
            context_public, context_edit = context
            x = self.attn2(
                self.norm2(x),
                context_edit=context_edit,
                context_public=context_public,
            ) + x
        else:
            x = self.attn2(self.norm2(x), context=context) + x
        return self.ff(self.norm3(x)) + x


class SafeSpatialTransformer(nn.Module):
    """Drop-in replacement for the external SpatialTransformer."""

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0.0, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = base_attention.Normalize(in_channels)
        self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList(
            [
                SafeBasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                )
                for _ in range(depth)
            ]
        )
        self.proj_out = base_attention.zero_module(
            nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, context=None):
        _, _, height, width = x.shape
        x_in = x
        x = self.proj_in(self.norm(x))
        x = rearrange(x, "b c h w -> b (h w) c")
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, "b (h w) c -> b c h w", h=height, w=width)
        return self.proj_out(x) + x_in
