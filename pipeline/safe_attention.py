"""Safe Attention layers used by the Unsafe2Safe editor."""

import torch
import torch.nn as nn
from einops import rearrange


class PrivateCrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.scale = dim_head**-0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k_priv = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_k_pub = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_pub = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_priv = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim * 2, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )
        self.map_fuse = nn.Sequential(
            nn.LayerNorm(78),
            nn.Linear(78, 128),
            nn.GELU(),
            nn.Linear(128, 77),
        )

    def forward(self, x, context_pub=None, context_priv=None, mask=None):
        q = self.to_q(x)
        context_pub = rearrange(context_pub, "b d -> b 1 d")
        k_priv = self.to_k_priv(context_priv)
        k_pub = self.to_k_pub(context_pub)
        v_pub = self.to_v_pub(context_pub)
        v_priv = self.to_v_priv(context_priv)
        q, k_pub, k_priv, v_pub, v_priv = map(
            lambda tensor: rearrange(tensor, "b n (h d) -> (b h) n d", h=self.heads),
            (q, k_pub, k_priv, v_pub, v_priv),
        )

        sim_pub = torch.einsum("b i d, b j d -> b i j", q, k_pub) * self.scale
        sim_priv = torch.einsum("b i d, b j d -> b i j", q, k_priv) * self.scale
        attn_pub = sim_pub.softmax(dim=-1)
        attn_priv = sim_priv.softmax(dim=-1)
        out_pub = torch.einsum("b i j, b j d -> b i d", attn_pub, v_pub)

        attn_concat = torch.cat([attn_pub, attn_priv], dim=-1)
        attn_concat = attn_concat / (attn_concat.sum(dim=-1, keepdim=True) + 1e-8)
        attn_priv = self.map_fuse(attn_concat).softmax(dim=-1)
        out_priv = torch.einsum("b i j, b j d -> b i d", attn_priv, v_priv)

        out = torch.cat([out_priv, out_pub], dim=-1)
        out = rearrange(out, "(b h) n d -> b n (h d)", h=self.heads)
        return self.to_out(out)


class PrivFilter(nn.Module):
    def __init__(self, dim: int, priv_dim: int, heads: int = 8, dim_head: int = 64, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(priv_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, c_priv):
        batch_size, tokens, _ = x.shape
        _, private_tokens, _ = c_priv.shape
        q = self.to_q(c_priv)
        k = self.to_k(x)
        v = self.to_v(x)
        q, k, v = [tensor.view(batch_size, -1, self.heads, self.dim_head).transpose(1, 2) for tensor in (q, k, v)]
        sim = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn_priv = torch.softmax(sim, dim=-1)
        priv_out = torch.einsum("bhij,bhjd->bhid", attn_priv, v)
        priv_out = priv_out.transpose(1, 2).contiguous().view(batch_size, private_tokens, -1)
        priv_out = self.to_out(priv_out)
        x_filtered = x - priv_out.mean(dim=1, keepdim=True)
        return x_filtered, attn_priv
