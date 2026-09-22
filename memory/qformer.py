import math
import os

import torch
import torch.nn as nn


def feed_forward(dim: int, mult: int) -> nn.Sequential:
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):

    def __init__(self, *, dim: int, dim_head: int, heads: int):
        super().__init__()
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, n, _ = x.shape
        _, m, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat([x, latents], dim=1)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = q.view(b, m, self.heads, self.dim_head).transpose(1, 2)
        k = k.view(b, n + m, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(b, n + m, self.heads, self.dim_head).transpose(1, 2)

        scale = 1.0 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.transpose(1, 2).reshape(b, m, -1)
        return self.to_out(out)


class QFormer(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int,
        dim_head: int,
        heads: int,
        num_queries: int,
        ff_mult: int,
        share_layers: bool,
    ):
        super().__init__()
        self.config = {
            "dim": dim, "depth": depth, "dim_head": dim_head, "heads": heads,
            "num_queries": num_queries, "ff_mult": ff_mult, "share_layers": share_layers,
        }
        self.num_queries = num_queries
        self.ff_mult = ff_mult
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)
        self.depth = depth
        self.shared = share_layers

        if share_layers:
            self.layers = nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                feed_forward(dim, ff_mult),
            ])
        else:
            self.layers = nn.ModuleList()
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    feed_forward(dim, ff_mult),
                ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latents = self.latents.expand(x.size(0), -1, -1)
        x = self.proj_in(x)

        if self.shared:
            attn, ff = self.layers
            for _ in range(self.depth):
                latents = attn(x, latents) + latents
                latents = ff(latents) + latents
        else:
            for attn, ff in self.layers:
                latents = attn(x, latents) + latents
                latents = ff(latents) + latents

        return self.norm_out(self.proj_out(latents))

    def save_checkpoint(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"qformer_state_dict": self.state_dict(), "config": dict(self.config)}, path)
