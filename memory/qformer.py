
import math

import torch
import torch.nn as nn


def FeedForward(dim: int, mult: int = 4) -> nn.Sequential:
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


class PerceiverAttention(nn.Module):

    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        if mask is not None:
            full_mask = torch.ones(b, n + m, device=mask.device, dtype=torch.bool)
            full_mask[:, :n] = mask
            full_mask = full_mask.unsqueeze(1).unsqueeze(2)
            weight = weight.masked_fill(~full_mask, -torch.finfo(weight.dtype).max)

        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v
        out = out.transpose(1, 2).reshape(b, m, -1)
        return self.to_out(out)


class QFormer(nn.Module):

    def __init__(
        self,
        dim: int = 2560,
        depth: int = 8,
        dim_head: int = 80,
        heads: int = 32,
        num_queries: int = 8,
        ff_mult: int = 4,
        share_layers: bool = True,
    ):
        super().__init__()
        self.num_queries = num_queries
        self._ff_mult = ff_mult
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim ** 0.5)
        self.proj_in = nn.Linear(dim, dim)
        self.proj_out = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)

        if share_layers:
            self.layers = nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            ])
            self.depth = depth
            self.shared = True
        else:
            self.layers = nn.ModuleList()
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                    FeedForward(dim=dim, mult=ff_mult),
                ]))
            self.depth = depth
            self.shared = False

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latents = self.latents.expand(x.size(0), -1, -1)
        x = self.proj_in(x)

        if self.shared:
            attn, ff = self.layers
            for _ in range(self.depth):
                latents = attn(x, latents, mask) + latents
                latents = ff(latents) + latents
        else:
            for attn, ff in self.layers:
                latents = attn(x, latents, mask) + latents
                latents = ff(latents) + latents

        return self.norm_out(self.proj_out(latents))


    def save_checkpoint(self, path: str, optimizer=None, step: int = 0, dataset_offset: int = 0) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {
            "qformer_state_dict": self.state_dict(),
            "config": {
                "dim": self.proj_in.in_features,
                "depth": self.depth,
                "dim_head": self.layers[0].dim_head if self.shared else self.layers[0][0].dim_head,
                "heads": self.layers[0].heads if self.shared else self.layers[0][0].heads,
                "num_queries": self.num_queries,
                "ff_mult": self._ff_mult,
                "share_layers": self.shared,
            },
            "step": step,
            "dataset_offset": dataset_offset,
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu", optimizer=None):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = ckpt["config"]
        qformer = cls(**config)
        qformer.load_state_dict(ckpt["qformer_state_dict"])
        qformer = qformer.to(device)
        if "optimizer_state_dict" in ckpt:
            qformer._loaded_optimizer_state_dict = ckpt["optimizer_state_dict"]
        step = ckpt.get("step", 0)
        dataset_offset = ckpt.get("dataset_offset", 0)
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            qformer._loaded_optimizer_state_dict = None
        return qformer, step, dataset_offset


    @classmethod
    def from_model_config(
        cls,
        model_path: str,
        depth: int = 8,
        num_queries: int = 8,
        ff_mult: int = 4,
        share_layers: bool = True,
    ):
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        text_cfg = getattr(cfg, "text_config", cfg)
        dim = text_cfg.hidden_size
        heads = text_cfg.num_attention_heads
        dim_head = dim // heads
        return cls(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_queries=num_queries,
            ff_mult=ff_mult,
            share_layers=share_layers,
        )
