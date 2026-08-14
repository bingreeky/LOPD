
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from memory.qformer import QFormer


def build_compressor_from_model_config(
    model_path: str,
    qformer_cfg: dict[str, Any] | None,
) -> QFormer:
    cfg = dict(qformer_cfg or {})
    dim, heads, dim_head = _load_model_dimensions(model_path)
    return QFormer(
        dim=dim,
        depth=int(cfg.get("depth", 8)),
        dim_head=dim_head,
        heads=heads,
        num_queries=int(cfg.get("num_queries", 8)),
        ff_mult=int(cfg.get("ff_mult", 4)),
        share_layers=bool(cfg.get("share_layers", True)),
    )


def forward_compressor(
    compressor: QFormer,
    *,
    memory_hidden: torch.Tensor,
    task_hidden: torch.Tensor | None = None,
    memory_mask: torch.Tensor | None = None,
    task_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    return compressor(memory_hidden, mask=memory_mask)


def save_compressor_checkpoint(
    compressor: QFormer,
    path: str | os.PathLike[str],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    dataset_offset: int = 0,
) -> None:
    compressor.save_checkpoint(
        str(path),
        optimizer=optimizer,
        step=step,
        dataset_offset=dataset_offset,
    )


def load_compressor_checkpoint(
    path: str | os.PathLike[str],
    device: str | torch.device = "cpu",
    *,
    expected_architecture: str | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[QFormer, int, int]:
    checkpoint_path = _resolve_checkpoint_path(path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid compressor checkpoint: {checkpoint_path}")

    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("compressor checkpoint is missing a dict 'config'")

    state_dict = checkpoint.get("qformer_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("compressor checkpoint is missing 'qformer_state_dict'")

    compressor = QFormer(**config)
    compressor.load_state_dict(state_dict, strict=True)
    compressor = compressor.to(device)

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        compressor._loaded_optimizer_state_dict = optimizer_state
        if optimizer is not None:
            optimizer.load_state_dict(optimizer_state)
            compressor._loaded_optimizer_state_dict = None

    return (
        compressor,
        int(checkpoint.get("step", 0)),
        int(checkpoint.get("dataset_offset", 0)),
    )


def _load_model_dimensions(model_path: str) -> tuple[int, int, int]:
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    text_config = getattr(model_config, "text_config", model_config)
    dim = int(text_config.hidden_size)
    heads = int(text_config.num_attention_heads)
    if dim % heads != 0:
        raise ValueError(
            f"model hidden_size={dim} is not divisible by num_attention_heads={heads}",
        )
    return dim, heads, dim // heads


def _resolve_checkpoint_path(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "qformer.pt"
    if not candidate.is_file():
        raise FileNotFoundError(f"compressor checkpoint not found: {candidate}")
    return candidate
