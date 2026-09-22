from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from memory.lora import resolve_peft_adapter_dir, save_lora_adapter
from memory.qformer import QFormer

QFORMER_FILE = "qformer.pt"
ADAPTER_DIR = "lora_adapter"
MANIFEST_FILE = "manifest.json"
QFORMER_OPTIMIZER_FILE = "qformer_optimizer.pt"
LORA_OPTIMIZER_FILE = "lora_optimizer.pt"


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
        num_queries=int(cfg.get("num_queries", 32)),
        ff_mult=int(cfg.get("ff_mult", 4)),
        share_layers=bool(cfg.get("share_layers", True)),
    )


def load_compressor_checkpoint(
    path: str | os.PathLike[str],
    device: str | torch.device = "cpu",
) -> QFormer:
    checkpoint_path = _resolve_qformer_file(path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"invalid compressor checkpoint: {checkpoint_path}")

    config = checkpoint.get("config")
    state_dict = checkpoint.get("qformer_state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError(
            f"compressor checkpoint is missing 'config' or 'qformer_state_dict': {checkpoint_path}"
        )

    compressor = QFormer(**config)
    compressor.load_state_dict(state_dict, strict=True)
    return compressor.to(device)


def save_compressor(
    ckpt_dir: str | os.PathLike[str],
    compressor: QFormer,
    peft_model,
    *,
    step: int,
    manifest: dict[str, Any],
) -> None:
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    compressor.save_checkpoint(str(ckpt_dir / QFORMER_FILE))
    save_lora_adapter(peft_model, str(ckpt_dir / ADAPTER_DIR), manifest["adapter_name"])
    with open(ckpt_dir / MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump({"step": step, **manifest}, f, indent=2, ensure_ascii=False)


def compressor_fingerprint(ckpt_dir: str | os.PathLike[str], model_path: str | os.PathLike[str]) -> str:
    ckpt_dir, model_dir = Path(ckpt_dir), Path(model_path)
    with open(ckpt_dir / MANIFEST_FILE, encoding="utf-8") as f:
        adapter_name = json.load(f)["adapter_name"]
    adapter_dir = Path(resolve_peft_adapter_dir(str(ckpt_dir / ADAPTER_DIR), adapter_name))
    digest = hashlib.sha256()
    for path in (ckpt_dir / QFORMER_FILE, adapter_dir / "adapter_config.json", adapter_dir / "adapter_model.safetensors", model_dir / "config.json"):
        digest.update(path.read_bytes())
    for shard in sorted(model_dir.glob("*.safetensors")):
        digest.update(f"{shard.name}:{shard.stat().st_size}".encode())
        with open(shard, "rb") as f:
            digest.update(f.read(1 << 20))
    return digest.hexdigest()[:16]


def load_compressor(
    ckpt_dir: str | os.PathLike[str],
    device: str | torch.device = "cpu",
) -> tuple[QFormer, str, dict[str, Any]]:
    ckpt_dir = Path(ckpt_dir)
    with open(ckpt_dir / MANIFEST_FILE, encoding="utf-8") as f:
        manifest = json.load(f)
    compressor = load_compressor_checkpoint(ckpt_dir / QFORMER_FILE, device=device)
    adapter_dir = resolve_peft_adapter_dir(str(ckpt_dir / ADAPTER_DIR), manifest["adapter_name"])
    return compressor, adapter_dir, manifest


@dataclass
class CompressorTrainState:
    qformer_optimizer: dict[str, Any]
    lora_optimizer: dict[str, dict]
    epoch: int
    position: int
    seed: int


def save_train_state(
    ckpt_dir: str | os.PathLike[str],
    *,
    qformer_optimizer: torch.optim.Optimizer,
    lora_optimizer_state: dict[str, dict],
    epoch: int,
    position: int,
    seed: int,
) -> None:
    ckpt_dir = Path(ckpt_dir)
    torch.save(
        {
            "optimizer_state_dict": qformer_optimizer.state_dict(),
            "epoch": epoch, "position": position, "seed": seed,
        },
        ckpt_dir / QFORMER_OPTIMIZER_FILE,
    )
    torch.save({"optimizer_state_by_name": lora_optimizer_state}, ckpt_dir / LORA_OPTIMIZER_FILE)


def load_train_state(ckpt_dir: str | os.PathLike[str]) -> CompressorTrainState:
    ckpt_dir = Path(ckpt_dir)
    qformer_path = ckpt_dir / QFORMER_OPTIMIZER_FILE
    lora_path = ckpt_dir / LORA_OPTIMIZER_FILE
    if not qformer_path.is_file() or not lora_path.is_file():
        raise FileNotFoundError(
            f"{ckpt_dir} holds compressor weights only; "
            f"{QFORMER_OPTIMIZER_FILE} and {LORA_OPTIMIZER_FILE} are required to resume training"
        )
    qformer_state = torch.load(qformer_path, map_location="cpu", weights_only=True)
    lora_state = torch.load(lora_path, map_location="cpu", weights_only=True)
    return CompressorTrainState(
        qformer_optimizer=qformer_state["optimizer_state_dict"],
        lora_optimizer=lora_state["optimizer_state_by_name"],
        epoch=int(qformer_state.get("epoch", 0)),
        position=int(qformer_state.get("position", 0)),
        seed=int(qformer_state["seed"]),
    )


def _load_model_dimensions(model_path: str) -> tuple[int, int, int]:
    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    dim = int(model_config.hidden_size)
    heads = int(model_config.num_attention_heads)
    if dim % heads != 0:
        raise ValueError(
            f"model hidden_size={dim} is not divisible by num_attention_heads={heads}",
        )
    return dim, heads, dim // heads


def _resolve_qformer_file(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / QFORMER_FILE
    if not candidate.is_file():
        raise FileNotFoundError(f"compressor checkpoint not found: {candidate}")
    return candidate
