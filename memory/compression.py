from __future__ import annotations

import hashlib

import torch

from memory.prompt_protocol import CompressorPromptProtocol
from memory.serialization import serialize_trajectory


def compress_texts_via_pytorch(
    model,
    compressor: torch.nn.Module,
    texts: tuple[str, ...],
    tokenizer,
    device: str,
    *,
    train_encoder_lora: bool,
    task_text: str | None,
    protocol: CompressorPromptProtocol,
) -> torch.Tensor:
    hiddens = encode_texts_via_pytorch(
        model, texts, tokenizer, device,
        train_encoder_lora=train_encoder_lora, task_text=task_text, protocol=protocol,
    )
    return compress_hiddens(compressor, hiddens)


def compress_hiddens(compressor: torch.nn.Module, hiddens: list[torch.Tensor]) -> torch.Tensor:
    device = next(compressor.parameters()).device
    return torch.cat([compressor(hidden.to(device)).squeeze(0) for hidden in hiddens], dim=0)


def encode_texts_via_pytorch(
    model,
    texts: tuple[str, ...],
    tokenizer,
    device: str,
    *,
    train_encoder_lora: bool,
    task_text: str | None,
    protocol: CompressorPromptProtocol,
) -> list[torch.Tensor]:
    if not texts:
        raise ValueError("encode_texts_via_pytorch requires at least one text")

    hiddens = []
    for text in texts:
        if task_text is not None and task_text.strip():
            task_prefix_ids = tokenizer(protocol.task_prefix(task_text), return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            trajectory_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            input_ids = torch.cat([task_prefix_ids, trajectory_ids], dim=1)
        else:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        hiddens.append(_encode_hidden(model, input_ids, train_encoder_lora=train_encoder_lora))
    return hiddens


def _encode_hidden(model, input_ids: torch.Tensor, *, train_encoder_lora: bool) -> torch.Tensor:
    grad_context = torch.enable_grad() if train_encoder_lora else torch.no_grad()
    with grad_context:
        return model(input_ids=input_ids, output_hidden_states=True, use_cache=False).hidden_states[-1]


def trajectory_text(entry: dict, tokenizer, *, enable_thinking: bool | None) -> str:
    return serialize_trajectory(entry["trajectory"], tokenizer, enable_thinking=enable_thinking)


def latent_cache_key(text: str, task_text: str | None = None) -> str:
    if task_text is None:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()
    h = hashlib.sha1()
    h.update(b"task:")
    h.update(task_text.encode("utf-8"))
    h.update(b"\x00traj:")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def compress_hidden_to_latent(hidden: torch.Tensor, qformer: torch.nn.Module, qformer_device) -> torch.Tensor | None:
    if hidden.dim() != 2 or hidden.shape[0] == 0 or hidden.shape[1] == 0:
        return None
    with torch.no_grad():
        parameter = next(qformer.parameters())
        h = hidden.unsqueeze(0).to(qformer_device, dtype=parameter.dtype)
        return qformer(h).squeeze(0).detach().cpu()
