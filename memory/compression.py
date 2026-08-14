
from __future__ import annotations

import hashlib

import torch

from memory.compressor import forward_compressor
from memory.prompt_protocol import (
    CompressorPromptProtocol,
    resolve_compressor_prompt_protocol,
)


def compress_texts_via_pytorch(
    model,
    compressor: torch.nn.Module,
    texts: tuple[str, ...],
    tokenizer,
    device: str,
    *,
    train_encoder_lora: bool,
    task_text: str | None = None,
    prompt_protocol: str | CompressorPromptProtocol | None = None,
) -> torch.Tensor:
    protocol = resolve_compressor_prompt_protocol(prompt_protocol)

    all_latents = []
    for text in texts:
        if task_text is not None and task_text.strip():
            task_prefix = (
                protocol.task_prefix_head
                + task_text.rstrip()
                + protocol.task_prefix_tail
            )
            task_prefix_ids = tokenizer(
                task_prefix,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(device)
            trajectory_ids = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(device)
            input_ids = torch.cat([task_prefix_ids, trajectory_ids], dim=1)
        else:
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        hidden = _encode_hidden(model, input_ids, train_encoder_lora=train_encoder_lora)
        hidden = hidden.to(next(compressor.parameters()).device)
        latent = compressor(hidden)
        all_latents.append(latent.squeeze(0))

    if not all_latents:
        num_queries = compressor.num_queries
        hidden_size = compressor.proj_in.in_features
        parameter = next(compressor.parameters())
        return torch.zeros(
            num_queries, hidden_size,
            device=parameter.device, dtype=parameter.dtype, requires_grad=True,
        )
    return torch.cat(all_latents, dim=0)


def _encode_hidden(
    model,
    input_ids: torch.Tensor,
    *,
    train_encoder_lora: bool,
) -> torch.Tensor:
    grad_context = torch.enable_grad() if train_encoder_lora else torch.no_grad()
    with grad_context:
        return model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        ).hidden_states[-1]


def trajectory_text(entry: dict, tokenizer, enable_thinking: bool = True) -> str:
    from memory.serialization import serialize_trajectory

    return serialize_trajectory(
        entry["trajectory"], entry.get("tools"), tokenizer,
        enable_thinking=enable_thinking,
    )


def latent_cache_key(text: str, task_text: str | None = None) -> str:
    if task_text is None:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()
    h = hashlib.sha1()
    h.update(b"task:")
    h.update(task_text.encode("utf-8"))
    h.update(b"\x00traj:")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def compress_hidden_to_latent(
    hidden: torch.Tensor,
    qformer: torch.nn.Module,
    qformer_device,
    *,
    task_hidden: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if hidden.dim() != 2 or hidden.shape[0] == 0 or hidden.shape[1] == 0:
        return None
    with torch.no_grad():
        parameter = next(qformer.parameters())
        h = hidden.unsqueeze(0).to(qformer_device, dtype=parameter.dtype)
        latent = forward_compressor(
            qformer,
            memory_hidden=h,
        ).squeeze(0).detach().cpu()
    return latent
