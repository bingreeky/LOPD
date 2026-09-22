from __future__ import annotations

import copy
from typing import Any

import torch
from transformers import PreTrainedTokenizer

from memory.prompt_protocol import CompressorPromptProtocol

LATENT_PLACEHOLDER = "<|LATENT_PH|>"
IGNORE_INDEX = -100


def chat_template_kwargs(enable_thinking: bool | None, *, add_generation_prompt: bool = False) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": add_generation_prompt}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return kwargs


def serialize_trajectory(trajectory: list[dict] | dict[str, Any], tokenizer: PreTrainedTokenizer, *, enable_thinking: bool | None) -> str:
    if isinstance(trajectory, dict) and "raw_text" in trajectory:
        return str(trajectory["raw_text"])
    return tokenizer.apply_chat_template(trajectory, **chat_template_kwargs(enable_thinking))


def dummy_token_id(tokenizer: PreTrainedTokenizer) -> int:
    for attr in ("pad_token_id", "eos_token_id", "unk_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    raise ValueError("Tokenizer has no pad/eos/unk/bos token id for latent placeholder")


def build_latent_injected_ids(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    n_latent: int,
    protocol: CompressorPromptProtocol,
    *,
    enable_thinking: bool | None,
    add_generation_prompt: bool = False,
) -> tuple[list[int], list[int]]:
    if n_latent <= 0:
        raise ValueError("n_latent must be positive")
    modified = copy.deepcopy(messages)
    first_user_idx = next((i for i, msg in enumerate(modified) if msg.get("role") == "user"), None)
    if first_user_idx is None:
        raise ValueError("messages contains no user role")
    modified[first_user_idx]["content"] = (
        protocol.latent_framing_before + LATENT_PLACEHOLDER + protocol.latent_framing_after
        + str(modified[first_user_idx].get("content", ""))
    )
    prompt_text = tokenizer.apply_chat_template(
        modified, **chat_template_kwargs(enable_thinking, add_generation_prompt=add_generation_prompt),
    )
    if LATENT_PLACEHOLDER not in prompt_text:
        raise RuntimeError(f"Placeholder {LATENT_PLACEHOLDER!r} not found in chat template output")
    before_text, after_text = prompt_text.split(LATENT_PLACEHOLDER, 1)
    before_ids = tokenizer.encode(before_text, add_special_tokens=False)
    after_ids = tokenizer.encode(after_text, add_special_tokens=False)
    input_ids = before_ids + [dummy_token_id(tokenizer)] * n_latent + after_ids
    positions = list(range(len(before_ids), len(before_ids) + n_latent))
    return input_ids, positions


def build_supervised_inputs(
    messages: list[dict],
    tokenizer: PreTrainedTokenizer,
    *,
    enable_thinking: bool | None,
    max_seq_len: int,
    n_latent: int = 0,
    protocol: CompressorPromptProtocol | None = None,
) -> tuple[list[int], list[int], torch.Tensor]:
    def render(prefix: list[dict], add_generation_prompt: bool) -> tuple[list[int], list[int]]:
        if n_latent:
            return build_latent_injected_ids(
                prefix, tokenizer, n_latent, protocol,
                enable_thinking=enable_thinking, add_generation_prompt=add_generation_prompt,
            )
        text = tokenizer.apply_chat_template(prefix, **chat_template_kwargs(enable_thinking, add_generation_prompt=add_generation_prompt))
        return tokenizer.encode(text, add_special_tokens=False), []

    input_ids, positions = render(messages, False)
    target = torch.full((len(input_ids),), IGNORE_INDEX, dtype=torch.long)
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        start = len(render(messages[:i], True)[0])
        end = min(len(render(messages[: i + 1], False)[0]), len(input_ids))
        if end > start:
            target[start:end] = torch.tensor(input_ids[start:end], dtype=torch.long)

    if len(input_ids) > max_seq_len:
        if positions and positions[-1] >= max_seq_len:
            raise ValueError(f"latent span ends at {positions[-1]} but max_seq_len={max_seq_len}")
        input_ids, target = input_ids[:max_seq_len], target[:max_seq_len]
    return input_ids, positions, target


def inject_latent_tokens_into_embeds(
    prompt_embeds: torch.Tensor,
    latent_positions: list[int],
    latent_tokens: torch.Tensor,
) -> torch.Tensor:
    if not latent_positions:
        return prompt_embeds
    start = latent_positions[0]
    end = latent_positions[-1] + 1
    if latent_positions != list(range(start, end)):
        raise ValueError("latent_positions must form one contiguous span")
    latent_tokens = latent_tokens.to(device=prompt_embeds.device, dtype=prompt_embeds.dtype)
    return torch.cat([prompt_embeds[:, :start, :], latent_tokens.unsqueeze(0), prompt_embeds[:, end:, :]], dim=1)
