
import copy
from typing import Any, Optional

import torch
from transformers import PreTrainedTokenizer

from envs.envscaler.prompts import (  # noqa: F401
    MEMORY_FRAMING_BEFORE as DEFAULT_LATENT_FRAMING_BEFORE,
    MEMORY_FRAMING_AFTER as DEFAULT_LATENT_FRAMING_AFTER,
)
DEFAULT_LATENT_PLACEHOLDER = "<|LATENT_PH|>"


def serialize_trajectory(
    messages: list[dict] | dict[str, Any] | str,
    tools: Optional[list[dict]],
    tokenizer: PreTrainedTokenizer,
    enable_thinking: bool = False,
) -> str:
    if isinstance(messages, dict) and "raw_text" in messages:
        return str(messages["raw_text"])
    if isinstance(messages, str):
        return messages

    kwargs = {
        "tools": tools or None,
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    return tokenizer.apply_chat_template(
        messages,
        **kwargs,
    )


def dummy_token_id(tokenizer: PreTrainedTokenizer) -> int:
    for attr in ("pad_token_id", "eos_token_id", "unk_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attr, None)
        if token_id is not None:
            return int(token_id)
    raise ValueError("Tokenizer has no pad/eos/unk/bos token id for latent placeholder")


def build_latent_injected_ids(
    messages: list[dict],
    tools: Optional[list[dict]],
    tokenizer: PreTrainedTokenizer,
    n_latent: int,
    *,
    enable_thinking: bool | None = False,
    add_generation_prompt: bool = False,
    framing_before: str = DEFAULT_LATENT_FRAMING_BEFORE,
    framing_after: str = DEFAULT_LATENT_FRAMING_AFTER,
    placeholder: str = DEFAULT_LATENT_PLACEHOLDER,
    placeholder_token_id: int | None = None,
) -> tuple[list[int], list[int], str, str]:
    if n_latent < 0:
        raise ValueError("n_latent must be >= 0")
    if placeholder_token_id is None:
        placeholder_token_id = dummy_token_id(tokenizer)

    modified = copy.deepcopy(messages)
    first_user_idx = next(
        (i for i, msg in enumerate(modified) if msg.get("role") == "user"),
        None,
    )
    if first_user_idx is None:
        raise ValueError("messages contains no user role")

    original_content = modified[first_user_idx].get("content", "")
    modified[first_user_idx]["content"] = (
        framing_before + placeholder + framing_after + str(original_content)
    )

    kwargs = {
        "tools": tools or None,
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    prompt_text = tokenizer.apply_chat_template(modified, **kwargs)

    if placeholder not in prompt_text:
        raise RuntimeError(f"Placeholder {placeholder!r} not found in chat template output")
    before_text, after_text = prompt_text.split(placeholder, 1)

    before_ids = tokenizer.encode(before_text, add_special_tokens=False)
    after_ids = tokenizer.encode(after_text, add_special_tokens=False)
    input_ids = before_ids + [placeholder_token_id] * n_latent + after_ids
    positions = list(range(len(before_ids), len(before_ids) + n_latent))
    return input_ids, positions, before_text, after_text


def build_latent_injected_target_ids(
    messages: list[dict],
    tools: Optional[list[dict]],
    tokenizer: PreTrainedTokenizer,
    n_latent: int,
    *,
    enable_thinking: bool | None = False,
    target_scope: str = "all_assistant",
    framing_before: str = DEFAULT_LATENT_FRAMING_BEFORE,
    framing_after: str = DEFAULT_LATENT_FRAMING_AFTER,
    placeholder: str = DEFAULT_LATENT_PLACEHOLDER,
    placeholder_token_id: int | None = None,
) -> torch.Tensor:
    IGNORE_INDEX = -100
    if target_scope not in {"all_assistant", "last_assistant"}:
        raise ValueError(
            "build_latent_injected_target_ids currently supports "
            f"'all_assistant' and 'last_assistant', got: {target_scope}"
        )

    input_ids, positions, _, _ = build_latent_injected_ids(
        messages,
        tools,
        tokenizer,
        n_latent,
        enable_thinking=enable_thinking,
        add_generation_prompt=False,
        framing_before=framing_before,
        framing_after=framing_after,
        placeholder=placeholder,
        placeholder_token_id=placeholder_token_id,
    )
    target = torch.full((len(input_ids),), IGNORE_INDEX, dtype=torch.long)

    assistant_ranges: list[tuple[int, int]] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        prefix_ids, _, _, _ = build_latent_injected_ids(
            messages[:i],
            tools,
            tokenizer,
            n_latent,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
            framing_before=framing_before,
            framing_after=framing_after,
            placeholder=placeholder,
            placeholder_token_id=placeholder_token_id,
        )
        include_ids, _, _, _ = build_latent_injected_ids(
            messages[: i + 1],
            tools,
            tokenizer,
            n_latent,
            enable_thinking=enable_thinking,
            add_generation_prompt=False,
            framing_before=framing_before,
            framing_after=framing_after,
            placeholder=placeholder,
            placeholder_token_id=placeholder_token_id,
        )
        if len(include_ids) > len(prefix_ids):
            assistant_ranges.append((len(prefix_ids), len(include_ids)))

    if target_scope == "last_assistant":
        assistant_ranges = assistant_ranges[-1:]

    for start, end in assistant_ranges:
        end = min(end, len(input_ids))
        target[start:end] = torch.tensor(input_ids[start:end], dtype=torch.long)

    for pos in positions:
        if 0 <= pos < target.shape[0]:
            target[pos] = IGNORE_INDEX

    return target.unsqueeze(0)


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
    return torch.cat(
        [
            prompt_embeds[:, :start, :],
            latent_tokens.unsqueeze(0),
            prompt_embeds[:, end:, :],
        ],
        dim=1,
    )


def build_target_ids(
    messages: list[dict],
    tools: Optional[list[dict]],
    tokenizer: PreTrainedTokenizer,
    latent_prefix_len: int = 0,
    enable_thinking: bool = False,
    target_scope: str = "all_assistant",
) -> torch.Tensor:
    IGNORE_INDEX = -100

    full_text = serialize_trajectory(messages, tools, tokenizer, enable_thinking)
    full_ids = tokenizer(full_text, return_tensors="pt").input_ids[0]
    seq_len = full_ids.shape[0]

    target = torch.full((seq_len,), IGNORE_INDEX, dtype=torch.long)

    if target_scope == "assistant_after_think":
        assistant_ranges = _find_assistant_after_think_token_ranges(
            messages, tools, tokenizer, enable_thinking, full_text,
        )
    else:
        assistant_ranges = _find_assistant_token_ranges(
            messages, tools, tokenizer, enable_thinking,
        )

    if target_scope == "last_assistant":
        assistant_ranges = assistant_ranges[-1:]
    elif target_scope not in {"all_assistant", "assistant_after_think"}:
        raise ValueError(f"Unsupported target_scope: {target_scope}")

    for start, end in assistant_ranges:
        end = min(end, seq_len)
        target[start:end] = full_ids[start:end]

    if latent_prefix_len > 0:
        latent_ignore = torch.full((latent_prefix_len,), IGNORE_INDEX, dtype=torch.long)
        target = torch.cat([latent_ignore, target])

    return target.unsqueeze(0)


def _find_assistant_token_ranges(
    messages: list[dict],
    tools: Optional[list[dict]],
    tokenizer: PreTrainedTokenizer,
    enable_thinking: bool,
) -> list[tuple[int, int]]:
    ranges = []
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        prefix_messages = messages[:i]
        include_messages = messages[: i + 1]

        prefix_kwargs = {
            "tools": tools or None,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        include_kwargs = {
            "tools": tools or None,
            "tokenize": False,
            "add_generation_prompt": False,
        }
        if enable_thinking is not None:
            prefix_kwargs["enable_thinking"] = enable_thinking
            include_kwargs["enable_thinking"] = enable_thinking

        prefix_text = tokenizer.apply_chat_template(
            prefix_messages,
            **prefix_kwargs,
        )
        include_text = tokenizer.apply_chat_template(
            include_messages,
            **include_kwargs,
        )

        prefix_len = len(tokenizer(prefix_text).input_ids)
        include_len = len(tokenizer(include_text).input_ids)

        if include_len > prefix_len:
            ranges.append((prefix_len, include_len))

    return ranges


def _find_assistant_after_think_token_ranges(
    messages: list[dict],
    tools: Optional[list[dict]],
    tokenizer: PreTrainedTokenizer,
    enable_thinking: bool,
    full_text: str,
) -> list[tuple[int, int]]:
    ranges = []
    close_marker = "</think>"
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        content = str(msg.get("content", ""))
        if close_marker not in content:
            continue

        prefix_messages = messages[:i]
        include_messages = messages[: i + 1]
        prefix_kwargs = {
            "tools": tools or None,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        include_kwargs = {
            "tools": tools or None,
            "tokenize": False,
            "add_generation_prompt": False,
        }
        if enable_thinking is not None:
            prefix_kwargs["enable_thinking"] = enable_thinking
            include_kwargs["enable_thinking"] = enable_thinking

        prefix_text = tokenizer.apply_chat_template(
            prefix_messages,
            **prefix_kwargs,
        )
        include_text = tokenizer.apply_chat_template(
            include_messages,
            **include_kwargs,
        )

        include_start = full_text.find(include_text)
        if include_start < 0:
            include_start = 0
        search_start = len(prefix_text)
        marker_pos = include_text.find(close_marker, search_start)
        if marker_pos < 0:
            marker_pos = include_text.find(close_marker)
        if marker_pos < 0:
            continue

        start_text = full_text[: include_start + marker_pos]
        start_idx = len(tokenizer(start_text).input_ids)
        end_idx = len(tokenizer(full_text[: include_start + len(include_text)]).input_ids)
        if end_idx > start_idx:
            ranges.append((start_idx, end_idx))

    return ranges
