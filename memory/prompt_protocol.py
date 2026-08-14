
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressorPromptProtocol:
    name: str
    latent_framing_before: str
    latent_framing_after: str
    task_prefix_head: str
    task_prefix_tail: str


_ALIASES = {
    "envscaler": "envscaler_v1",
    "envscaler_v1": "envscaler_v1",
}


def resolve_compressor_prompt_protocol(
    protocol: str | CompressorPromptProtocol | None,
) -> CompressorPromptProtocol:
    if isinstance(protocol, CompressorPromptProtocol):
        return protocol

    raw_name = "envscaler_v1" if protocol is None else str(protocol).strip().lower()
    if not raw_name:
        raw_name = "envscaler_v1"
    try:
        name = _ALIASES[raw_name]
    except KeyError as exc:
        allowed = ", ".join(sorted(_ALIASES))
        raise ValueError(
            f"Unknown compressor prompt protocol {protocol!r}; expected one of: {allowed}"
        ) from exc

    if name == "envscaler_v1":
        from envs.envscaler.prompts import MEMORY_FRAMING_AFTER, MEMORY_FRAMING_BEFORE

        return CompressorPromptProtocol(
            name=name,
            latent_framing_before=MEMORY_FRAMING_BEFORE,
            latent_framing_after=MEMORY_FRAMING_AFTER,
            task_prefix_head="Task to solve:\n",
            task_prefix_tail="\n\nReference past trajectory:\n",
        )

    raise ValueError(f"Protocol {name!r} is registered but has no implementation")
