from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressorPromptProtocol:
    name: str
    latent_framing_before: str
    latent_framing_after: str
    task_prefix_head: str
    task_prefix_tail: str

    def task_prefix(self, task_text: str) -> str:
        return self.task_prefix_head + task_text.rstrip() + self.task_prefix_tail


ENVSCALER_V1 = CompressorPromptProtocol(
    name="envscaler_v1",
    latent_framing_before=(
        "The following is a REFERENCE EXAMPLE of how a DIFFERENT, already-solved task was "
        "handled, shown only to illustrate the general approach. It was performed in a SEPARATE "
        "session — in YOUR task below, NOTHING has been done yet and the environment is in "
        "its initial state. You must perform every step yourself by calling the tools and "
        "reading their actual results; do NOT assume any step is already complete.\n\n"
    ),
    latent_framing_after=(
        "\n\n--- end of reference example ---\n\n"
        "Now complete YOUR task below, starting from scratch (the environment is untouched):\n\n"
    ),
    task_prefix_head="Task to solve:\n",
    task_prefix_tail="\n\nReference past trajectory:\n",
)

_PROTOCOLS = {ENVSCALER_V1.name: ENVSCALER_V1}


def resolve_compressor_prompt_protocol(protocol: str | CompressorPromptProtocol) -> CompressorPromptProtocol:
    if isinstance(protocol, CompressorPromptProtocol):
        return protocol
    try:
        return _PROTOCOLS[str(protocol).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown compressor prompt protocol {protocol!r}; expected one of: {', '.join(sorted(_PROTOCOLS))}") from exc
