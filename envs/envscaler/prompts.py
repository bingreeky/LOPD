
from __future__ import annotations

import re
from typing import Optional

from envs.base import Observation
from envs.envscaler.runtime import SYSTEM_PROMPT

MEMORY_FRAMING_BEFORE = (
    "The following is a REFERENCE EXAMPLE of how a DIFFERENT, already-solved task was "
    "handled, shown only to illustrate the general approach. It was performed in a SEPARATE "
    "session — in YOUR task below, NOTHING has been done yet and the environment is in "
    "its initial state. You must perform every step yourself by calling the tools and "
    "reading their actual results; do NOT assume any step is already complete.\n\n"
)
MEMORY_FRAMING_AFTER = (
    "\n\n--- end of reference example ---\n\n"
    "Now complete YOUR task below, starting from scratch (the environment is untouched):\n\n"
)

TOOL_RESULT_ROLE = "user"


def build_system_prompt(_env_type: str) -> str:
    return SYSTEM_PROMPT


def build_task_message(instruction: str, initial_obs: Optional[Observation] = None) -> str:
    if initial_obs and initial_obs.get("text"):
        return initial_obs["text"]
    return instruction


def build_obs_message(obs: Observation) -> str:
    text = (obs.get("text") or "").strip()
    if obs.get("_obs_type") == "tool":
        return f"<tool_response>\n{text}\n</tool_response>"
    return text if text else "(no output)"


def parse_assistant_action(_env_type: str, content: str) -> tuple[dict, Optional[str], bool]:
    content = content or ""
    think = None
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if m:
        think = m.group(1).strip() or None
    return {"_raw_text": content}, think, True
