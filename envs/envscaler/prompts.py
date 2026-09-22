from __future__ import annotations

from typing import Optional

from envs.base import Observation

def build_task_message(instruction: str, initial_obs: Optional[Observation] = None) -> str:
    if initial_obs and initial_obs.get("text"):
        return initial_obs["text"]
    return instruction


def build_obs_message(obs: Observation) -> str:
    text = (obs.get("text") or "").strip()
    if obs.get("_obs_type") == "tool":
        return f"<tool_response>\n{text}\n</tool_response>"
    return text if text else "(no output)"


def parse_assistant_action(content: str) -> dict:
    return {"_raw_text": content or ""}
