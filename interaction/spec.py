from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Optional

from envs.base import Observation


@dataclass(frozen=True)
class ReActInferenceSpec:
    name: str
    build_system_prompt: Callable[[str], str]
    build_task_message: Callable[[str, Optional[Observation]], str]
    build_obs_message: Callable[[Observation], str]
    parse_assistant_action: Callable[[str, str], tuple[Any, str | None, bool]]
    tool_result_role: str


def load_react_inference_spec(name: str) -> ReActInferenceSpec:
    prompts = import_module(f"envs.{name}.prompts")

    return ReActInferenceSpec(
        name=name,
        build_system_prompt=prompts.build_system_prompt,
        build_task_message=prompts.build_task_message,
        build_obs_message=prompts.build_obs_message,
        parse_assistant_action=getattr(
            prompts,
            "parse_assistant_action",
            lambda _env_type, _content: ({}, None, False),
        ),
        tool_result_role=getattr(prompts, "TOOL_RESULT_ROLE", "user"),
    )
