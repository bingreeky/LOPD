from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Optional

from envs.base import Observation


@dataclass(frozen=True)
class ReActInferenceSpec:
    build_task_message: Callable[[str, Optional[Observation]], str]
    build_obs_message: Callable[[Observation], str]
    parse_assistant_action: Callable[[str], dict]


def load_react_inference_spec(env_type: str) -> ReActInferenceSpec:
    prompts = import_module(f"envs.{env_type}.prompts")
    return ReActInferenceSpec(
        build_task_message=prompts.build_task_message,
        build_obs_message=prompts.build_obs_message,
        parse_assistant_action=prompts.parse_assistant_action,
    )
