
from __future__ import annotations

from typing import Callable, List

from envs.envscaler.data import load_envscaler_samples
from envs.envscaler.env import ENV_TYPE, EnvScalerEnv


def build_envscaler_tasks(samples: List[dict]) -> List[dict]:
    return [
        {
            "index": i,
            "env_type": ENV_TYPE,
            "instruction": s["task"],
            "task_id": s["task_id"],
        }
        for i, s in enumerate(samples)
    ]


def build_envscaler_env_builder(
    samples: List[dict],
    *,
    max_steps: int = 30,
    tool_protocol: str = "qwen3",
) -> Callable[[dict], EnvScalerEnv]:
    def _builder(_task: dict) -> EnvScalerEnv:
        return EnvScalerEnv(
            samples,
            max_steps=max_steps,
            tool_protocol=tool_protocol,
        )

    return _builder


def load_rollout_setup(
    rl_scenario_path: str,
    env_meta_path: str,
    *,
    max_steps: int = 30,
    tool_protocol: str = "qwen3",
):
    samples = load_envscaler_samples(rl_scenario_path, env_meta_path)
    tasks = build_envscaler_tasks(samples)
    env_builder = build_envscaler_env_builder(
        samples,
        max_steps=max_steps,
        tool_protocol=tool_protocol,
    )
    return samples, tasks, env_builder
