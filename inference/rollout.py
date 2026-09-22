import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import torch

from backends.base import BaseLLM
from interaction.react import ReActAgent
from interaction.runner import EpisodeResult, run_episode
from interaction.spec import load_react_inference_spec

logger = logging.getLogger(__name__)


@dataclass
class EpisodeError:
    task_id: str
    instruction: str
    error_type: str
    error_message: str


async def rollout(
    llm: BaseLLM,
    tasks: list[dict],
    latent_memories: list[Optional[torch.Tensor]],
    env_builder,
    concurrency: int,
) -> list[EpisodeResult | EpisodeError]:
    if len(tasks) != len(latent_memories):
        raise ValueError("tasks and latent_memories must have the same length")

    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(len(tasks)):
        queue.put_nowait(i)
    results: list[EpisodeResult | EpisodeError | None] = [None] * len(tasks)
    specs = {env_type: load_react_inference_spec(env_type) for env_type in {task["env_type"] for task in tasks}}
    finished = 0

    async def worker() -> None:
        nonlocal finished
        while True:
            try:
                i = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            task = tasks[i]
            env = env_builder(task)
            try:
                agent = ReActAgent(llm, specs[task["env_type"]])
                result = await run_episode(env, agent, task["index"], latent_embeds=latent_memories[i])
                results[i] = result
                finished += 1
                logger.info("[%d/%d] task %s: reward=%.3f steps=%d",
                            finished, len(tasks), result.task_id, result.reward, result.steps)
            except Exception as exc:
                results[i] = EpisodeError(
                    task_id=str(task.get("task_id", "")),
                    instruction=str(task.get("instruction", "")),
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                )
                finished += 1
                logger.error("[%d/%d] task %s failed: %s", finished, len(tasks), task.get("task_id", "?"), exc)
            finally:
                env.close()

    await asyncio.gather(*[worker() for _ in range(max(1, min(concurrency, len(tasks))))])
    return results
