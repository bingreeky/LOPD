
import asyncio
import threading
from typing import Optional

import torch

from interaction.react import ReActAgent
from backends.base import BaseLLM
from interaction.runner import EpisodeResult, run_episode


async def rollout(
    llm: BaseLLM,
    tasks: list[dict],
    latent_memories: list[Optional[torch.Tensor]],
    env_builder,
    concurrency: int = 1,
    return_indices: bool = False,
    inference_name: str = "envscaler",
) -> list[EpisodeResult] | list[tuple[int, EpisodeResult]]:
    assert len(tasks) == len(latent_memories)

    if concurrency <= 1:
        return await _rollout_sequential(
            llm, tasks, latent_memories, env_builder,
            return_indices=return_indices, inference_name=inference_name,
        )
    return await _rollout_concurrent(
        llm, tasks, latent_memories, env_builder, concurrency,
        return_indices=return_indices, inference_name=inference_name,
    )


async def _rollout_sequential(llm, tasks, latent_memories, env_builder, return_indices=False,
                              inference_name="envscaler"):
    results = []
    for i, (task, latent) in enumerate(zip(tasks, latent_memories)):
        env = env_builder(task)
        try:
            agent = ReActAgent(llm=llm, env_type=task["env_type"], inference_name=inference_name)
            result = await run_episode(env, agent, task["index"], latent_embeds=latent)
            results.append((i, result) if return_indices else result)
        except Exception as exc:
            print(f"  rollout [{i+1}/{len(tasks)}] task {task.get('task_id','?')} ERROR: {exc}")
        finally:
            env.close()
    return results


async def _rollout_concurrent(llm, tasks, latent_memories, env_builder, concurrency, return_indices=False,
                              inference_name="envscaler"):
    actual = min(concurrency, len(tasks))
    task_queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(len(tasks)):
        task_queue.put_nowait(i)

    collected: dict[int, EpisodeResult] = {}
    counter = _AtomicCounter()

    async def _worker():
        while True:
            try:
                i = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            task = tasks[i]
            latent = latent_memories[i]
            env = env_builder(task)
            try:
                agent = ReActAgent(llm=llm, env_type=task["env_type"], inference_name=inference_name)
                result = await run_episode(env, agent, task["index"], latent_embeds=latent)
                collected[i] = result
                n = counter.increment()
                print(f"  rollout [{n}/{len(tasks)}] task {task.get('task_id','?')} "
                      f"reward={result.reward:.3f} steps={result.steps}", flush=True)
            except Exception as exc:
                n = counter.increment()
                print(f"  rollout [{n}/{len(tasks)}] task {task.get('task_id','?')} ERROR: {exc}", flush=True)
            finally:
                env.close()

    await asyncio.gather(*[_worker() for _ in range(actual)])
    if return_indices:
        return [(i, collected[i]) for i in range(len(tasks)) if i in collected]
    return [collected[i] for i in range(len(tasks)) if i in collected]


class _AtomicCounter:
    def __init__(self):
        self._value = 0
        self._lock = threading.Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value
