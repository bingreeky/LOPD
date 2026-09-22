from __future__ import annotations

import logging

from inference.rollout import EpisodeError, rollout

logger = logging.getLogger(__name__)


async def phase_a_rollout(
    llm,
    memory_bank,
    env_builder,
    tasks: list[dict],
    *,
    n_retrieve: int,
    concurrency: int,
) -> list[dict]:
    queries = [task["instruction"] for task in tasks]
    retrieved = memory_bank.retrieve_many(queries, k=n_retrieve)
    results = await rollout(
        llm=llm, tasks=tasks, latent_memories=[None] * len(tasks),
        env_builder=env_builder, concurrency=concurrency,
    )

    records = []
    for task, query, items, result in zip(tasks, queries, retrieved, results):
        record = {"task_id": task["task_id"], "query_text": query, "retrieved": items,
                  "trajectory": [], "reward": 0.0, "n_steps": 0, "failed": True}
        if isinstance(result, EpisodeError):
            logger.warning("Task %s: rollout failed (%s), no supervision", task["task_id"], result.error_type)
        elif not items:
            logger.warning("Task %s: nothing retrieved, no supervision", task["task_id"])
        else:
            record.update(trajectory=result.trajectory, reward=float(result.reward), n_steps=int(result.steps), failed=False)
        records.append(record)

    if all(record["failed"] for record in records):
        raise RuntimeError(f"all {len(records)} rollouts failed; aborting the step")
    return records
