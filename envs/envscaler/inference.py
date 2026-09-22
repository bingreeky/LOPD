import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

from backends.base import BaseLLM
from envs.envscaler.builder import load_rollout_setup
from inference.rollout import EpisodeError, EpisodeResult, rollout

logger = logging.getLogger(__name__)


@dataclass
class EvalSummary:
    total: int
    completed: int
    errors: int
    mean_reward: float
    full_pass_rate: float
    mean_steps: float
    result_dir: str


async def inference(llm: BaseLLM, envscaler_cfg: dict, *, result_dir: str) -> EvalSummary:
    tasks, env_builder = load_rollout_setup(envscaler_cfg)
    tasks = tasks[: envscaler_cfg.get("max_tasks")]
    concurrency = int(envscaler_cfg["concurrency"])

    run_dir = os.path.join(result_dir, f"envscaler_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    logger.info("EnvScaler inference: %d tasks, concurrency=%d, result_dir=%s", len(tasks), concurrency, run_dir)

    results = await rollout(
        llm=llm, tasks=tasks, latent_memories=[None] * len(tasks), env_builder=env_builder, concurrency=concurrency,
    )
    return save_results(results, run_dir)


def save_results(
    results: list[EpisodeResult | EpisodeError],
    run_dir: str,
    extra_columns: dict | None = None,
    extra_summary: dict | None = None,
) -> EvalSummary:
    rows = []
    for item in results:
        if isinstance(item, EpisodeError):
            rows.append({
                "task_id": item.task_id, "instruction": item.instruction,
                "reward": 0.0, "steps": 0, "trajectory": "[]", "info": "{}",
                "error_type": item.error_type, "error_message": item.error_message,
            })
        else:
            rows.append({
                "task_id": item.task_id, "instruction": item.instruction,
                "reward": float(item.reward), "steps": int(item.steps),
                "trajectory": json.dumps(item.trajectory, ensure_ascii=False),
                "info": json.dumps(item.info, ensure_ascii=False),
                "error_type": "", "error_message": "",
            })
        rows[-1].update(extra_columns or {})

    df = pd.DataFrame(rows)
    n_errors = sum(isinstance(item, EpisodeError) for item in results)
    summary = EvalSummary(
        total=len(results),
        completed=len(results) - n_errors,
        errors=n_errors,
        mean_reward=float(df["reward"].mean()) if len(df) else 0.0,
        full_pass_rate=float((df["reward"] >= 1.0).mean()) if len(df) else 0.0,
        mean_steps=float(df["steps"].mean()) if len(df) else 0.0,
        result_dir=run_dir,
    )

    os.makedirs(run_dir, exist_ok=True)
    df.to_parquet(os.path.join(run_dir, "trajectories.parquet"), index=False)
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({**asdict(summary), **(extra_summary or {})}, f, indent=2, ensure_ascii=False)
    logger.info(
        "Saved %s: total=%d completed=%d errors=%d mean_reward=%.4f full_pass=%.2f%% mean_steps=%.1f",
        run_dir, summary.total, summary.completed, summary.errors,
        summary.mean_reward, 100 * summary.full_pass_rate, summary.mean_steps,
    )
    return summary
