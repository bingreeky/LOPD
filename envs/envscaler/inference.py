
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import pandas as pd

from inference.rollout import rollout
from envs.envscaler.builder import load_rollout_setup
from backends.base import BaseLLM


@dataclass
class EvalSummary:
    total: int
    completed: int
    mean_reward: float
    full_pass_rate: float
    mean_steps: float
    total_steps: int
    reward_buckets: dict
    result_dir: str


async def inference(llm: BaseLLM, envscaler_cfg: dict, result_dir: str = "results") -> EvalSummary:
    max_steps = envscaler_cfg.get("max_steps", 30)
    concurrency = envscaler_cfg.get("concurrency", 32)
    max_tasks = envscaler_cfg.get("max_tasks")
    tool_protocol = envscaler_cfg.get("tool_protocol", "qwen3")
    rl_path = envscaler_cfg["rl_scenario_path"]
    env_meta_path = envscaler_cfg["env_meta_path"]

    samples, tasks, env_builder = load_rollout_setup(
        rl_path,
        env_meta_path,
        max_steps=max_steps,
        tool_protocol=tool_protocol,
    )
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(result_dir, f"envscaler_{timestamp}")

    print("inference  : envscaler")
    print(f"tasks      : {len(tasks)}")
    print(f"max_steps  : {max_steps}  concurrency: {concurrency}")
    print(f"tool_proto : {tool_protocol}")
    print(f"result_dir : {run_dir}")

    results = await rollout(
        llm=llm,
        tasks=tasks,
        latent_memories=[None] * len(tasks),
        env_builder=env_builder,
        concurrency=concurrency,
        return_indices=True,
        inference_name="envscaler",
    )
    eps = [ep for _pos, ep in results]

    summary = _make_summary(len(tasks), eps, run_dir)
    _save_results(eps, summary)
    return summary


def _make_summary(total, eps, run_dir) -> EvalSummary:
    rewards = [float(ep.reward) for ep in eps]
    n = len(rewards)
    edges = [0.0, 0.001, 0.25, 0.5, 0.75, 0.999, 1.001]
    labels = ["==0", "(0,0.25)", "[0.25,0.5)", "[0.5,0.75)", "[0.75,1)", "==1.0"]
    buckets = {l: 0 for l in labels}
    for r in rewards:
        for i in range(len(labels)):
            if edges[i] <= r < edges[i + 1] or (i == len(labels) - 1 and r >= edges[i]):
                buckets[labels[i]] += 1
                break
    return EvalSummary(
        total=total, completed=n,
        mean_reward=sum(rewards) / n if n else 0.0,
        full_pass_rate=sum(1 for r in rewards if r >= 0.999) / n if n else 0.0,
        mean_steps=sum(ep.steps for ep in eps) / n if n else 0.0,
        total_steps=sum(ep.steps for ep in eps),
        reward_buckets=buckets, result_dir=run_dir,
    )


def _save_results(eps, summary: EvalSummary):
    os.makedirs(summary.result_dir, exist_ok=True)
    rows = [{
        "task_id": ep.task_id, "env_type": ep.env_type,
        "instruction": ep.instruction, "reward": ep.reward, "steps": ep.steps,
        "trajectory": json.dumps(ep.trajectory, ensure_ascii=False),
        "info": json.dumps(ep.info, ensure_ascii=False),
    } for ep in eps]
    pd.DataFrame(rows).to_parquet(
        os.path.join(summary.result_dir, "trajectories.parquet"), index=False,
    )
    with open(os.path.join(summary.result_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, ensure_ascii=False)
    print(f"\n{'='*56}")
    print(f"  saved → {summary.result_dir}")
    print(f"  completed   : {summary.completed}/{summary.total}")
    print(f"  mean_reward : {summary.mean_reward:.4f}")
    print(f"  full_pass   : {summary.full_pass_rate:.2%}  (reward==1.0)")
    print(f"  mean_steps  : {summary.mean_steps:.1f}")
    print(f"  reward dist : {summary.reward_buckets}")
    print(f"{'='*56}")
