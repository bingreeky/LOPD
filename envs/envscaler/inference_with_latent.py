
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import torch

from interaction.react import ReActAgent
from interaction.runner import run_episode
from envs.envscaler.builder import load_rollout_setup
from inference.rollout import _AtomicCounter
from memory.compression import (
    compress_hidden_to_latent,
    latent_cache_key,
    trajectory_text,
)
from memory.prompt_protocol import resolve_compressor_prompt_protocol


@dataclass
class _ErrorResult:
    task_id: str = ""
    instruction: str = ""
    reward: float = 0.0
    steps: int = 0
    trajectory: list = field(default_factory=list)
    error_type: str = ""
    error_message: str = ""


async def _rollout_with_latent(llm, tasks, latents, env_builder, concurrency):
    task_queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(len(tasks)):
        task_queue.put_nowait(i)
    collected: dict[int, object] = {}
    counter = _AtomicCounter()

    async def _worker():
        while True:
            try:
                i = task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            task = tasks[i]
            env = env_builder(task)
            try:
                agent = ReActAgent(llm=llm, env_type=task["env_type"], inference_name="envscaler")
                result = await run_episode(env, agent, task["index"], latent_embeds=latents[i])
                collected[i] = result
                n = counter.increment()
                print(f"  [{n}/{len(tasks)}] {task.get('task_id','?')} "
                      f"reward={result.reward:.3f} steps={result.steps}", flush=True)
            except Exception as exc:
                collected[i] = _ErrorResult(
                    task_id=task.get("task_id", ""),
                    instruction=task.get("instruction", ""),
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                )
                n = counter.increment()
                print(f"  [{n}/{len(tasks)}] {task.get('task_id','?')} ERROR: {exc}", flush=True)
            finally:
                env.close()

    actual = min(concurrency, len(tasks))
    await asyncio.gather(*[_worker() for _ in range(actual)])
    return collected


async def _compress_retrieved_for_task(
    *,
    llm,
    qformer,
    retrieved: list[dict],
    task_query: str,
    qformer_device,
    latent_cache: dict[str, torch.Tensor],
    encoder_lora_path: str | None,
    enable_thinking: bool,
    condition_on_task: bool,
    protocol,
) -> torch.Tensor | None:
    conditioned_task = task_query if condition_on_task else None
    task_prefix = (
        protocol.task_prefix_head
        + conditioned_task.rstrip()
        + protocol.task_prefix_tail
    ) if conditioned_task is not None else ""

    task_latents = []
    for entry in retrieved:
        traj_text = trajectory_text(entry, llm.tokenizer, enable_thinking=enable_thinking)
        encoder_text = task_prefix + traj_text if task_prefix else traj_text
        key = latent_cache_key(traj_text, task_text=conditioned_task)
        if key not in latent_cache:
            memory_hidden = await llm.get_hidden_states(
                encoder_text, lora_path=encoder_lora_path, timeout_s=120.0,
            )
            latent = compress_hidden_to_latent(memory_hidden, qformer, qformer_device)
            if latent is not None:
                latent_cache[key] = latent
        if key in latent_cache:
            task_latents.append(latent_cache[key])
    return torch.cat(task_latents, dim=0) if task_latents else None


async def inference_with_latent(
    llm, qformer, memory_bank, envscaler_cfg: dict, *,
    n_retrieve: int = 1, device: str = "cuda", result_dir: str = "results",
    encoder_lora_path: str | None = None,
    enable_thinking: bool = True, task_cond: bool = False,
    prompt_protocol: str = "envscaler_v1",
):
    max_steps = envscaler_cfg.get("max_steps", 30)
    concurrency = envscaler_cfg.get("concurrency", 32)
    max_tasks = envscaler_cfg.get("max_tasks")
    tool_protocol = envscaler_cfg.get("tool_protocol", "qwen3")
    rl_path = envscaler_cfg["rl_scenario_path"]
    env_meta_path = envscaler_cfg["env_meta_path"]

    samples, tasks, env_builder = load_rollout_setup(
        rl_path, env_meta_path,
        max_steps=max_steps, tool_protocol=tool_protocol,
    )
    if max_tasks is not None:
        tasks = tasks[:max_tasks]

    protocol = resolve_compressor_prompt_protocol(prompt_protocol)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(result_dir, f"envscaler_latent_nret{n_retrieve}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    print("inference  : envscaler (with latent memory)")
    print(f"tasks      : {len(tasks)} | nret={n_retrieve}")
    print(f"prompt     : {protocol.name}")
    print(f"result_dir : {run_dir}")

    queries = [t["instruction"] for t in tasks]
    retrieved_lists = memory_bank.retrieve_many(queries, k=n_retrieve)
    qformer_device = next(qformer.parameters()).device
    latent_cache: dict[str, torch.Tensor] = {}
    latents: list[torch.Tensor | None] = [None] * len(tasks)
    print("[A] compressing retrieved memories -> latent...", flush=True)
    for i, retrieved in enumerate(retrieved_lists):
        latents[i] = await _compress_retrieved_for_task(
            llm=llm, qformer=qformer, retrieved=retrieved,
            task_query=tasks[i]["instruction"], qformer_device=qformer_device,
            latent_cache=latent_cache, encoder_lora_path=encoder_lora_path,
            enable_thinking=enable_thinking,
            condition_on_task=bool(task_cond), protocol=protocol,
        )
        if (i + 1) % 20 == 0:
            print(f"  [A] compressed {i+1}/{len(tasks)}", flush=True)

    print("[B] rollout with latent injected...", flush=True)
    collected = await _rollout_with_latent(llm, tasks, latents, env_builder, concurrency)

    total_tasks = len(tasks)
    rows = []
    n_errors = 0
    for i in range(total_tasks):
        ep = collected.get(i)
        if ep is None:
            rows.append({
                "task_id": tasks[i].get("task_id", ""),
                "instruction": tasks[i].get("instruction", ""),
                "reward": 0.0, "steps": 0, "nret": int(n_retrieve),
                "trajectory": "[]",
                "error_type": "NotExecuted", "error_message": "task not reached",
            })
            n_errors += 1
        elif isinstance(ep, _ErrorResult):
            rows.append({
                "task_id": ep.task_id, "instruction": ep.instruction,
                "reward": 0.0, "steps": 0, "nret": int(n_retrieve),
                "trajectory": "[]",
                "error_type": ep.error_type, "error_message": ep.error_message,
            })
            n_errors += 1
        else:
            rows.append({
                "task_id": ep.task_id, "instruction": ep.instruction,
                "reward": float(ep.reward), "steps": int(ep.steps),
                "nret": int(n_retrieve),
                "trajectory": json.dumps(ep.trajectory, ensure_ascii=False),
                "error_type": "", "error_message": "",
            })
    df = pd.DataFrame(rows)
    out = os.path.join(run_dir, "latent_eval_results.parquet")
    df.to_parquet(out, index=False)
    n_completed = total_tasks - n_errors
    mean_r = float(df["reward"].mean())
    pass_rate = float((df["reward"] >= 1.0).mean())
    summary = {
        "n_total": total_tasks, "n_completed": n_completed, "n_errors": n_errors,
        "mean_reward": mean_r, "pass_rate": pass_rate,
        "n_retrieve": n_retrieve,
        "prompt_protocol": protocol.name,
        "result_dir": run_dir, "parquet": out,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n[done] total={total_tasks} completed={n_completed} errors={n_errors} "
          f"mean_reward={mean_r:.4f} pass(=1.0)={pass_rate:.2%} -> {out}")
    return summary
