from __future__ import annotations

import json
import logging
import os
import shutil
import time

import torch.multiprocessing as mp

from backends.event_loop import run_async
from lopd import ipc
from lopd.rollout import phase_a_rollout
from lopd.worker import phase_b_worker
from memory.compressor import compressor_fingerprint
from training.sampling import next_batch
from training.tensorboard import tensorboard_writer

logger = logging.getLogger(__name__)

STATE_FILE = "state.json"


def train_lopd(
    *,
    llm,
    memory_bank,
    tasks: list[dict],
    env_builder,
    model_path: str,
    cold_start_dir: str,
    run_dir: str,
    run_name: str,
    num_steps: int,
    batch_size: int,
    n_retrieve: int,
    rollout_concurrency: int,
    lr: float,
    grad_clip: float,
    top_k: int,
    max_seq_len: int,
    gradient_checkpointing: bool,
    enable_thinking: bool,
    save_every: int,
    nproc: int,
    workers_port: int,
    seed: int,
    constraint: dict,
    tensorboard_dir: str | None,
    state: dict,
) -> None:
    latest_dir = os.path.join(run_dir, "latest")
    student_dir = os.path.join(latest_dir, "student")
    compressor_dir = os.path.join(latest_dir, "compressor")
    anchor_cache_dir = os.path.join(run_dir, "c_phi0", compressor_fingerprint(cold_start_dir, model_path))
    os.makedirs(latest_dir, exist_ok=True)
    ipc.cleanup_run(run_name)

    start_step, epoch, position, beta = state["step"], state["epoch"], state["position"], state["beta"]
    if start_step >= num_steps:
        logger.info("Checkpoint is at step %d and num_steps is %d; nothing to train", start_step, num_steps)
        return
    if start_step > 0:
        run_async(llm.update_weights_from_path(student_dir))

    writer = tensorboard_writer(tensorboard_dir)
    for step in range(start_step, num_steps):
        t0 = time.time()
        batch, epoch, position = next_batch(tasks, seed, epoch, position, batch_size)

        rollouts = run_async(phase_a_rollout(llm, memory_bank, env_builder, batch, n_retrieve=n_retrieve, concurrency=rollout_concurrency))
        n_ok = sum(not r["failed"] for r in rollouts)
        pass_rate = sum(r["reward"] > 0 for r in rollouts if not r["failed"]) / max(n_ok, 1)
        mean_steps = sum(r["n_steps"] for r in rollouts if not r["failed"]) / max(n_ok, 1)
        run_async(llm.release_gpu())

        phase_cfg = {
            "step": step, "lr": lr, "grad_clip": grad_clip, "top_k": top_k, "max_seq_len": max_seq_len,
            "gradient_checkpointing": gradient_checkpointing, "enable_thinking": enable_thinking,
            "constraint": {**constraint, "beta": beta},
        }
        current_compressor = compressor_dir if os.path.isfile(os.path.join(compressor_dir, "qformer.pt")) else cold_start_dir
        metrics = _spawn_phase_b(
            rollouts, phase_cfg, run_name=run_name, step=step, nproc=nproc, workers_port=workers_port,
            model_path=model_path, student_dir=student_dir, compressor_dir=current_compressor,
            cold_start_dir=cold_start_dir, anchor_cache_dir=anchor_cache_dir, latest_dir=latest_dir,
        )
        beta = float(metrics.get("beta", beta))

        run_async(llm.resume_gpu(student_dir))

        metrics.update({"rollout_pass_rate": pass_rate, "rollout_mean_steps": mean_steps, "rollout_ok": n_ok, "step_seconds": time.time() - t0})
        logger.info("Step %d: %s", step, "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(metrics.items())))
        if writer is not None:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f"lopd/{key}", value, step)
            writer.flush()

        _write_state(latest_dir, {"step": step + 1, "epoch": epoch, "position": position, "seed": seed, "beta": beta})
        if (step + 1) % save_every == 0:
            step_dir = os.path.join(run_dir, f"step_{step + 1}")
            shutil.rmtree(step_dir, ignore_errors=True)
            shutil.copytree(latest_dir, step_dir, copy_function=os.link)
            logger.info("Saved checkpoint %s", step_dir)

    if writer is not None:
        writer.close()


def _write_state(latest_dir: str, state: dict) -> None:
    path = os.path.join(latest_dir, STATE_FILE)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(path + ".tmp", path)


def _spawn_phase_b(rollouts, phase_cfg, *, run_name, step, nproc, workers_port, model_path, student_dir,
                   compressor_dir, cold_start_dir, anchor_cache_dir, latest_dir) -> dict:
    ipc_dir = ipc.ipc_dir_for_step(run_name, step)
    ipc.write_inputs(ipc_dir, rollouts, phase_cfg)
    try:
        mp.spawn(
            phase_b_worker,
            args=(nproc, str(ipc_dir), model_path, student_dir, compressor_dir, cold_start_dir, anchor_cache_dir, latest_dir, workers_port),
            nprocs=nproc, join=True, daemon=False,
        )
        return ipc.read_metrics(ipc_dir)
    finally:
        ipc.cleanup(ipc_dir)
