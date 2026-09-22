from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from datetime import datetime

import torch

from configs.config import build_llm_from_config, build_memory_bank_from_config, load_config
from envs.envscaler.builder import load_rollout_setup
from training.sampling import seed_everything
from lopd.trainer import STATE_FILE, train_lopd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent on-policy self-distillation")
    parser.add_argument("--config", required=True, help="Training config yaml")
    return parser.parse_args()


def main() -> None:
    config = load_config(parse_args().config)
    train_cfg = config["training"]
    env_cfg = config["datasets"]["envscaler"]
    if config["llm"]["model_path"] != config["model_path"]:
        raise ValueError("llm.model_path must equal model_path: the student is served by the same backbone it is trained from")
    nproc = int(train_cfg["nproc"])
    batch_size = int(train_cfg["batch_size"])
    if batch_size < nproc:
        raise ValueError(f"batch_size={batch_size} must be at least nproc={nproc}")
    if torch.cuda.device_count() < nproc:
        raise RuntimeError(f"nproc={nproc} but only {torch.cuda.device_count()} GPUs are visible")
    seed = int(train_cfg["seed"])
    seed_everything(seed)

    checkpoint = train_cfg.get("checkpoint")
    if checkpoint:
        run_dir = os.path.dirname(os.path.abspath(checkpoint))
        latest_dir = os.path.join(run_dir, "latest")
        if os.path.realpath(checkpoint) != os.path.realpath(latest_dir):
            shutil.rmtree(latest_dir, ignore_errors=True)
            shutil.copytree(checkpoint, latest_dir, copy_function=os.link)
        with open(os.path.join(latest_dir, STATE_FILE), encoding="utf-8") as f:
            state = json.load(f)
        if state["seed"] != seed:
            raise ValueError(f"training.seed={seed} does not match the checkpoint's seed={state['seed']}")
        logger.info("Resuming from %s at step %d", checkpoint, state["step"])
    else:
        run_name = train_cfg.get("run_name") or f"lopd_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(train_cfg["checkpoint_dir"], run_name)
        state = {"step": 0, "epoch": 0, "position": 0, "seed": seed, "beta": float(train_cfg["constraint"]["beta_init"])}
    run_name = os.path.basename(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    llm = build_llm_from_config(config)
    try:
        memory_bank = build_memory_bank_from_config(config)
        tasks, env_builder = load_rollout_setup(env_cfg)
        logger.info("Task pool: %d tasks; run directory %s", len(tasks), run_dir)
        train_lopd(
            llm=llm,
            memory_bank=memory_bank,
            tasks=tasks,
            env_builder=env_builder,
            model_path=config["model_path"],
            cold_start_dir=config["compressor"]["checkpoint"],
            run_dir=run_dir,
            run_name=run_name,
            num_steps=int(train_cfg["num_steps"]),
            batch_size=batch_size,
            n_retrieve=int(train_cfg["n_retrieve"]),
            rollout_concurrency=int(env_cfg["concurrency"]),
            lr=float(train_cfg["lr"]),
            grad_clip=float(train_cfg["grad_clip"]),
            top_k=int(train_cfg["top_k"]),
            max_seq_len=int(train_cfg["max_seq_len"]),
            gradient_checkpointing=bool(train_cfg["gradient_checkpointing"]),
            enable_thinking=bool(config["llm"]["enable_thinking"]),
            save_every=int(train_cfg["save_every"]),
            nproc=nproc,
            workers_port=int(train_cfg["workers_port"]),
            seed=seed,
            constraint={
                "enabled": bool(train_cfg["constraint"]["enabled"]),
                "margin_m": float(train_cfg["constraint"]["margin_m"]),
                "anchor_lambda": float(train_cfg["constraint"]["anchor_lambda"]),
                "eta_beta": float(train_cfg["constraint"]["eta_beta"]),
                "lr_phi": float(train_cfg["constraint"]["lr_phi"]),
            },
            tensorboard_dir=train_cfg.get("tensorboard_dir"),
            state=state,
        )
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
