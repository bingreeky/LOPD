from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.config import build_memory_bank_from_config, load_config
from memory.compressor import (
    build_compressor_from_model_config,
    load_compressor,
    load_train_state,
)
from memory.lora import attach_lora, create_lora_config
from memory.qformer import QFormer
from memory.train_sft import load_sft_dataset, train_compressor_sft
from training.sampling import seed_everything

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cold-start training of the latent compressor")
    parser.add_argument("--config", required=True, help="Training config yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_path = config["model_path"]
    compressor_cfg = config["compressor"]
    data_cfg = config["data"]
    train_cfg = config["training"]
    qformer_device = train_cfg["qformer_device"]

    seed = int(train_cfg["seed"])
    seed_everything(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.requires_grad_(False)

    checkpoint = compressor_cfg.get("checkpoint")
    if checkpoint:
        qformer, adapter_dir, manifest = load_compressor(checkpoint, device=qformer_device)
        _check_qformer_architecture(qformer, compressor_cfg["qformer"])
        if manifest["prompt_protocol"] != data_cfg["prompt_protocol"]:
            raise ValueError(
                f"prompt_protocol mismatch: checkpoint={manifest['prompt_protocol']!r} "
                f"config={data_cfg['prompt_protocol']!r}"
            )
        adapter_name = manifest["adapter_name"]
        model = attach_lora(model, adapter_path=adapter_dir, adapter_name=adapter_name, is_trainable=True)
        train_state = load_train_state(checkpoint)
        start_step = int(manifest["step"])
        run_dir = os.path.dirname(os.path.abspath(checkpoint))
        logger.info("Resuming from %s at step %d; run directory %s", checkpoint, start_step, run_dir)
    else:
        lora_cfg = compressor_cfg["lora"]
        adapter_name = lora_cfg["adapter_name"]
        qformer = build_compressor_from_model_config(model_path, compressor_cfg["qformer"])
        model = attach_lora(
            model,
            config=create_lora_config(
                rank=int(lora_cfg["rank"]),
                alpha=int(lora_cfg["alpha"]),
                dropout=float(lora_cfg["dropout"]),
                target_modules=lora_cfg["target_modules"],
            ),
            adapter_name=adapter_name,
        )
        train_state = None
        start_step = 0
        run_name = train_cfg.get("run_name") or f"sft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = os.path.join(train_cfg["checkpoint_dir"], run_name)
        logger.info("Initialized a fresh compressor; run directory %s", run_dir)

    qformer = qformer.bfloat16()
    model.train()
    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    memory_bank = build_memory_bank_from_config(config)
    dataset = load_sft_dataset(data_cfg["path"])

    train_compressor_sft(
        model=model,
        tokenizer=tokenizer,
        qformer=qformer,
        memory_bank=memory_bank,
        dataset=dataset,
        run_dir=run_dir,
        adapter_name=adapter_name,
        num_steps=int(train_cfg["num_steps"]),
        batch_size=int(train_cfg["batch_size"]),
        train_mini_batch_size=int(train_cfg["train_mini_batch_size"]),
        lr=float(train_cfg["lr"]),
        qformer_grad_clip=float(train_cfg["qformer_grad_clip"]),
        lora_grad_clip=float(train_cfg["lora_grad_clip"]),
        n_retrieve=int(train_cfg["n_retrieve"]),
        task_cond=bool(train_cfg["task_cond"]),
        enable_thinking=bool(train_cfg["enable_thinking"]),
        prompt_protocol=data_cfg["prompt_protocol"],
        max_seq_len=int(train_cfg["max_seq_len"]),
        qformer_device=qformer_device,
        save_every=int(train_cfg["save_every"]),
        seed=seed,
        tensorboard_dir=train_cfg.get("tensorboard_dir"),
        start_step=start_step,
        train_state=train_state,
    )


def _check_qformer_architecture(qformer: QFormer, qformer_cfg: dict) -> None:
    expected = {
        "depth": int(qformer_cfg["depth"]),
        "num_queries": int(qformer_cfg["num_queries"]),
        "ff_mult": int(qformer_cfg["ff_mult"]),
        "share_layers": bool(qformer_cfg["share_layers"]),
    }
    actual = {
        "depth": qformer.depth,
        "num_queries": qformer.num_queries,
        "ff_mult": qformer.ff_mult,
        "share_layers": qformer.shared,
    }
    if expected != actual:
        raise ValueError(f"QFormer architecture mismatch: checkpoint={actual} config={expected}")


if __name__ == "__main__":
    main()
