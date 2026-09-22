from __future__ import annotations

import logging
import os
from datetime import datetime

import torch

from envs.envscaler.builder import load_rollout_setup
from envs.envscaler.inference import EvalSummary, save_results
from inference.rollout import rollout
from memory.compression import compress_hidden_to_latent, latent_cache_key, trajectory_text
from memory.prompt_protocol import CompressorPromptProtocol, resolve_compressor_prompt_protocol

logger = logging.getLogger(__name__)


async def inference_with_latent(
    llm,
    qformer,
    memory_bank,
    envscaler_cfg: dict,
    *,
    n_retrieve: int,
    result_dir: str,
    encoder_lora_name: str,
    enable_thinking: bool,
    task_cond: bool,
    prompt_protocol: str | CompressorPromptProtocol,
) -> EvalSummary:
    tasks, env_builder = load_rollout_setup(envscaler_cfg)
    tasks = tasks[: envscaler_cfg.get("max_tasks")]
    concurrency = int(envscaler_cfg["concurrency"])

    protocol = resolve_compressor_prompt_protocol(prompt_protocol)
    run_dir = os.path.join(
        result_dir, f"envscaler_latent_nret{n_retrieve}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    logger.info(
        "EnvScaler latent inference: %d tasks, n_retrieve=%d, prompt_protocol=%s, result_dir=%s",
        len(tasks), n_retrieve, protocol.name, run_dir,
    )

    queries = [task["instruction"] for task in tasks]
    retrieved_lists = memory_bank.retrieve_many(queries, k=n_retrieve)
    qformer_device = next(qformer.parameters()).device
    latent_cache: dict[str, torch.Tensor] = {}
    latents: list[torch.Tensor | None] = []
    logger.info("Compressing retrieved experiences into latent tokens")
    for i, (task, retrieved) in enumerate(zip(tasks, retrieved_lists)):
        if not retrieved:
            logger.warning("Task %s: no retrieved experience, running without memory", task["task_id"])
        latents.append(await _compress_retrieved(
            llm=llm, qformer=qformer, retrieved=retrieved,
            task_text=task["instruction"] if task_cond else None,
            qformer_device=qformer_device, latent_cache=latent_cache,
            encoder_lora_name=encoder_lora_name, enable_thinking=enable_thinking, protocol=protocol,
        ))
        if (i + 1) % 20 == 0:
            logger.info("Compressed %d/%d", i + 1, len(tasks))

    logger.info("Rolling out with latent tokens injected")
    results = await rollout(
        llm=llm, tasks=tasks, latent_memories=latents,
        env_builder=env_builder, concurrency=concurrency,
    )
    return save_results(
        results, run_dir,
        extra_columns={"nret": int(n_retrieve)},
        extra_summary={"n_retrieve": int(n_retrieve), "prompt_protocol": protocol.name},
    )


async def _compress_retrieved(
    *,
    llm,
    qformer,
    retrieved: list[dict],
    task_text: str | None,
    qformer_device,
    latent_cache: dict[str, torch.Tensor],
    encoder_lora_name: str,
    enable_thinking: bool,
    protocol: CompressorPromptProtocol,
) -> torch.Tensor | None:
    task_prefix = protocol.task_prefix(task_text) if task_text else ""
    latents = []
    for entry in retrieved:
        text = trajectory_text(entry, llm.tokenizer, enable_thinking=enable_thinking)
        key = latent_cache_key(text, task_text=task_text)
        if key not in latent_cache:
            hidden = await llm.get_hidden_states(task_prefix + text, lora_path=encoder_lora_name, timeout_s=120.0)
            latent = compress_hidden_to_latent(hidden, qformer, qformer_device)
            if latent is None:
                continue
            latent_cache[key] = latent
        latents.append(latent_cache[key])
    return torch.cat(latents, dim=0) if latents else None
