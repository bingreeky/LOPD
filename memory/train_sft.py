from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd
import torch

from memory.bank import MemoryBank
from memory.compression import compress_texts_via_pytorch, trajectory_text
from memory.compressor import CompressorTrainState, save_compressor, save_train_state
from memory.lora import (
    base_model_view,
    iter_lora_named_parameters,
    load_optimizer_state_by_name,
    optimizer_state_to_cpu_by_name,
)
from memory.prompt_protocol import CompressorPromptProtocol, resolve_compressor_prompt_protocol
from memory.qformer import QFormer
from memory.serialization import (
    IGNORE_INDEX,
    build_supervised_inputs,
    inject_latent_tokens_into_embeds,
)
from training.sampling import next_batch
from training.tensorboard import tensorboard_writer

logger = logging.getLogger(__name__)

LATEST_DIR = "latest"


def load_sft_dataset(path: str) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    missing = {"query_text", "trajectory"} - set(df.columns)
    if missing:
        raise ValueError(f"SFT parquet {path} is missing columns: {sorted(missing)}")

    samples: list[dict[str, Any]] = []
    for index, row in df.iterrows():
        query_text = str(row["query_text"]).strip()
        trajectory = _parse_json_cell(row["trajectory"])
        if not query_text or not trajectory:
            raise ValueError(f"SFT parquet {path} row {index}: empty query_text or trajectory")
        samples.append({
            "sample_id": str(row["task_id"]) if "task_id" in df.columns else str(index),
            "query_text": query_text,
            "trajectory": trajectory,
        })
    logger.info("Loaded %d SFT samples from %s", len(samples), path)
    return samples


def train_compressor_sft(
    *,
    model,
    tokenizer,
    qformer: QFormer,
    memory_bank: MemoryBank,
    dataset: list[dict[str, Any]],
    run_dir: str,
    adapter_name: str,
    num_steps: int,
    batch_size: int,
    train_mini_batch_size: int,
    lr: float,
    qformer_grad_clip: float,
    lora_grad_clip: float,
    n_retrieve: int,
    task_cond: bool,
    enable_thinking: bool,
    prompt_protocol: str | CompressorPromptProtocol,
    max_seq_len: int,
    qformer_device: str,
    save_every: int,
    seed: int,
    tensorboard_dir: str | None,
    start_step: int = 0,
    train_state: CompressorTrainState | None = None,
) -> None:
    if start_step >= num_steps:
        logger.info("Checkpoint is at step %d and num_steps is %d; nothing to train", start_step, num_steps)
        return
    protocol = resolve_compressor_prompt_protocol(prompt_protocol)
    manifest = {
        "prompt_protocol": protocol.name,
        "adapter_name": adapter_name,
        "task_cond": task_cond,
    }

    qformer = qformer.to(qformer_device)
    qformer_optimizer = torch.optim.AdamW(qformer.parameters(), lr=lr)

    lora_named_params = list(iter_lora_named_parameters(model))
    if not lora_named_params:
        raise RuntimeError("No trainable LoRA parameters found on the encoder model")
    lora_params = [param for _, param in lora_named_params]
    lora_optimizer = torch.optim.AdamW(lora_params, lr=lr)

    epoch, position = 0, 0
    if train_state is not None:
        if train_state.seed != seed:
            raise ValueError(f"training.seed={seed} does not match the checkpoint's seed={train_state.seed}")
        qformer_optimizer.load_state_dict(train_state.qformer_optimizer)
        load_optimizer_state_by_name(lora_optimizer, lora_named_params, train_state.lora_optimizer)
        epoch, position = train_state.epoch, train_state.position
        logger.info("Resumed at step %d (epoch %d, position %d)", start_step, epoch, position)

    writer = tensorboard_writer(tensorboard_dir)
    embed_device = model.get_input_embeddings().weight.device
    backbone = base_model_view(model)

    for step in range(start_step, num_steps):
        batch, epoch, position = next_batch(dataset, seed, epoch, position, batch_size)
        retrieved = memory_bank.retrieve_many([s["query_text"] for s in batch], k=n_retrieve)
        pairs = [(sample, items) for sample, items in zip(batch, retrieved) if items]
        if len(pairs) < len(batch):
            logger.warning(
                "Step %d: %d/%d samples had no retrieved experience",
                step, len(batch) - len(pairs), len(batch),
            )
        if not pairs:
            continue

        prepared = [
            (sample, items, build_supervised_inputs(
                sample["trajectory"], tokenizer, enable_thinking=enable_thinking, max_seq_len=max_seq_len,
                n_latent=len(items) * qformer.num_queries, protocol=protocol,
            ))
            for sample, items in pairs
        ]
        supervised_tokens = sum(int((targets != IGNORE_INDEX).sum()) for _, _, (_, _, targets) in prepared)
        if supervised_tokens == 0:
            logger.warning("Step %d: no supervised assistant tokens, skipping", step)
            continue

        qformer_optimizer.zero_grad()
        lora_optimizer.zero_grad()
        loss_total = 0.0
        tokens_total = 0
        for start in range(0, len(prepared), train_mini_batch_size):
            mini_batch = prepared[start:start + train_mini_batch_size]
            latents = [
                compress_texts_via_pytorch(
                    model,
                    qformer,
                    tuple(
                        trajectory_text(item, tokenizer, enable_thinking=enable_thinking)
                        for item in items
                    ),
                    tokenizer,
                    str(embed_device),
                    train_encoder_lora=True,
                    task_text=sample["query_text"] if task_cond else None,
                    protocol=protocol,
                )
                for sample, items, _ in mini_batch
            ]
            loss_sum, n_tokens = _mini_batch_nll(
                backbone, [inputs for _, _, inputs in mini_batch], latents, device=embed_device,
            )
            if n_tokens == 0:
                continue
            (loss_sum / supervised_tokens).backward()
            loss_total += loss_sum.item()
            tokens_total += n_tokens
            del latents, loss_sum

        if tokens_total == 0:
            logger.warning("Step %d: no valid mini-batch, skipping optimizer step", step)
            continue

        qformer_grad_norm = torch.nn.utils.clip_grad_norm_(qformer.parameters(), qformer_grad_clip)
        lora_grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, lora_grad_clip)
        qformer_optimizer.step()
        lora_optimizer.step()

        mean_loss = loss_total / tokens_total
        logger.info(
            "Step %d: loss=%.4f trajectories=%d tokens=%d qformer_grad_norm=%.4f lora_grad_norm=%.4f",
            step, mean_loss, len(pairs), tokens_total, float(qformer_grad_norm), float(lora_grad_norm),
        )
        if writer is not None:
            writer.add_scalar("sft/loss", mean_loss, step)
            writer.add_scalar("sft/tokens", tokens_total, step)
            writer.add_scalar("sft/qformer_grad_norm", float(qformer_grad_norm), step)
            writer.add_scalar("sft/lora_grad_norm", float(lora_grad_norm), step)
            writer.flush()

        completed = step + 1
        checkpoint_dirs = [os.path.join(run_dir, LATEST_DIR)]
        if completed % save_every == 0:
            checkpoint_dirs.append(os.path.join(run_dir, f"step_{completed}"))
        lora_optimizer_state = optimizer_state_to_cpu_by_name(lora_optimizer, lora_named_params)
        for ckpt_dir in checkpoint_dirs:
            save_compressor(ckpt_dir, qformer, model, step=completed, manifest=manifest)
            save_train_state(
                ckpt_dir,
                qformer_optimizer=qformer_optimizer,
                lora_optimizer_state=lora_optimizer_state,
                epoch=epoch, position=position, seed=seed,
            )
        logger.info("Saved checkpoint(s): %s", ", ".join(checkpoint_dirs))

    if writer is not None:
        writer.close()


def _mini_batch_nll(
    backbone,
    inputs: list[tuple[list[int], list[int], torch.Tensor]],
    latents: list[torch.Tensor],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    embeds_list: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []
    for (input_ids, positions, target_ids), latent in zip(inputs, latents):
        ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            embeds = backbone.get_input_embeddings()(ids)
        embeds_list.append(inject_latent_tokens_into_embeds(embeds, positions, latent))
        targets_list.append(target_ids.to(device))

    embeds, attention_mask, targets = _pad_batch(embeds_list, targets_list)
    logits = backbone(inputs_embeds=embeds, attention_mask=attention_mask, use_cache=False).logits

    shift_logits = logits[:, :-1, :]
    shift_targets = targets[:, 1:].to(shift_logits.device)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = shift_targets != IGNORE_INDEX
    return -(token_log_probs * mask).sum(), int(mask.sum().item())


def _pad_batch(
    embeds_list: list[torch.Tensor],
    targets_list: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(e.shape[1] for e in embeds_list)
    padded_embeds, masks, padded_targets = [], [], []
    for embeds, targets in zip(embeds_list, targets_list):
        pad = max_len - embeds.shape[1]
        padded_embeds.append(torch.nn.functional.pad(embeds, (0, 0, 0, pad)))
        padded_targets.append(torch.nn.functional.pad(targets, (0, pad), value=IGNORE_INDEX))
        mask = torch.zeros(max_len, dtype=torch.long, device=embeds.device)
        mask[: embeds.shape[1]] = 1
        masks.append(mask)
    return torch.cat(padded_embeds), torch.stack(masks), torch.stack(padded_targets)


def _parse_json_cell(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return json.loads(value) if value.strip() else None
    raise ValueError("the trajectory column must hold JSON strings")
