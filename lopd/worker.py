from __future__ import annotations

import functools
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def phase_b_worker(
    rank: int,
    world_size: int,
    ipc_dir: str,
    model_path: str,
    student_dir: str,
    compressor_dir: str,
    cold_start_dir: str,
    anchor_cache_dir: str,
    latest_dir: str,
    workers_port: int,
) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(workers_port), RANK=str(rank),
                      WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s rank{rank} %(name)s: %(message)s")

    import torch
    import torch.distributed as dist
    from lopd import ipc

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    try:
        rollouts, cfg = ipc.read_input_shard(Path(ipc_dir), rank, world_size)
        support = _teacher_phase(rollouts, cfg, model_path=model_path, compressor_dir=compressor_dir, device=device)
        metrics, student_outputs = _student_phase(
            rollouts, support, cfg, model_path=model_path, student_dir=student_dir,
            latest_dir=latest_dir, rank=rank, world_size=world_size, device=device,
        )
        torch.cuda.empty_cache()
        if cfg["constraint"]["enabled"] and not metrics.get("skip"):
            metrics.update(_composer_phase(
                rollouts, student_outputs, cfg, model_path=model_path, compressor_dir=compressor_dir,
                cold_start_dir=cold_start_dir, anchor_cache_dir=anchor_cache_dir, latest_dir=latest_dir, rank=rank, device=device,
            ))
        if rank == 0:
            ipc.write_metrics(Path(ipc_dir), metrics)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _teacher_phase(rollouts, cfg, *, model_path, compressor_dir, device):
    import torch
    from lopd.teacher import LatentTeacher

    t0 = time.time()
    teacher = LatentTeacher(model_path, compressor_dir, device=str(device), enable_thinking=cfg["enable_thinking"])
    support = [
        None if record["failed"] else teacher.topk_log_probs(record, top_k=cfg["top_k"], max_seq_len=cfg["max_seq_len"])
        for record in rollouts
    ]
    del teacher
    torch.cuda.empty_cache()
    logger.info("teacher phase: %d rollouts in %.1fs", len(rollouts), time.time() - t0)
    return support


def supervised_logits_model(lm):
    import torch.nn as nn

    class SupervisedLogits(nn.Module):
        def __init__(self, lm):
            super().__init__()
            self.lm = lm

        def forward(self, input_ids, sup_mask=None):
            hidden = self.lm.model(input_ids=input_ids, use_cache=False).last_hidden_state[0]
            if sup_mask is None:
                return self.lm.lm_head(hidden)
            return self.lm.lm_head(hidden[:-1][sup_mask])

    return SupervisedLogits(lm)


def student_loss(model, tokenizer, record, teacher_entry, *, enable_thinking, max_seq_len, device):
    import torch
    import torch.nn.functional as F
    from lopd.loss import topk_reverse_kl
    from memory.serialization import IGNORE_INDEX, build_supervised_inputs

    teacher_values, teacher_indices = teacher_entry
    input_ids, _, target_ids = build_supervised_inputs(record["trajectory"], tokenizer, enable_thinking=enable_thinking, max_seq_len=max_seq_len)
    sup_mask = (target_ids[1:] != IGNORE_INDEX).to(device)
    if not sup_mask.any():
        return None, 0, None

    ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    log_probs = F.log_softmax(model(input_ids=ids, sup_mask=sup_mask).float(), dim=-1)
    n = min(log_probs.shape[0], teacher_values.shape[0])
    log_probs, teacher_values, teacher_indices = log_probs[:n], teacher_values[:n].to(device), teacher_indices[:n].to(device)
    student_topk = log_probs.gather(-1, teacher_indices)
    loss_sum, n_tokens = topk_reverse_kl(student_topk, teacher_values)

    next_tokens = torch.tensor(input_ids[1:], dtype=torch.long, device=device)[sup_mask][:n]
    outputs = {
        "topk_lp": student_topk.detach().cpu(),
        "indices": teacher_indices.cpu(),
        "lp_at_token": log_probs.detach().gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1).cpu(),
    }
    return loss_sum, n_tokens, outputs


def _student_phase(rollouts, support, cfg, *, model_path, student_dir, latest_dir, rank, world_size, device):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_from = student_dir if os.path.isfile(os.path.join(student_dir, "config.json")) else model_path
    t0 = time.time()
    lm = AutoModelForCausalLM.from_pretrained(
        load_from, torch_dtype=torch.float32, trust_remote_code=True, attn_implementation="flash_attention_2",
    )
    lm.train()
    if cfg["gradient_checkpointing"]:
        lm.gradient_checkpointing_enable()
        lm.config.use_cache = False
    decoder_layer = type(lm.model.layers[0])
    model = FSDP(
        supervised_logits_model(lm),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={decoder_layer}),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16),
        device_id=rank,
        use_orig_params=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    shards_dir = os.path.join(student_dir, "optim_shards")
    if os.path.isdir(shards_dir):
        shard_path = os.path.join(shards_dir, f"rank_{rank}_of_{world_size}.pt")
        if not os.path.isfile(shard_path):
            raise FileNotFoundError(f"{shard_path} is missing; the checkpoint was saved with a different nproc")
        optimizer.load_state_dict(torch.load(shard_path, map_location=device, weights_only=True))
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logger.info("student phase: loaded %s in %.1fs", load_from, time.time() - t0)

    local_tokens = sum(0 if entry is None else int(entry[0].shape[0]) for entry in support)
    global_tokens = torch.tensor([local_tokens], device=device)
    dist.all_reduce(global_tokens, op=dist.ReduceOp.SUM)
    global_tokens = int(global_tokens.item())
    if global_tokens == 0:
        logger.warning("student phase: no supervised tokens in this step, skipping")
        return {"loss": 0.0, "n_tokens": 0, "grad_norm": 0.0, "skip": True}, [None] * len(rollouts)

    def dummy_step():
        out = model(input_ids=torch.tensor([[0]], dtype=torch.long, device=device))
        (out.sum() * 0.0).backward()

    optimizer.zero_grad()
    loss_total, tokens_total, outputs = 0.0, 0, []
    for record, entry in zip(rollouts, support):
        loss_sum, n_tokens, out = (None, 0, None) if entry is None else student_loss(
            model, tokenizer, record, entry, enable_thinking=cfg["enable_thinking"], max_seq_len=cfg["max_seq_len"], device=device,
        )
        if loss_sum is None:
            dummy_step()
        else:
            (loss_sum * world_size / global_tokens).backward()
            loss_total += float(loss_sum.detach())
            tokens_total += n_tokens
        outputs.append(out)

    grad_norm = model.clip_grad_norm_(cfg["grad_clip"])
    finite = bool(torch.isfinite(grad_norm))
    if finite:
        optimizer.step()
    stats = torch.tensor([loss_total, float(tokens_total)], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)

    _save_student(model, optimizer, model_path=model_path, student_dir=student_dir, latest_dir=latest_dir, rank=rank, world_size=world_size)
    metrics = {"loss": float(stats[0] / max(stats[1], 1)), "n_tokens": int(stats[1]),
               "grad_norm": float(grad_norm) if finite else float("nan"), "skip": not finite}
    return metrics, outputs


def _save_student(model, optimizer, *, model_path, student_dir, latest_dir, rank, world_size):
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

    tmp = student_dir + ".tmp"
    if rank == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(os.path.join(tmp, "optim_shards"))
    dist.barrier()
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = model.state_dict()
    if rank == 0:
        write_hf_directory(state, model_path, tmp)
    torch.save(optimizer.state_dict(), os.path.join(tmp, "optim_shards", f"rank_{rank}_of_{world_size}.pt"))
    dist.barrier()
    if rank == 0:
        os.makedirs(latest_dir, exist_ok=True)
        shutil.rmtree(student_dir, ignore_errors=True)
        os.rename(tmp, student_dir)
    dist.barrier()


def write_hf_directory(state: dict, model_path: str, out_dir: str) -> None:
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(model_path):
        src = os.path.join(model_path, name)
        if os.path.isfile(src) and not name.endswith((".safetensors", ".bin", ".pt", ".index.json")):
            shutil.copy(src, os.path.join(out_dir, name))
    cleaned, seen = {}, {}
    for key, value in state.items():
        for prefix in ("_fsdp_wrapped_module.", "lm."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        value = value.detach().cpu().contiguous()
        if seen.setdefault((value.data_ptr(), tuple(value.shape)), key) == key:
            cleaned[key] = value
    save_file(cleaned, os.path.join(out_dir, "model.safetensors"))


def composer_loss(teacher, record, student_out, *, beta, constraint, max_seq_len):
    from lopd.loss import topk_reverse_kl

    out = teacher.privileged_log_probs(record, student_out["indices"], max_seq_len=max_seq_len)
    device = out["support_lp"].device
    n = min(out["support_lp"].shape[0], student_out["topk_lp"].shape[0])
    kl_sum, n_tokens = topk_reverse_kl(student_out["topk_lp"][:n].to(device), out["support_lp"][:n])
    kl = kl_sum / max(n_tokens, 1)
    advantage = 2.0 * float(record["reward"]) - 1.0
    delta = advantage * (out["lp_at_token"][:n] - student_out["lp_at_token"][:n].to(device)).mean()
    anchor = (out["c_phi"] - out["c_phi0"]).pow(2).mean()
    loss = kl + beta * (constraint["margin_m"] - delta) + constraint["anchor_lambda"] * anchor
    return loss, (float(delta.detach()), n_tokens, float(kl.detach()), float(anchor.detach()))


def _composer_phase(rollouts, student_outputs, cfg, *, model_path, compressor_dir, cold_start_dir, anchor_cache_dir, latest_dir, rank, device):
    import torch
    import torch.distributed as dist
    from lopd.teacher import LatentTeacher
    from memory.compressor import save_compressor
    from memory.lora import load_optimizer_state_by_name, optimizer_state_to_cpu_by_name

    constraint = cfg["constraint"]
    beta = float(constraint["beta"])
    t0 = time.time()
    teacher = LatentTeacher(model_path, compressor_dir, device=str(device), enable_thinking=cfg["enable_thinking"],
                            trainable=True, cold_start_dir=cold_start_dir, anchor_cache_dir=anchor_cache_dir)
    phi_named = teacher.named_parameters()
    phi_params = [p for _, p in phi_named]
    phi_optimizer = torch.optim.AdamW(phi_params, lr=constraint["lr_phi"])
    phi_state = os.path.join(compressor_dir, "phi_optimizer.pt")
    if os.path.isfile(phi_state):
        load_optimizer_state_by_name(phi_optimizer, phi_named, torch.load(phi_state, map_location="cpu", weights_only=True))
    phi_optimizer.zero_grad(set_to_none=True)

    active = [(record, out) for record, out in zip(rollouts, student_outputs) if out is not None]
    delta_weighted, tokens, kl_total, anchor_total = 0.0, 0, 0.0, 0.0
    for record, out in active:
        loss, (delta_i, n_tokens, kl_i, anchor_i) = composer_loss(
            teacher, record, out, beta=beta, constraint=constraint, max_seq_len=cfg["max_seq_len"],
        )
        (loss / max(len(active), 1)).backward()
        delta_weighted += delta_i * n_tokens
        tokens += n_tokens
        kl_total += kl_i
        anchor_total += anchor_i

    for param in phi_params:
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    stats = torch.tensor([delta_weighted, float(tokens), kl_total, anchor_total, float(len(active))], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    delta_sum, n_tokens, kl_sum, anchor_sum, n_used = stats.tolist()
    delta = delta_sum / n_tokens if n_tokens else 0.0

    grad_norm = torch.nn.utils.clip_grad_norm_(phi_params, cfg["grad_clip"])
    if torch.isfinite(grad_norm):
        phi_optimizer.step()
    new_beta = max(0.0, beta + constraint["eta_beta"] * (constraint["margin_m"] - delta))

    if rank == 0:
        out_dir = os.path.join(latest_dir, "compressor")
        tmp = out_dir + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        save_compressor(tmp, teacher.qformer, teacher.model, step=cfg["step"] + 1, manifest={
            "prompt_protocol": teacher.protocol.name, "adapter_name": teacher.adapter_name, "task_cond": teacher.task_cond,
        })
        torch.save(optimizer_state_to_cpu_by_name(phi_optimizer, phi_named), os.path.join(tmp, "phi_optimizer.pt"))
        shutil.rmtree(out_dir, ignore_errors=True)
        os.rename(tmp, out_dir)
    dist.barrier()
    logger.info("composer phase: %d rollouts in %.1fs", int(n_used), time.time() - t0)
    return {"beta": new_beta, "delta": delta, "kl_phi": kl_sum / max(n_used, 1.0),
            "anchor": anchor_sum / max(n_used, 1.0), "phi_grad_norm": float(grad_norm)}
