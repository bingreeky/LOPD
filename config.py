import logging

import yaml

from backends import BaseLLM, SGLangLLM

try:
    from backends import VLLMLlm
except ImportError:
    VLLMLlm = None

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    candidate_paths = [path]
    if "/" not in path and "\\" not in path:
        candidate_paths.extend([
            f"configs/{path}",
            f"configs/inference/{path}",
            f"configs/training/{path}",
        ])

    for candidate in candidate_paths:
        try:
            with open(candidate) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"Config file not found: {path}. Tried: {candidate_paths}"
    )


def build_llm_from_config(config: dict) -> BaseLLM:
    cfg = config["llm"]
    llm_type = cfg.get("type", "sglang")

    if llm_type == "sglang":
        lora_cfg = config.get("lora", {})
        return SGLangLLM(
            model_path=cfg["model_path"],
            temperature=cfg.get("temperature", 0.6),
            max_tokens=cfg.get("max_tokens", 4096),
            max_retries=cfg.get("max_retries", 2),
            enable_thinking=cfg.get("enable_thinking"),
            top_p=cfg.get("top_p", 0.95),
            top_k=cfg.get("top_k", 20),
            min_p=cfg.get("min_p", 0.0),
            tp_size=cfg.get("tp_size", 1),
            dp_size=cfg.get("dp_size", 1),
            dtype=cfg.get("dtype", "bfloat16"),
            mem_fraction_static=cfg.get("mem_fraction_static", 0.8),
            disable_radix_cache=cfg.get("disable_radix_cache", True),
            enable_memory_saver=cfg.get("enable_memory_saver", False),
            enable_weights_cpu_backup=cfg.get("enable_weights_cpu_backup", False),
            enable_return_hidden_states=cfg.get("enable_return_hidden_states", False),
            enable_lora=cfg.get("enable_lora", bool(lora_cfg.get("enabled", False))),
            max_lora_rank=cfg.get("max_lora_rank", lora_cfg.get("rank")),
            lora_target_modules=cfg.get("lora_target_modules", lora_cfg.get("target_modules")),
            lora_paths=cfg.get("lora_paths"),
            max_loras_per_batch=cfg.get("max_loras_per_batch", lora_cfg.get("max_loras_per_batch", 4)),
            max_loaded_loras=cfg.get("max_loaded_loras", lora_cfg.get("max_loaded_loras", 4)),
            enable_deterministic_inference=cfg.get("enable_deterministic_inference", False),
            sampling_seed=cfg.get("sampling_seed", None),
            disable_cuda_graph=cfg.get("disable_cuda_graph", False),
            disable_custom_all_reduce=cfg.get("disable_custom_all_reduce", False),
        )

    if llm_type == "vllm":
        if VLLMLlm is None:
            raise ImportError("vllm is not installed. Install it with: pip install vllm")
        lora_cfg = config.get("lora", {})
        return VLLMLlm(
            model_path=cfg["model_path"],
            temperature=cfg.get("temperature", 0.6),
            max_tokens=cfg.get("max_tokens", 4096),
            max_retries=cfg.get("max_retries", 2),
            enable_thinking=cfg.get("enable_thinking"),
            top_p=cfg.get("top_p", 0.95),
            top_k=cfg.get("top_k", 20),
            min_p=cfg.get("min_p", 0.0),
            sampling_seed=cfg.get("sampling_seed"),
            stop=cfg.get("stop"),
            tp_size=cfg.get("tp_size", 1),
            dp_size=cfg.get("dp_size", 1),
            dtype=cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
            max_model_len=cfg.get("max_model_len", None),
            max_num_seqs=cfg.get("max_num_seqs"),
            enforce_eager=cfg.get("enforce_eager", False),
            enable_prefix_caching=cfg.get("enable_prefix_caching", False),
            enable_chunked_prefill=cfg.get("enable_chunked_prefill"),
            disable_custom_all_reduce=cfg.get("disable_custom_all_reduce", False),
            distributed_executor_backend=cfg.get("distributed_executor_backend"),
            compilation_config=cfg.get("compilation_config"),
            enable_sleep_mode=cfg.get("enable_sleep_mode", False),
            weight_transfer_backend=cfg.get("weight_transfer_backend"),
            lifecycle_drain_timeout_s=cfg.get("lifecycle_drain_timeout_s", 300),
            enable_lora=cfg.get("enable_lora", bool(lora_cfg.get("enabled", False))),
            max_lora_rank=cfg.get("max_lora_rank", lora_cfg.get("rank")),
            lora_paths=cfg.get("lora_paths"),
            max_loaded_loras=cfg.get(
                "max_loaded_loras", lora_cfg.get("max_loaded_loras", 4),
            ),
            encoder_device=cfg.get("encoder_device", "cuda:0"),
            encoder_dtype=cfg.get("encoder_dtype"),
        )

    raise ValueError(f"Unsupported llm.type: {llm_type!r}. Supported: sglang, vllm")


def build_compressor_from_config(config: dict, resume_path: str = None):
    from memory.compressor import (
        build_compressor_from_model_config,
        load_compressor_checkpoint,
    )

    model_path = config["model_path"]
    qf_cfg = config.get("qformer", {})

    ckpt = resume_path or qf_cfg.get("checkpoint")
    if ckpt:
        logger.info("Resuming compressor from %s", ckpt)
        qformer, step, dataset_offset = load_compressor_checkpoint(ckpt, device="cuda")
        return qformer.bfloat16(), step, dataset_offset

    logger.info("Creating compressor from model config: %s", model_path)
    qformer = build_compressor_from_model_config(model_path, qf_cfg)
    return qformer.bfloat16(), 0, 0


def build_memory_bank_from_config(config: dict):
    from memory.bank import MemoryBank

    bank_cfg = config.get("memory_bank", {})
    bank = MemoryBank(
        encoder_model=bank_cfg.get("encoder_model"),
        allow_self_retrieval=bool(bank_cfg.get("allow_self_retrieval", False)),
    )

    index_path = bank_cfg.get("index_path")
    if not index_path:
        raise ValueError("memory_bank.index_path is required")
    bank.load(index_path)
    logger.info("Loaded memory bank: %d entries", len(bank.entries))
    return bank
