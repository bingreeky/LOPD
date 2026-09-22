import json
import logging
import os

import yaml

from backends import BaseLLM, SGLangLLM, VLLMLlm

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_llm_from_config(
    config: dict,
    *,
    encoder_adapter: tuple[str, str] | None = None,
    latent_prompt_protocol: str | None = None,
) -> BaseLLM:
    cfg = config["llm"]
    llm_type = cfg.get("type", "sglang")
    lora_kwargs = _engine_lora_kwargs(encoder_adapter)

    if llm_type == "sglang":
        return SGLangLLM(
            model_path=cfg["model_path"],
            latent_prompt_protocol=latent_prompt_protocol,
            temperature=cfg.get("temperature", 0.6),
            max_tokens=cfg.get("max_tokens", 4096),
            max_retries=cfg.get("max_retries", 2),
            enable_thinking=cfg.get("enable_thinking"),
            top_p=cfg.get("top_p", 0.95),
            top_k=cfg.get("top_k", 20),
            tp_size=cfg.get("tp_size", 1),
            dtype=cfg.get("dtype", "bfloat16"),
            mem_fraction_static=cfg.get("mem_fraction_static", 0.8),
            disable_radix_cache=cfg.get("disable_radix_cache", True),
            enable_memory_saver=cfg.get("enable_memory_saver", False),
            enable_return_hidden_states=cfg.get("enable_return_hidden_states", False),
            **lora_kwargs,
            enable_deterministic_inference=cfg.get("enable_deterministic_inference", False),
            sampling_seed=cfg.get("sampling_seed", None),
        )

    if llm_type == "vllm":
        if VLLMLlm is None:
            raise ImportError("vllm is not installed. Install it with: pip install vllm")
        return VLLMLlm(
            model_path=cfg["model_path"],
            latent_prompt_protocol=latent_prompt_protocol,
            temperature=cfg.get("temperature", 0.6),
            max_tokens=cfg.get("max_tokens", 4096),
            max_retries=cfg.get("max_retries", 2),
            enable_thinking=cfg.get("enable_thinking"),
            top_p=cfg.get("top_p", 0.95),
            top_k=cfg.get("top_k", 20),
            min_p=cfg.get("min_p", 0.0),
            sampling_seed=cfg.get("sampling_seed"),
            tp_size=cfg.get("tp_size", 1),
            dp_size=cfg.get("dp_size", 1),
            dtype=cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=cfg.get("gpu_memory_utilization", 0.85),
            max_model_len=cfg.get("max_model_len", None),
            max_num_seqs=cfg.get("max_num_seqs"),
            enforce_eager=cfg.get("enforce_eager", False),
            enable_prefix_caching=cfg.get("enable_prefix_caching", False),
            enable_chunked_prefill=cfg.get("enable_chunked_prefill"),
            distributed_executor_backend=cfg.get("distributed_executor_backend"),
            enable_sleep_mode=cfg.get("enable_sleep_mode", False),
            weight_transfer_backend=cfg.get("weight_transfer_backend"),
            lifecycle_drain_timeout_s=cfg.get("lifecycle_drain_timeout_s", 300),
            enable_lora=lora_kwargs.get("enable_lora", False),
            max_lora_rank=lora_kwargs.get("max_lora_rank"),
            lora_paths=lora_kwargs.get("lora_paths"),
            encoder_device=cfg.get("encoder_device", "cuda:0"),
        )

    raise ValueError(f"Unsupported llm.type: {llm_type!r}. Supported: sglang, vllm")


def _engine_lora_kwargs(encoder_adapter: tuple[str, str] | None) -> dict:
    if encoder_adapter is None:
        return {}
    name, adapter_dir = encoder_adapter
    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as f:
        adapter_config = json.load(f)
    return {
        "enable_lora": True,
        "max_lora_rank": int(adapter_config["r"]),
        "lora_target_modules": list(adapter_config["target_modules"]),
        "lora_paths": [{"lora_name": name, "lora_path": adapter_dir, "pinned": True}],
    }


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
