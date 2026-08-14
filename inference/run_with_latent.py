
import argparse
import asyncio
import logging
import os
import sys
import threading

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


_persistent_loop: asyncio.AbstractEventLoop | None = None
_loop_thread = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _persistent_loop, _loop_thread
    if _persistent_loop is None or _persistent_loop.is_closed():
        _persistent_loop = asyncio.new_event_loop()

        def _run():
            asyncio.set_event_loop(_persistent_loop)
            _persistent_loop.run_forever()

        _loop_thread = threading.Thread(target=_run, daemon=True)
        _loop_thread.start()
    return _persistent_loop


def _run_async(coro):
    loop = _get_loop()
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with latent memory")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="QFormer checkpoint path")
    parser.add_argument("--result_dir", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    from config import (
        build_llm_from_config,
        build_compressor_from_config,
        build_memory_bank_from_config,
    )

    run_cfg = config.get("run", {})
    inference_cfg = config.get("inference", {})
    llm_cfg = config.get("llm", {})
    lora_cfg = dict(config.get("lora", {}))
    result_dir = args.result_dir or run_cfg.get("result_dir", "results")
    n_retrieve = inference_cfg.get("n_retrieve", 3)
    enable_thinking = llm_cfg.get("enable_thinking")
    task_cond = bool(inference_cfg.get("task_cond", False))
    prompt_protocol = inference_cfg.get("prompt_protocol", "envscaler_v1")

    preload_lora = bool(lora_cfg.get("preload", False))
    preloaded_lora_dir = None
    if bool(lora_cfg.get("enabled", False)) and preload_lora:
        encoder_lora_name = lora_cfg.get("adapter_name", "encoder")
        adapter_path = _resolve_lora_adapter_path(config, lora_cfg, args.resume)
        preloaded_lora_dir = _resolve_peft_adapter_dir(adapter_path, encoder_lora_name)
        llm_cfg["lora_paths"] = [{
            "lora_name": encoder_lora_name,
            "lora_path": preloaded_lora_dir,
            "pinned": bool(lora_cfg.get("pinned", True)),
        }]

    datasets_cfg = config.get("datasets", {})
    if "envscaler" not in datasets_cfg:
        print("ERROR: 'datasets.envscaler' block is required")
        sys.exit(1)

    logger.info("Loading LLM...")
    llm = build_llm_from_config(config)

    logger.info("Loading compressor...")
    qformer, step, _ = build_compressor_from_config(config, resume_path=args.resume)
    logger.info("Compressor loaded from step %d (%d params)",
                step, sum(p.numel() for p in qformer.parameters()))

    logger.info("Loading MemoryBank...")
    memory_bank = build_memory_bank_from_config(config)

    async def _main_async():
        encoder_lora_name = None
        if bool(lora_cfg.get("enabled", False)):
            encoder_lora_name = lora_cfg.get("adapter_name", "encoder")
            if preload_lora:
                resolved = preloaded_lora_dir
            else:
                adapter_path = _resolve_lora_adapter_path(config, lora_cfg, args.resume)
                pinned = bool(lora_cfg.get("pinned", True))
                resolved = await llm.async_load_lora_adapter(
                    encoder_lora_name, adapter_path, pinned=pinned,
                )
            print(f"  [LoRA] encoder LoRA: name={encoder_lora_name!r} "
                  f"path={resolved!r} preloaded={preload_lora}", flush=True)

        from envs.envscaler.inference_with_latent import inference_with_latent

        return await inference_with_latent(
            llm=llm,
            qformer=qformer,
            memory_bank=memory_bank,
            envscaler_cfg=datasets_cfg["envscaler"],
            n_retrieve=n_retrieve,
            device="cuda",
            result_dir=result_dir,
            encoder_lora_path=encoder_lora_name,
            enable_thinking=enable_thinking,
            task_cond=task_cond,
            prompt_protocol=prompt_protocol,
        )

    _run_async(_main_async())


def _resolve_lora_adapter_path(config: dict, lora_cfg: dict, resume_path: str | None) -> str:
    if resume_path and os.path.isdir(resume_path):
        resume_lora_adapter = os.path.join(resume_path, "lora_adapter")
        if os.path.exists(resume_lora_adapter):
            return resume_lora_adapter

    adapter_path = (
        lora_cfg.get("adapter_path")
        or lora_cfg.get("resume_path")
        or lora_cfg.get("checkpoint")
    )
    if adapter_path:
        return adapter_path

    qformer_ckpt = config.get("qformer", {}).get("checkpoint")
    if qformer_ckpt and os.path.isdir(qformer_ckpt):
        candidate = os.path.join(qformer_ckpt, "lora_adapter")
        if os.path.exists(candidate):
            return candidate

    raise ValueError(
        "lora.enabled=true requires lora.adapter_path, or a checkpoint directory "
        "containing lora_adapter via --resume or qformer.checkpoint."
    )


def _resolve_peft_adapter_dir(path: str, adapter_name: str) -> str:
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    nested = os.path.join(path, adapter_name)
    if os.path.isfile(os.path.join(nested, "adapter_config.json")):
        return nested
    raise FileNotFoundError(
        f"Could not find adapter_config.json in {path!r} or {nested!r}"
    )


if __name__ == "__main__":
    main()
