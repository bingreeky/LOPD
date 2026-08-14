
import argparse
import asyncio
import os
import sys

from config import load_config, build_llm_from_config
from utils.store import RunLock


class _DefaultLoRALLM:
    def __init__(self, llm, *, adapter_name: str):
        self._llm = llm
        self._adapter_name = adapter_name

    def __getattr__(self, name):
        return getattr(self._llm, name)

    async def __call__(self, messages, tools=None, **kwargs):
        kwargs.setdefault("lora_path", self._adapter_name)
        return await self._llm(messages, tools=tools, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="LOPD inference")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--result_dir", help="Override result directory")
    parser.add_argument("--model_path", default=None, help="Override llm.model_path")
    return parser.parse_args()


async def main():
    args = parse_args()
    config = load_config(args.config)

    if args.model_path:
        config.setdefault("llm", {})["model_path"] = args.model_path

    run_cfg = config.get("run", {})
    result_dir = args.result_dir or run_cfg.get("result_dir", "results")
    run_name = run_cfg.get("run_name")
    run_lock = RunLock(os.path.join(result_dir, run_name)).acquire() if run_name else None

    datasets_cfg = config.get("datasets", {})
    if not isinstance(datasets_cfg, dict) or not datasets_cfg:
        print("ERROR: 'datasets' must be a non-empty dict in config")
        sys.exit(1)

    llm = build_llm_from_config(config)

    lora_cfg = config.get("lora", {})
    if lora_cfg.get("enabled") and lora_cfg.get("adapter_path"):
        adapter_name = lora_cfg.get("adapter_name", "encoder")
        pinned = bool(lora_cfg.get("pinned", True))
        if hasattr(llm, "async_load_lora_adapter"):
            await llm.async_load_lora_adapter(adapter_name, lora_cfg["adapter_path"], pinned=pinned)
        elif hasattr(llm, "load_lora_adapter"):
            llm.load_lora_adapter(adapter_name, lora_cfg["adapter_path"], pinned=pinned)
        else:
            raise RuntimeError("LoRA adapter configured but LLM does not support it")
        llm = _DefaultLoRALLM(llm, adapter_name=adapter_name)

    for dataset_name, dataset_cfg in datasets_cfg.items():
        if dataset_name == "envscaler":
            from envs.envscaler.inference import inference
            await inference(llm=llm, envscaler_cfg=dataset_cfg, result_dir=result_dir)
        else:
            print(f"WARNING: dataset '{dataset_name}' not yet supported, skipping")


if __name__ == "__main__":
    asyncio.run(main())
