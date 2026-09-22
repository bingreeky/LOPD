import argparse
import logging

from backends.event_loop import run_async
from configs.config import build_llm_from_config, build_memory_bank_from_config, load_config
from envs.envscaler.inference_with_latent import inference_with_latent
from memory.compressor import load_compressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EnvScaler evaluation with latent memory")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--result_dir", default=None, help="Override run.result_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    inference_cfg = config["inference"]
    result_dir = args.result_dir or config["run"]["result_dir"]
    if not config["llm"].get("enable_return_hidden_states"):
        raise ValueError("llm.enable_return_hidden_states must be true: experiences are encoded through the engine")

    checkpoint = config["compressor"]["checkpoint"]
    logger.info("Loading compressor from %s", checkpoint)
    qformer, adapter_dir, manifest = load_compressor(checkpoint, device=inference_cfg["qformer_device"])
    adapter_name = manifest["adapter_name"]

    llm = build_llm_from_config(
        config, encoder_adapter=(adapter_name, adapter_dir), latent_prompt_protocol=manifest["prompt_protocol"],
    )
    try:
        memory_bank = build_memory_bank_from_config(config)
        run_async(inference_with_latent(
            llm, qformer, memory_bank, config["datasets"]["envscaler"],
            n_retrieve=int(inference_cfg["n_retrieve"]),
            result_dir=result_dir,
            encoder_lora_name=adapter_name,
            enable_thinking=bool(config["llm"]["enable_thinking"]),
            task_cond=bool(inference_cfg.get("task_cond", manifest["task_cond"])),
            prompt_protocol=manifest["prompt_protocol"],
        ))
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
