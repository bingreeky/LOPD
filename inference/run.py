import argparse
import logging

from backends.event_loop import run_async
from configs.config import build_llm_from_config, load_config
from envs.envscaler.inference import inference

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EnvScaler evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--result_dir", default=None, help="Override run.result_dir")
    parser.add_argument("--model_path", default=None, help="Override llm.model_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.model_path:
        config["llm"]["model_path"] = args.model_path
    result_dir = args.result_dir or config["run"]["result_dir"]

    llm = build_llm_from_config(config)
    try:
        run_async(inference(llm, config["datasets"]["envscaler"], result_dir=result_dir))
    finally:
        llm.shutdown()


if __name__ == "__main__":
    main()
