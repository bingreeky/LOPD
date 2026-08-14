
import argparse
import json
import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Build a FAISS memory bank")
    parser.add_argument("--trajectories", required=True, nargs="+",
                        help="Parquet file(s) with columns: task_id, instruction, reward, trajectory")
    parser.add_argument("--encoder_model", required=True,
                        help="Sentence-transformer model name or path")
    parser.add_argument("--output", required=True,
                        help="Output directory for the memory bank")
    parser.add_argument("--min_reward", type=float, default=0.0,
                        help="Only include trajectories with reward >= this value (default: 0.0)")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for the sentence encoder (default: cuda:0)")
    parser.add_argument("--build_query_cache", action="store_true",
                        help="Pre-compute query embeddings for all entries")
    return parser.parse_args()


def main():
    args = parse_args()

    from memory.bank import MemoryBank

    bank = MemoryBank(encoder_model=args.encoder_model, device=args.device)

    total_loaded = 0
    total_filtered = 0
    for parquet_path in args.trajectories:
        df = pd.read_parquet(parquet_path)
        for _, row in df.iterrows():
            total_loaded += 1
            reward = float(row.get("reward", 0))
            if reward < args.min_reward:
                total_filtered += 1
                continue
            trajectory = row.get("trajectory", "[]")
            if isinstance(trajectory, str):
                trajectory = json.loads(trajectory)
            bank.add(
                instruction=str(row.get("instruction", "")),
                trajectory=trajectory,
                success=reward >= 0.8,
            )

    logger.info("Loaded %d trajectories, filtered %d (reward < %.2f), kept %d",
                total_loaded, total_filtered, args.min_reward, len(bank.entries))

    if not bank.entries:
        logger.error("No entries to index. Check --min_reward or input data.")
        return

    logger.info("Encoding and building FAISS index...")
    bank.build_index()

    if args.build_query_cache:
        query_texts = [e["query_text"] for e in bank.entries]
        bank.build_query_cache(query_texts, args.output)

    bank.save(args.output)
    logger.info("Saved memory bank to %s (%d entries)", args.output, len(bank.entries))


if __name__ == "__main__":
    main()
