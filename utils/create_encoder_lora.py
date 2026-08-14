
import argparse
import json
import os

import torch


def main():
    parser = argparse.ArgumentParser(description="Create a fresh encoder LoRA adapter")
    parser.add_argument("--base_model", required=True, help="Base model path")
    parser.add_argument("--output", required=True, help="Output adapter directory")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adapter_name", default="encoder")
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"])
    args = parser.parse_args()

    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
    try:
        import peft.tuners.lora.torchao as _tao
        _tao.is_torchao_available = lambda: False
    except Exception:
        pass

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map="cpu",
    )

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print(f"Attaching LoRA: rank={args.rank}, alpha={args.alpha}, "
          f"modules={args.target_modules}")
    peft_model = get_peft_model(model, lora_config, adapter_name=args.adapter_name)

    adapter_dir = os.path.join(args.output, args.adapter_name)
    os.makedirs(adapter_dir, exist_ok=True)
    peft_model.save_pretrained(args.output)

    num_params = sum(
        p.numel() for n, p in peft_model.named_parameters() if "lora_" in n
    )
    meta = {
        "base_model": args.base_model,
        "adapter_name": args.adapter_name,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "target_modules": args.target_modules,
        "num_lora_parameters": num_params,
    }
    with open(os.path.join(args.output, "adapter_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {args.output}")
    print(f"  LoRA parameters: {num_params:,}")
    print("Done.")


if __name__ == "__main__":
    main()
