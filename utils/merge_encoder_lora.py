
import argparse
import os
import shutil

import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
try:
    import peft.tuners.lora.torchao as _tao
    _tao.is_torchao_available = lambda: False
except Exception:
    pass

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Merge encoder LoRA into base model")
    parser.add_argument("--base", required=True, help="Base model path")
    parser.add_argument("--adapter", required=True, help="LoRA adapter path")
    parser.add_argument("--output", required=True, help="Output merged model path")
    args = parser.parse_args()

    print(f"Loading base model: {args.base}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="cpu",
    )

    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter, adapter_name="encoder")

    print("Merging...")
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    print(f"Saving to: {args.output}")
    model.save_pretrained(args.output)
    AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(args.output)

    for f in ["config.json", "preprocessor_config.json"]:
        src = os.path.join(args.base, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output, f))

    print("Done.")


if __name__ == "__main__":
    main()
