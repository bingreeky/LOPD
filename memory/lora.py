from __future__ import annotations

import os
from typing import Iterable, Iterator

import torch

def create_lora_config(*, rank: int, alpha: int, dropout: float, target_modules: Iterable[str]):
    from peft import LoraConfig

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def attach_lora(model, *, adapter_name: str, config=None, adapter_path: str | None = None, is_trainable: bool = True):
    if (config is None) == (adapter_path is None):
        raise ValueError("attach_lora takes exactly one of config or adapter_path")
    if adapter_path is not None:
        from peft import PeftModel

        return PeftModel.from_pretrained(model, adapter_path, adapter_name=adapter_name, is_trainable=is_trainable)

    from peft import get_peft_model

    peft_model = get_peft_model(model, config, adapter_name=adapter_name)
    peft_model.set_adapter(adapter_name)
    return peft_model


def resolve_peft_adapter_dir(path: str, adapter_name: str) -> str:
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    nested = os.path.join(path, adapter_name)
    if os.path.isfile(os.path.join(nested, "adapter_config.json")):
        return nested
    raise FileNotFoundError(
        f"Could not find adapter_config.json in {path!r} or {nested!r}"
    )


def save_lora_adapter(model, path: str, adapter_name: str) -> None:
    model.save_pretrained(path, selected_adapters=[adapter_name])


def iter_lora_named_parameters(model) -> Iterator[tuple[str, torch.nn.Parameter]]:
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            yield name, param


def base_model_view(peft_model):
    base = peft_model.get_base_model()
    with torch.device("meta"):
        view = type(base)(base.config)
    params = dict(base.named_parameters(remove_duplicate=False))
    buffers = dict(base.named_buffers(remove_duplicate=False))
    for name, _ in list(view.named_parameters(remove_duplicate=False)):
        module_name, _, attr = name.rpartition(".")
        wrapped = f"{module_name}.base_layer.{attr}"
        _set_tensor(view, name, params[wrapped] if wrapped in params else params[name])
    for name, _ in list(view.named_buffers(remove_duplicate=False)):
        _set_tensor(view, name, buffers[name])
    view.train(base.training)
    if base.is_gradient_checkpointing:
        view.gradient_checkpointing_enable()
    device_map = getattr(base, "hf_device_map", None)
    if device_map and len(set(device_map.values())) > 1:
        from accelerate import dispatch_model

        dispatch_model(view, device_map=device_map)
    return view


def _set_tensor(module: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    parent_name, _, attr = name.rpartition(".")
    setattr(module.get_submodule(parent_name) if parent_name else module, attr, tensor)


def optimizer_state_to_cpu_by_name(
    optimizer: torch.optim.Optimizer,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for name, param in named_parameters:
        state_cpu = {}
        for key, value in optimizer.state.get(param, {}).items():
            state_cpu[key] = value.detach().cpu().clone() if torch.is_tensor(value) else value
        result[name] = {"shape": list(param.shape), "state": state_cpu}
    return result


def load_optimizer_state_by_name(
    optimizer: torch.optim.Optimizer,
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    state_by_name: dict[str, dict],
) -> None:
    named = list(named_parameters)
    missing = [name for name, _ in named if name not in state_by_name]
    unexpected = sorted(set(state_by_name) - {name for name, _ in named})
    if missing or unexpected:
        raise KeyError(
            "LoRA optimizer state names do not match: "
            f"missing={missing}, unexpected={unexpected}"
        )

    for name, param in named:
        saved = state_by_name[name]
        if list(saved["shape"]) != list(param.shape):
            raise ValueError(
                f"Optimizer state shape mismatch for {name}: "
                f"saved={saved['shape']}, current={list(param.shape)}"
            )
        optimizer.state[param] = {
            key: value.to(device=param.device) if torch.is_tensor(value) else value
            for key, value in saved["state"].items()
        }
