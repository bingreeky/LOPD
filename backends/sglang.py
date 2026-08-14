
import asyncio
import copy
import json
import re
import uuid
import concurrent.futures
import threading
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

try:
    from memory.serialization import build_latent_injected_ids
except ImportError:
    build_latent_injected_ids = None

from .base import BaseLLM

_DEFAULT_RETRIES = 2


def _load_embed_tokens_cpu(model_path: str) -> torch.nn.Embedding:
    import json as _json
    from pathlib import Path
    from safetensors.torch import load_file

    model_dir = Path(model_path)
    candidate_weight_keys = [
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    ]

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = _json.load(f)
        weight_map = index["weight_map"]
        weight_key = next((key for key in candidate_weight_keys if key in weight_map), None)
        if weight_key is None:
            raise KeyError(
                "No supported embed_tokens weight key found in safetensors index. "
                f"Tried: {candidate_weight_keys}"
            )
        shard_file = weight_map[weight_key]
        weight = load_file(str(model_dir / shard_file), device="cpu")[weight_key]
    else:
        single = model_dir / "model.safetensors"
        if single.exists():
            tensors = load_file(str(single), device="cpu")
            weight_key = next((key for key in candidate_weight_keys if key in tensors), None)
            if weight_key is None:
                raise KeyError(
                    "No supported embed_tokens weight key found in model.safetensors. "
                    f"Tried: {candidate_weight_keys}"
                )
            weight = tensors[weight_key]
        else:
            raise FileNotFoundError(f"No safetensors checkpoint found in {model_path}")

    vocab_size, hidden_size = weight.shape
    embedding = torch.nn.Embedding(vocab_size, hidden_size, _weight=weight.to(torch.bfloat16))
    embedding.requires_grad_(False)
    return embedding


class SGLangLLM(BaseLLM):

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.6,
        max_tokens: int = 4096,
        max_retries: int = _DEFAULT_RETRIES,
        enable_thinking: bool | None = None,
        top_p: float = 0.95,
        top_k: int = 20,
        min_p: float = 0.0,
        tp_size: int = 8,
        dp_size: int = 1,
        dtype: str = "bfloat16",
        mem_fraction_static: float = 0.8,
        disable_radix_cache: bool = True,
        enable_memory_saver: bool = True,
        enable_weights_cpu_backup: bool = False,
        enable_return_hidden_states: bool = False,
        enable_lora: bool = False,
        max_lora_rank: int | None = None,
        lora_target_modules: list[str] | None = None,
        lora_paths: list[dict] | dict[str, str] | list[str] | None = None,
        max_loras_per_batch: int = 4,
        max_loaded_loras: int = 4,
        enable_deterministic_inference: bool = False,
        sampling_seed: int | None = None,
        disable_cuda_graph: bool = False,
        disable_custom_all_reduce: bool = False,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.sampling_seed = sampling_seed

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )

        import sglang as sgl
        engine_kwargs = dict(
            model_path=model_path,
            tp_size=tp_size,
            dp_size=dp_size,
            dtype=dtype,
            mem_fraction_static=mem_fraction_static,
            trust_remote_code=True,
            disable_radix_cache=disable_radix_cache,
            enable_memory_saver=enable_memory_saver,
            enable_weights_cpu_backup=enable_weights_cpu_backup,
            enable_return_hidden_states=enable_return_hidden_states,
        )
        if disable_custom_all_reduce:
            engine_kwargs["disable_custom_all_reduce"] = True
        if disable_cuda_graph:
            engine_kwargs["disable_cuda_graph"] = True
        if enable_deterministic_inference:
            engine_kwargs["enable_deterministic_inference"] = True
        if enable_lora:
            if max_lora_rank is None:
                raise ValueError("llm.max_lora_rank or lora.rank is required when LoRA is enabled")
            if not lora_target_modules:
                raise ValueError("llm.lora_target_modules or lora.target_modules is required when LoRA is enabled")
            engine_kwargs.update(
                enable_lora=True,
                max_lora_rank=max_lora_rank,
                lora_target_modules=lora_target_modules,
                max_loras_per_batch=max_loras_per_batch,
                max_loaded_loras=max_loaded_loras,
            )
            if lora_paths:
                engine_kwargs["lora_paths"] = lora_paths
        try:
            self.engine = sgl.Engine(**engine_kwargs)
        except TypeError as exc:
            if "enable_deterministic_inference" not in engine_kwargs:
                raise
            print(
                "  [SGLangLLM] enable_deterministic_inference is not supported by "
                f"this SGLang build, continuing without it: {exc}",
                flush=True,
            )
            engine_kwargs.pop("enable_deterministic_inference", None)
            self.engine = sgl.Engine(**engine_kwargs)
        self._engine_loop_lock = threading.RLock()
        self._engine_loop_thread: threading.Thread | None = None

        self.embed_layer = _load_embed_tokens_cpu(model_path)


    def _ensure_engine_loop_running(self) -> None:
        loop = getattr(self.engine, "loop", None)
        if loop is None or loop.is_running():
            return
        if self._engine_loop_thread is not None and self._engine_loop_thread.is_alive():
            return

        started = threading.Event()

        def _run_loop():
            asyncio.set_event_loop(loop)
            started.set()
            loop.run_forever()

        self._engine_loop_thread = threading.Thread(
            target=_run_loop,
            name="sglang-engine-loop",
            daemon=True,
        )
        self._engine_loop_thread.start()
        started.wait(timeout=5.0)



    @staticmethod
    def _resolve_lora_adapter_path(path: str, adapter_name: str) -> str:
        adapter_dir = Path(path)
        if (adapter_dir / "adapter_config.json").exists():
            return str(adapter_dir)
        nested = adapter_dir / adapter_name
        if (nested / "adapter_config.json").exists():
            return str(nested)
        raise FileNotFoundError(
            f"Could not find adapter_config.json in {adapter_dir} or {nested}. "
            "Pass the PEFT adapter directory or its parent directory."
        )

    @staticmethod
    def _lora_load_success(result) -> bool:
        if result is False:
            return False
        if isinstance(result, dict) and result.get("success") is False:
            return False
        if hasattr(result, "success") and getattr(result, "success") is False:
            return False
        return True

    def load_lora_adapter(self, name: str, path: str, pinned: bool = True) -> str:
        if not hasattr(self.engine, "load_lora_adapter"):
            raise RuntimeError("This SGLang Engine does not expose load_lora_adapter")
        adapter_dir = self._resolve_lora_adapter_path(path, name)
        loop = getattr(self.engine, "loop", None)
        if loop is not None and loop.is_running() and hasattr(self.engine, "async_load_lora_adapter"):
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is loop:
                raise RuntimeError(
                    "SGLang synchronous LoRA loading was called from the same "
                    "running event loop as the Engine. Use async_load_lora_adapter "
                    "to avoid deadlock."
                )
            result = self._run_on_engine_loop(
                self.engine.async_load_lora_adapter(name, adapter_dir, pinned=pinned)
            )
        else:
            result = self.engine.load_lora_adapter(name, adapter_dir, pinned=pinned)
        if not self._lora_load_success(result):
            raise RuntimeError(f"SGLang failed to load LoRA adapter {name!r}: {result}")
        logger_msg = f"Loaded SGLang LoRA adapter name={name!r} path={adapter_dir!r}"
        print(f"  [LoRA] {logger_msg}", flush=True)
        return adapter_dir

    async def async_load_lora_adapter(self, name: str, path: str, pinned: bool = True) -> str:
        adapter_dir = self._resolve_lora_adapter_path(path, name)
        if hasattr(self.engine, "async_load_lora_adapter"):
            result = await self.engine.async_load_lora_adapter(name, adapter_dir, pinned=pinned)
        else:
            from sglang.srt.managers.io_struct import LoadLoRAAdapterReqInput

            result = await self.engine.tokenizer_manager.load_lora_adapter(
                LoadLoRAAdapterReqInput(
                    lora_name=name,
                    lora_path=adapter_dir,
                    pinned=pinned,
                ),
                None,
            )
        if not self._lora_load_success(result):
            raise RuntimeError(f"SGLang failed to load LoRA adapter {name!r}: {result}")
        print(f"  [LoRA] Loaded SGLang LoRA adapter name={name!r} path={adapter_dir!r}", flush=True)
        return adapter_dir

    def unload_lora_adapter(self, name: str) -> None:
        if not hasattr(self.engine, "unload_lora_adapter"):
            raise RuntimeError("This SGLang Engine does not expose unload_lora_adapter")
        loop = getattr(self.engine, "loop", None)
        if loop is not None and loop.is_running() and hasattr(self.engine, "async_unload_lora_adapter"):
            result = self._run_on_engine_loop(self.engine.async_unload_lora_adapter(name))
        else:
            result = self.engine.unload_lora_adapter(name)
        if not self._lora_load_success(result):
            raise RuntimeError(f"SGLang failed to unload LoRA adapter {name!r}: {result}")

    async def async_unload_lora_adapter(self, name: str) -> None:
        if hasattr(self.engine, "async_unload_lora_adapter"):
            result = await self.engine.async_unload_lora_adapter(name)
        else:
            from sglang.srt.managers.io_struct import UnloadLoRAAdapterReqInput

            result = await self.engine.tokenizer_manager.unload_lora_adapter(
                UnloadLoRAAdapterReqInput(lora_name=name),
                None,
            )
        if not self._lora_load_success(result):
            raise RuntimeError(f"SGLang failed to unload LoRA adapter {name!r}: {result}")


    def _sampling_params(self) -> dict:
        params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "stop": [self.tokenizer.eos_token],
        }
        if self.sampling_seed is not None:
            params["sampling_seed"] = int(self.sampling_seed)
        return params

    def _with_sampling_seed(self, sampling_params: Optional[dict]) -> dict:
        params = copy.deepcopy(sampling_params or {})
        if self.sampling_seed is not None and "sampling_seed" not in params:
            params["sampling_seed"] = int(self.sampling_seed)
        return params

    def _dummy_token_id(self) -> int:
        for attr in ("pad_token_id", "eos_token_id", "unk_token_id", "bos_token_id"):
            token_id = getattr(self.tokenizer, attr, None)
            if token_id is not None:
                return int(token_id)
        raise ValueError("Tokenizer has no pad/eos/unk/bos token id for latent prefix")

    @staticmethod
    def _normalize_prompt_ids(prompt_ids) -> list[int]:
        if isinstance(prompt_ids, dict):
            prompt_ids = prompt_ids["input_ids"]
        elif hasattr(prompt_ids, "input_ids"):
            prompt_ids = prompt_ids.input_ids
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids and isinstance(prompt_ids[0], list):
            if len(prompt_ids) != 1:
                raise ValueError("Expected a single prompt, got batched prompt ids")
            prompt_ids = prompt_ids[0]
        return [int(token_id) for token_id in prompt_ids]


    async def __call__(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        latent_embeds: Optional[torch.Tensor] = None,
        lora_path: Optional[str] = None,
        latent_framing_before: Optional[str] = None,
        latent_framing_after: Optional[str] = None,
    ) -> dict:
        if latent_embeds is None:
            chat_template_kwargs = {
                "tools": tools or None,
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if self.enable_thinking is not None:
                chat_template_kwargs["enable_thinking"] = self.enable_thinking
            prompt = await asyncio.to_thread(
                self.tokenizer.apply_chat_template,
                messages,
                **chat_template_kwargs,
            )
        else:
            inject_kwargs = dict(
                enable_thinking=self.enable_thinking,
                add_generation_prompt=True,
                placeholder_token_id=self._dummy_token_id(),
            )
            if latent_framing_before is not None:
                inject_kwargs["framing_before"] = latent_framing_before
            if latent_framing_after is not None:
                inject_kwargs["framing_after"] = latent_framing_after
            input_ids, positions, _, _ = await asyncio.to_thread(
                build_latent_injected_ids,
                messages,
                tools,
                self.tokenizer,
                latent_embeds.shape[0],
                **inject_kwargs,
            )

        from sglang.srt.managers.io_struct import GenerateReqInput

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                if latent_embeds is not None:
                    from sglang.srt.managers.embed_types import PositionalEmbeds
                    req = GenerateReqInput(
                        input_ids=input_ids,
                        positional_embed_overrides=PositionalEmbeds(
                            embeds=latent_embeds.to("cuda:0").float(),
                            positions=positions,
                        ),
                        sampling_params=self._sampling_params(),
                        lora_path=lora_path,
                    )
                else:
                    req = GenerateReqInput(
                        text=prompt,
                        sampling_params=self._sampling_params(),
                        lora_path=lora_path,
                    )
                agen = self.engine.tokenizer_manager.generate_request(req, None)
                result = await agen.__anext__()
                text = result.get("text", "")
                return _parse_response(text)

            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    print(f"\n  [LLM] {type(e).__name__}: {e}, retrying in {wait}s...")
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries + 1} attempts"
        ) from last_exc

    def _generate_request_sync(self, req, timeout_s: float | None = None):

        async def _request():
            agen = self.engine.tokenizer_manager.generate_request(req, None)
            return await agen.__anext__()

        return self._run_on_engine_loop(_request(), timeout_s=timeout_s)


    async def release_gpu(self) -> None:
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput
        n_gpus = torch.cuda.device_count()

        def _log_all_gpus(label: str):
            print(f"  [release_gpu] {label}:", flush=True)
            import subprocess
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
                     "--format=csv,noheader,nounits"],
                    text=True,
                )
                for line in out.strip().split("\n"):
                    print(f"    {line.strip()}", flush=True)
            except Exception:
                for i in range(n_gpus):
                    free, total = torch.cuda.mem_get_info(i)
                    print(f"    GPU{i}: used={(total-free)/1024**3:.1f}GB", flush=True)

        _log_all_gpus("before")
        await asyncio.sleep(5.0)
        await self.engine.tokenizer_manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=["kv_cache"]), None
        )
        await self.engine.tokenizer_manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=["weights"]), None
        )
        torch.cuda.empty_cache()
        _log_all_gpus("after")

    async def resume_gpu(self, weights_path: str | None = None) -> None:
        from sglang.srt.managers.io_struct import (
            ResumeMemoryOccupationReqInput, UpdateWeightFromDiskReqInput,
        )
        await self.engine.tokenizer_manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=["weights"]), None
        )
        await self.engine.tokenizer_manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=["kv_cache"]), None
        )
        path_to_load = weights_path if weights_path is not None else self.model_path
        await self.engine.tokenizer_manager.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path=path_to_load, load_format=None), None
        )

    async def update_weights_from_path(self, weights_path: str) -> None:
        from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput
        await self.engine.tokenizer_manager.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path=weights_path, load_format=None), None
        )


    def _run_on_engine_loop(self, coro, timeout_s: float | None = None):
        loop = getattr(self.engine, "loop", None)
        if timeout_s is not None:
            coro = asyncio.wait_for(coro, timeout=timeout_s)

        if loop is None:
            return asyncio.run(coro)

        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            try:
                return future.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError as exc:
                future.cancel()
                raise TimeoutError(f"SGLang engine request timed out after {timeout_s}s") from exc

        with self._engine_loop_lock:
            return loop.run_until_complete(coro)

    @staticmethod
    def _normalize_hidden_states(result) -> torch.Tensor:
        hidden = result["meta_info"]["hidden_states"]

        if isinstance(hidden, list):
            hidden = torch.tensor(hidden)
        if not isinstance(hidden, torch.Tensor):
            raise ValueError(f"Unexpected hidden_states type: {type(hidden)}")

        hidden = hidden.cpu()

        if hidden.dim() == 1:
            hidden = hidden.unsqueeze(0)
        elif hidden.dim() == 3:
            hidden = hidden.squeeze(0)
        elif hidden.dim() != 2:
            raise ValueError(f"Unexpected hidden_states shape: {hidden.shape}")

        return hidden

    def get_hidden_states_sync(
        self,
        text: str,
        lora_path: str | None = None,
        timeout_s: float | None = None,
    ) -> torch.Tensor:
        from sglang.srt.managers.io_struct import GenerateReqInput

        req = GenerateReqInput(
            text=text,
            sampling_params=self._with_sampling_seed({"max_new_tokens": 1, "temperature": 0.0}),
            return_hidden_states=True,
            lora_path=lora_path,
        )

        result = self._generate_request_sync(req, timeout_s=timeout_s)
        return self._normalize_hidden_states(result)

    async def get_hidden_states(
        self,
        text: str,
        lora_path: str | None = None,
        timeout_s: float | None = None,
    ) -> torch.Tensor:
        from sglang.srt.managers.io_struct import GenerateReqInput

        req = GenerateReqInput(
            text=text,
            sampling_params=self._with_sampling_seed({"max_new_tokens": 1, "temperature": 0.0}),
            return_hidden_states=True,
            lora_path=lora_path,
        )
        agen = self.engine.tokenizer_manager.generate_request(req, None)
        result = await asyncio.wait_for(agen.__anext__(), timeout=timeout_s) if timeout_s else await agen.__anext__()
        return self._normalize_hidden_states(result)

    def shutdown(self):
        if hasattr(self, "engine") and self.engine is not None:
            loop = getattr(self.engine, "loop", None)
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(loop.stop)
                if self._engine_loop_thread is not None:
                    self._engine_loop_thread.join(timeout=5.0)
            self.engine.shutdown()
            self.engine = None



def _parse_response(text: str) -> dict:
    tool_calls = _extract_tool_calls(text)
    return {"role": "assistant", "content": text, "tool_calls": tool_calls}


def _extract_tool_calls(text: str) -> list[dict] | None:
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if not blocks:
        return None
    for raw in reversed(blocks):
        try:
            obj = json.loads(raw.strip())
        except Exception:
            continue
        name = obj.get("name") or obj.get("function_name")
        if not name:
            continue
        args = obj.get("arguments") or obj.get("parameters") or {}
        return [
            {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": (
                        json.dumps(args, ensure_ascii=False)
                        if isinstance(args, dict)
                        else str(args)
                    ),
                },
            }
        ]
    return None
