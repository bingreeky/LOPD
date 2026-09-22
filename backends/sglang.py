import logging
import asyncio
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from memory.prompt_protocol import resolve_compressor_prompt_protocol
from memory.serialization import build_latent_injected_ids, chat_template_kwargs

from .base import DEFAULT_MAX_RETRIES, BaseLLM

logger = logging.getLogger(__name__)


def _load_embed_tokens_cpu(model_path: str) -> torch.nn.Embedding:
    import json as _json
    from safetensors.torch import load_file

    model_dir = Path(model_path)
    weight_key = "model.embed_tokens.weight"

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            shard_file = _json.load(f)["weight_map"][weight_key]
        weight = load_file(str(model_dir / shard_file), device="cpu")[weight_key]
    else:
        single = model_dir / "model.safetensors"
        if not single.exists():
            raise FileNotFoundError(f"No safetensors checkpoint found in {model_path}")
        weight = load_file(str(single), device="cpu")[weight_key]

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
        max_retries: int = DEFAULT_MAX_RETRIES,
        enable_thinking: bool | None = None,
        latent_prompt_protocol: str | None = None,
        top_p: float = 0.95,
        top_k: int = 20,
        tp_size: int = 1,
        dtype: str = "bfloat16",
        mem_fraction_static: float = 0.8,
        disable_radix_cache: bool = True,
        enable_memory_saver: bool = False,
        enable_return_hidden_states: bool = False,
        enable_lora: bool = False,
        max_lora_rank: int | None = None,
        lora_target_modules: list[str] | None = None,
        lora_paths: list[dict] | dict[str, str] | list[str] | None = None,
        enable_deterministic_inference: bool = False,
        sampling_seed: int | None = None,
    ):
        self.model_path = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.enable_thinking = enable_thinking
        self.latent_protocol = None if latent_prompt_protocol is None else resolve_compressor_prompt_protocol(latent_prompt_protocol)
        self.top_p = top_p
        self.top_k = top_k
        self.sampling_seed = sampling_seed

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )

        import sglang as sgl
        engine_kwargs = dict(
            model_path=model_path,
            tp_size=tp_size,
            dtype=dtype,
            mem_fraction_static=mem_fraction_static,
            trust_remote_code=True,
            disable_radix_cache=disable_radix_cache,
            enable_memory_saver=enable_memory_saver,
            enable_return_hidden_states=enable_return_hidden_states,
        )
        if enable_deterministic_inference:
            engine_kwargs["enable_deterministic_inference"] = True
        if enable_lora:
            if max_lora_rank is None:
                raise ValueError("max_lora_rank is required when LoRA is enabled")
            if not lora_target_modules:
                raise ValueError("lora_target_modules is required when LoRA is enabled")
            engine_kwargs.update(
                enable_lora=True,
                max_lora_rank=max_lora_rank,
                lora_target_modules=lora_target_modules,
            )
            if lora_paths:
                engine_kwargs["lora_paths"] = lora_paths
        self.engine = sgl.Engine(**engine_kwargs)

        self.embed_layer = _load_embed_tokens_cpu(model_path)

    def _sampling_params(self, **overrides) -> dict:
        params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stop": [self.tokenizer.eos_token],
            **overrides,
        }
        if self.sampling_seed is not None:
            params["sampling_seed"] = int(self.sampling_seed)
        return params

    async def __call__(
        self,
        messages: list[dict],
        *,
        latent_embeds: Optional[torch.Tensor] = None,
        lora_path: Optional[str] = None,
    ) -> dict:
        if latent_embeds is None:
            prompt = await asyncio.to_thread(
                self.tokenizer.apply_chat_template, messages,
                **chat_template_kwargs(self.enable_thinking, add_generation_prompt=True),
            )
        else:
            if self.latent_protocol is None:
                raise ValueError("latent injection requires latent_prompt_protocol")
            input_ids, positions = await asyncio.to_thread(
                build_latent_injected_ids, messages, self.tokenizer, latent_embeds.shape[0], self.latent_protocol,
                enable_thinking=self.enable_thinking, add_generation_prompt=True,
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
                return {"role": "assistant", "content": result.get("text", "")}

            except Exception as e:
                last_exc = e
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning("SGLang call failed (%s: %s), retrying in %ds", type(e).__name__, e, wait)
                    await asyncio.sleep(wait)

        raise RuntimeError(
            f"LLM call failed after {self.max_retries + 1} attempts"
        ) from last_exc

    async def release_gpu(self) -> None:
        from sglang.srt.managers.io_struct import ReleaseMemoryOccupationReqInput

        await asyncio.sleep(5.0)
        await self.engine.tokenizer_manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=["kv_cache"]), None
        )
        await self.engine.tokenizer_manager.release_memory_occupation(
            ReleaseMemoryOccupationReqInput(tags=["weights"]), None
        )
        torch.cuda.empty_cache()

    async def resume_gpu(self, weights_path: str) -> None:
        from sglang.srt.managers.io_struct import ResumeMemoryOccupationReqInput
        await self.engine.tokenizer_manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=["weights"]), None
        )
        await self.engine.tokenizer_manager.resume_memory_occupation(
            ResumeMemoryOccupationReqInput(tags=["kv_cache"]), None
        )
        await self.update_weights_from_path(weights_path)

    async def update_weights_from_path(self, weights_path: str) -> None:
        from sglang.srt.managers.io_struct import UpdateWeightFromDiskReqInput
        await self.engine.tokenizer_manager.update_weights_from_disk(
            UpdateWeightFromDiskReqInput(model_path=weights_path, load_format=None), None
        )

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

    async def get_hidden_states(
        self,
        text: str,
        lora_path: str | None = None,
        timeout_s: float | None = None,
    ) -> torch.Tensor:
        from sglang.srt.managers.io_struct import GenerateReqInput

        req = GenerateReqInput(
            text=text,
            sampling_params=self._sampling_params(max_new_tokens=1, temperature=0.0),
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
            self.engine.shutdown()
            self.engine = None


