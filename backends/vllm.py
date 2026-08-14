
from __future__ import annotations

import ast
import asyncio
import gc
import hashlib
import inspect
import json
import math
import os
import re
import struct
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from memory.serialization import build_latent_injected_ids
except ImportError:
    build_latent_injected_ids = None

from .base import BaseLLM

_DEFAULT_RETRIES = 2
_UPDATE_MANIFEST_SCHEMA_VERSION = 2
_UPDATE_MANIFEST_KEYS = frozenset({
    "schema_version",
    "session_id",
    "backend",
    "full_model",
    "weights_path",
    "tensor_count",
    "tensor_names_sha256",
    "tensor_specs_sha256",
    "weight_map_sha256",
    "chunk_count",
})
_IPC_UPDATE_INFO_KEYS = frozenset({
    "names",
    "dtype_names",
    "shapes",
    "ipc_handles",
    "ipc_handles_pickled",
    "tensor_sizes",
    "packed",
})
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

_SAFETENSORS_DTYPES: dict[str, tuple[str, int]] = {
    "BOOL": ("bool", 1),
    "U8": ("uint8", 1),
    "I8": ("int8", 1),
    "I16": ("int16", 2),
    "U16": ("uint16", 2),
    "I32": ("int32", 4),
    "U32": ("uint32", 4),
    "I64": ("int64", 8),
    "U64": ("uint64", 8),
    "BF16": ("bfloat16", 2),
    "F16": ("float16", 2),
    "F32": ("float32", 4),
    "F64": ("float64", 8),
    "F8_E4M3": ("float8_e4m3fn", 1),
    "F8_E5M2": ("float8_e5m2", 1),
    "C64": ("complex64", 8),
    "C128": ("complex128", 16),
}


class VLLMLifecycleState(str, Enum):

    RUNNING = "RUNNING"
    DRAINING = "DRAINING"
    SLEEPING = "SLEEPING"
    UPDATING = "UPDATING"
    ENCODING = "ENCODING"
    FAILED = "FAILED"


@dataclass
class _WeightUpdateSession:
    session_id: str
    weights_path: str
    expected_names: frozenset[str]
    expected_specs: dict[str, dict[str, Any]]
    expected_chunk_count: int
    prepared_embed_layer: Any
    seen_names: set[str]
    next_chunk_index: int = 0


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_safetensors_specs(path: Path) -> dict[str, dict[str, Any]]:
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        raw_header_size = handle.read(8)
        if len(raw_header_size) != 8:
            raise ValueError(f"Invalid safetensors header prefix: {path}")
        (header_size,) = struct.unpack("<Q", raw_header_size)
        if header_size <= 0 or header_size > file_size - 8:
            raise ValueError(
                f"Invalid safetensors header size {header_size} for {path}"
            )
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"Truncated safetensors header: {path}")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid safetensors metadata JSON: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Safetensors metadata must be an object: {path}")

    data_size = file_size - 8 - header_size
    result: dict[str, dict[str, Any]] = {}
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not name or not isinstance(metadata, dict):
            raise ValueError(f"Invalid tensor metadata entry in {path}: {name!r}")
        dtype_code = metadata.get("dtype")
        dtype_spec = _SAFETENSORS_DTYPES.get(dtype_code)
        if dtype_spec is None:
            raise ValueError(
                f"Unsupported safetensors dtype {dtype_code!r} for {name!r}"
            )
        canonical_dtype, itemsize = dtype_spec
        shape = metadata.get("shape")
        if not isinstance(shape, list) or not all(
            isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0
            for dim in shape
        ):
            raise ValueError(f"Invalid shape metadata for {name!r} in {path}")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(
                isinstance(offset, int)
                and not isinstance(offset, bool)
                and offset >= 0
                for offset in offsets
            )
        ):
            raise ValueError(f"Invalid data_offsets for {name!r} in {path}")
        start, end = offsets
        if end < start or end > data_size:
            raise ValueError(f"Out-of-range data_offsets for {name!r} in {path}")
        numel = 1
        for dim in shape:
            numel *= dim
        expected_nbytes = numel * itemsize
        actual_nbytes = end - start
        if actual_nbytes != expected_nbytes:
            raise ValueError(
                f"Safetensors byte-size mismatch for {name!r}: "
                f"metadata={actual_nbytes}, dtype/shape={expected_nbytes}"
            )
        result[name] = {
            "dtype": canonical_dtype,
            "shape": list(shape),
            "nbytes": actual_nbytes,
        }
    if not result:
        raise ValueError(f"Empty safetensors checkpoint shard: {path}")
    return result


def packed_weight_update_preflight(
    tensor_names: list[str] | tuple[str, ...],
    tensor_specs: dict[str, dict[str, Any]],
    packed_buffer_size_bytes: int,
) -> dict[str, Any]:
    if (
        not isinstance(packed_buffer_size_bytes, int)
        or isinstance(packed_buffer_size_bytes, bool)
        or packed_buffer_size_bytes <= 0
    ):
        raise ValueError("packed_buffer_size_bytes must be a positive integer")
    if not isinstance(tensor_names, (list, tuple)) or not tensor_names:
        raise ValueError("tensor_names must be a non-empty ordered list/tuple")
    names = list(tensor_names)
    if not all(isinstance(name, str) and name for name in names):
        raise ValueError("tensor_names contains an invalid name")
    if len(set(names)) != len(names):
        raise ValueError("tensor_names contains duplicates")
    if not isinstance(tensor_specs, dict):
        raise TypeError("tensor_specs must be a dict")
    missing = [name for name in names if name not in tensor_specs]
    extra = sorted(set(tensor_specs) - set(names))
    if missing or extra:
        raise ValueError(
            f"tensor_names/spec coverage mismatch: missing={missing}, extra={extra}"
        )

    sizes: list[int] = []
    for name in names:
        spec = tensor_specs[name]
        nbytes = spec.get("nbytes") if isinstance(spec, dict) else None
        if (
            not isinstance(nbytes, int)
            or isinstance(nbytes, bool)
            or nbytes <= 0
        ):
            raise ValueError(f"Invalid nbytes spec for tensor {name!r}: {nbytes!r}")
        sizes.append(nbytes)
    max_index = max(range(len(names)), key=sizes.__getitem__)
    max_name = names[max_index]
    max_nbytes = sizes[max_index]
    if packed_buffer_size_bytes < max_nbytes:
        raise ValueError(
            "packed buffer cannot fit the largest single tensor: "
            f"buffer={packed_buffer_size_bytes}, tensor={max_name!r}, "
            f"nbytes={max_nbytes}"
        )

    chunks: list[list[str]] = []
    chunk_sizes: list[int] = []
    current_names: list[str] = []
    current_nbytes = 0
    for name, nbytes in zip(names, sizes):
        if current_names and current_nbytes + nbytes > packed_buffer_size_bytes:
            chunks.append(current_names)
            chunk_sizes.append(current_nbytes)
            current_names = []
            current_nbytes = 0
        current_names.append(name)
        current_nbytes += nbytes
    if current_names:
        chunks.append(current_names)
        chunk_sizes.append(current_nbytes)
    if [name for chunk in chunks for name in chunk] != names:
        raise AssertionError("packed preflight changed tensor order")
    return {
        "packed_buffer_size_bytes": packed_buffer_size_bytes,
        "tensor_count": len(names),
        "total_nbytes": sum(sizes),
        "max_tensor_name": max_name,
        "max_tensor_nbytes": max_nbytes,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "chunk_nbytes": chunk_sizes,
    }


def _checkpoint_inventory(weights_path: str) -> dict[str, Any]:
    root = Path(weights_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"weights_path must be a checkpoint directory: {root}")

    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid or empty weight_map in {index_path}")
        if not all(
            isinstance(name, str) and name
            and isinstance(shard, str) and shard
            for name, shard in weight_map.items()
        ):
            raise ValueError(f"Invalid weight_map entries in {index_path}")
        missing = sorted({
            shard for shard in weight_map.values()
            if not (root / shard).is_file()
        })
        if missing:
            raise FileNotFoundError(
                f"Checkpoint references missing safetensors shards: {missing}"
            )
    else:
        single_path = root / "model.safetensors"
        if not single_path.is_file():
            raise FileNotFoundError(
                f"No model.safetensors.index.json or model.safetensors in {root}"
            )
        shard_specs = _read_safetensors_specs(single_path)
        weight_map = {name: single_path.name for name in shard_specs}

    names = tuple(sorted(weight_map))
    tensor_specs: dict[str, dict[str, Any]] = {}
    for shard in sorted(set(weight_map.values())):
        shard_path = root / shard
        specs = _read_safetensors_specs(shard_path)
        expected_in_shard = {
            name for name, mapped_shard in weight_map.items()
            if mapped_shard == shard
        }
        actual_in_shard = set(specs)
        if actual_in_shard != expected_in_shard:
            raise ValueError(
                f"Safetensors index/header mismatch for {shard}: "
                f"missing={sorted(expected_in_shard - actual_in_shard)}, "
                f"extra={sorted(actual_in_shard - expected_in_shard)}"
            )
        tensor_specs.update(specs)
    ordered_specs = {
        name: tensor_specs[name]
        for name in names
    }
    return {
        "weights_path": str(root),
        "names": names,
        "tensor_count": len(names),
        "tensor_names_sha256": hashlib.sha256(
            "\n".join(names).encode("utf-8")
        ).hexdigest(),
        "tensor_specs": ordered_specs,
        "tensor_specs_sha256": _sha256_json(ordered_specs),
        "weight_map_sha256": _sha256_json(weight_map),
    }


def checkpoint_packed_weight_update_preflight(
    weights_path: str,
    *,
    packed_buffer_size_bytes: int,
    tensor_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    inventory = _checkpoint_inventory(weights_path)
    names = inventory["names"] if tensor_names is None else tensor_names
    plan = packed_weight_update_preflight(
        names,
        inventory["tensor_specs"],
        packed_buffer_size_bytes,
    )
    return {
        **plan,
        "weights_path": inventory["weights_path"],
        "tensor_specs": inventory["tensor_specs"],
        "tensor_specs_sha256": inventory["tensor_specs_sha256"],
    }


def _load_embed_tokens_cpu(model_path: str) -> torch.nn.Embedding:
    from safetensors import safe_open

    model_dir = Path(model_path)
    candidate_keys = (
        "model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
    )
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as handle:
            weight_map = json.load(handle)["weight_map"]
        key = next((name for name in candidate_keys if name in weight_map), None)
        if key is None:
            raise KeyError(f"No input embedding key found; tried {candidate_keys}")
        shard_path = model_dir / weight_map[key]
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(key)
    else:
        weights_path = model_dir / "model.safetensors"
        if not weights_path.exists():
            raise FileNotFoundError(f"No safetensors checkpoint found in {model_path}")
        with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            key = next((name for name in candidate_keys if name in available), None)
            if key is None:
                raise KeyError(f"No input embedding key found; tried {candidate_keys}")
            weight = handle.get_tensor(key)

    embedding = torch.nn.Embedding(*weight.shape, _weight=weight.to(torch.bfloat16))
    embedding.requires_grad_(False)
    return embedding


class VLLMLlm(BaseLLM):

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
        sampling_seed: int | None = None,
        stop: list[str] | str | None = None,
        tp_size: int = 1,
        dp_size: int = 1,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.85,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        enforce_eager: bool = False,
        enable_prefix_caching: bool = False,
        enable_chunked_prefill: bool | None = None,
        disable_custom_all_reduce: bool = False,
        distributed_executor_backend: str | None = None,
        compilation_config: dict | None = None,
        enable_sleep_mode: bool = False,
        weight_transfer_backend: str | None = None,
        lifecycle_drain_timeout_s: int = 300,
        enable_lora: bool = False,
        max_lora_rank: int | None = None,
        lora_paths: list[dict] | dict[str, str] | list[str] | None = None,
        max_loaded_loras: int = 4,
        encoder_device: str = "cuda:0",
        encoder_dtype: str | None = None,
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
        self.stop = [stop] if isinstance(stop, str) else stop
        self.encoder_device = encoder_device
        self.encoder_dtype = encoder_dtype or dtype
        self._sleep_mode_enabled = bool(enable_sleep_mode)
        self._weight_transfer_backend = (
            str(weight_transfer_backend).lower()
            if weight_transfer_backend is not None else None
        )
        if self._weight_transfer_backend not in (None, "ipc"):
            raise ValueError(
                "VLLMLlm only permits the audited same-node IPC weight-transfer "
                f"backend, got {weight_transfer_backend!r}"
            )
        if self._weight_transfer_backend and not self._sleep_mode_enabled:
            raise ValueError(
                "weight_transfer_backend='ipc' requires enable_sleep_mode=true"
            )
        if (
            self._weight_transfer_backend == "ipc"
            and os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1"
        ):
            raise RuntimeError(
                "vLLM 0.25 IPC handle transport requires the process-local "
                "VLLM_ALLOW_INSECURE_SERIALIZATION=1 opt-in. This wrapper never "
                "sets it globally. Only use an authenticated loopback controller; "
                "never expose the IPC update endpoint over HTTP."
            )
        if (
            not isinstance(lifecycle_drain_timeout_s, int)
            or isinstance(lifecycle_drain_timeout_s, bool)
            or lifecycle_drain_timeout_s <= 0
        ):
            raise ValueError("lifecycle_drain_timeout_s must be a positive integer")
        self._lifecycle_drain_timeout_s = lifecycle_drain_timeout_s

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        self.embed_layer = _load_embed_tokens_cpu(model_path)

        self._engine_kwargs = {
            "model": model_path,
            "tensor_parallel_size": int(tp_size),
            "data_parallel_size": int(dp_size),
            "dtype": dtype,
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "trust_remote_code": True,
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "disable_log_stats": True,
            "enable_prompt_embeds": True,
            "enforce_eager": bool(enforce_eager),
            "enable_prefix_caching": bool(enable_prefix_caching),
            "enable_chunked_prefill": enable_chunked_prefill,
            "disable_custom_all_reduce": bool(disable_custom_all_reduce),
            "distributed_executor_backend": distributed_executor_backend,
            "compilation_config": compilation_config,
            "enable_sleep_mode": self._sleep_mode_enabled,
            "seed": sampling_seed,
            "enable_lora": bool(enable_lora),
            "max_lora_rank": max_lora_rank,
            "max_loras": int(max_loaded_loras),
        }
        self.engine = None
        self._engine_lock = asyncio.Lock()
        self._encoder_lock = asyncio.Lock()
        self._encoder_activity_condition = asyncio.Condition()
        self._active_encoder_calls = 0
        self._encoder_activity_blocked = False
        self._encoder_model = None
        self._encoder_adapter_name: str | None = None
        self._adapter_paths: dict[str, str] = {}
        self._lora_requests: dict[str, Any] = {}
        self._next_lora_id = 1
        self._parse_preloaded_loras(lora_paths)

        self._lifecycle_state = VLLMLifecycleState.RUNNING
        self._lifecycle_failure: BaseException | None = None
        self._lifecycle_condition = asyncio.Condition()
        self._lifecycle_operation_lock = asyncio.Lock()
        self._active_generations = 0
        self._weight_update_session: _WeightUpdateSession | None = None
        self._weight_transfer_initialized = False


    @property
    def lifecycle_state(self) -> str:
        return self._lifecycle_state.value

    def lifecycle_status(self) -> dict[str, Any]:
        session = self._weight_update_session
        return {
            "state": self.lifecycle_state,
            "active_generations": self._active_generations,
            "active_encoder_calls": self._active_encoder_calls,
            "encoder_activity_blocked": self._encoder_activity_blocked,
            "failure": (
                None if self._lifecycle_failure is None
                else f"{type(self._lifecycle_failure).__name__}: "
                     f"{self._lifecycle_failure}"
            ),
            "session_id": None if session is None else session.session_id,
            "next_chunk_index": (
                None if session is None else session.next_chunk_index
            ),
            "expected_chunk_count": (
                None if session is None else session.expected_chunk_count
            ),
            "seen_tensor_count": (
                None if session is None else len(session.seen_names)
            ),
        }

    def _require_sleep_mode(self) -> None:
        if not self._sleep_mode_enabled:
            raise NotImplementedError(
                "vLLM hot lifecycle is disabled; set enable_sleep_mode=true "
                "explicitly to use level-1 sleep/wake"
            )

    def _require_ipc_weight_transfer(self) -> None:
        self._require_sleep_mode()
        if self._weight_transfer_backend != "ipc":
            raise NotImplementedError(
                "Full-model hot updates require weight_transfer_backend='ipc'"
            )

    def _raise_if_failed(self) -> None:
        if self._lifecycle_state is VLLMLifecycleState.FAILED:
            raise RuntimeError(
                "vLLM lifecycle is fail-closed after a previous error; create a "
                "new VLLMLlm instance after inspecting the original exception"
            ) from self._lifecycle_failure

    async def _set_lifecycle_state(self, state: VLLMLifecycleState) -> None:
        async with self._lifecycle_condition:
            self._lifecycle_state = state
            self._lifecycle_condition.notify_all()

    async def _fail_lifecycle(self, exc: BaseException) -> None:
        async with self._lifecycle_condition:
            if self._lifecycle_failure is None:
                self._lifecycle_failure = exc
            self._lifecycle_state = VLLMLifecycleState.FAILED
            self._lifecycle_condition.notify_all()

    @staticmethod
    async def _await_cleanup_completion(awaitable):
        task = asyncio.create_task(awaitable)
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError as exc:
                if task.cancelled():
                    if cancellation is not None:
                        raise exc from cancellation
                    raise
                if cancellation is None:
                    cancellation = exc
            except BaseException as exc:
                if cancellation is not None:
                    raise exc from cancellation
                raise
        if cancellation is not None:
            raise cancellation
        return result

    @asynccontextmanager
    async def _generation_lease(self):
        async with self._lifecycle_condition:
            while self._lifecycle_state is not VLLMLifecycleState.RUNNING:
                self._raise_if_failed()
                await self._lifecycle_condition.wait()
            self._active_generations += 1
        try:
            yield
        finally:
            async def _finish_generation_lease() -> None:
                async with self._lifecycle_condition:
                    self._active_generations -= 1
                    if self._active_generations < 0:
                        invariant = RuntimeError(
                            "negative vLLM generation lease count"
                        )
                        self._active_generations = 0
                        if self._lifecycle_failure is None:
                            self._lifecycle_failure = invariant
                        self._lifecycle_state = VLLMLifecycleState.FAILED
                    self._lifecycle_condition.notify_all()

            await self._await_cleanup_completion(_finish_generation_lease())

    @asynccontextmanager
    async def _encoder_activity_lease(self):
        async with self._encoder_activity_condition:
            while self._encoder_activity_blocked:
                self._raise_if_failed()
                await self._encoder_activity_condition.wait()
            self._raise_if_failed()
            if (
                self.engine is not None
                and self._lifecycle_state is not VLLMLifecycleState.ENCODING
            ):
                raise RuntimeError(
                    "with an initialized vLLM engine, encoder activity is "
                    "available only inside encoder_phase()"
                )
            if self._lifecycle_state not in (
                VLLMLifecycleState.RUNNING,
                VLLMLifecycleState.ENCODING,
            ):
                raise RuntimeError(
                    "encoder activity is unavailable during lifecycle state "
                    f"{self.lifecycle_state}"
                )
            self._active_encoder_calls += 1
        try:
            yield
        finally:
            async def _finish_encoder_activity() -> None:
                async with self._encoder_activity_condition:
                    self._active_encoder_calls -= 1
                    if self._active_encoder_calls < 0:
                        invariant = RuntimeError(
                            "negative vLLM encoder activity count"
                        )
                        self._active_encoder_calls = 0
                        if self._lifecycle_failure is None:
                            self._lifecycle_failure = invariant
                        self._lifecycle_state = VLLMLifecycleState.FAILED
                    self._encoder_activity_condition.notify_all()

            await self._await_cleanup_completion(_finish_encoder_activity())

    @asynccontextmanager
    async def _encoder_quiescence(
        self,
        *,
        fail_closed_on_timeout: bool = False,
    ):
        acquired = False

        async def _block_and_wait_for_activity() -> None:
            nonlocal acquired
            async with self._encoder_activity_condition:
                while self._encoder_activity_blocked:
                    self._raise_if_failed()
                    await self._encoder_activity_condition.wait()
                self._raise_if_failed()
                self._encoder_activity_blocked = True
                acquired = True
                while self._active_encoder_calls:
                    await self._encoder_activity_condition.wait()

        try:
            try:
                await asyncio.wait_for(
                    _block_and_wait_for_activity(),
                    timeout=self._lifecycle_drain_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                if fail_closed_on_timeout:
                    await self._await_cleanup_completion(
                        self._fail_lifecycle(exc),
                    )
                raise
            yield
        finally:
            if acquired:
                async def _release_encoder_barrier() -> None:
                    async with self._encoder_activity_condition:
                        self._encoder_activity_blocked = False
                        self._encoder_activity_condition.notify_all()

                await self._await_cleanup_completion(
                    _release_encoder_barrier(),
                )

    async def set_initial_weights_path(self, weights_path: str) -> str:
        self._raise_if_failed()
        async with self._lifecycle_condition:
            if self.engine is not None:
                raise RuntimeError(
                    "set_initial_weights_path must run before vLLM engine creation"
                )
            if self._active_generations:
                raise RuntimeError(
                    "set_initial_weights_path requires no active generation"
                )
            if self._lifecycle_state is not VLLMLifecycleState.RUNNING:
                raise RuntimeError(
                    "set_initial_weights_path requires RUNNING, got "
                    f"{self.lifecycle_state}"
                )

        inventory = await asyncio.to_thread(_checkpoint_inventory, weights_path)
        prepared_embed = await asyncio.to_thread(
            _load_embed_tokens_cpu, inventory["weights_path"],
        )

        async with self._lifecycle_operation_lock:
            async with self._engine_lock:
                async with self._encoder_quiescence():
                    transitioned = False
                    try:
                        async with self._lifecycle_condition:
                            self._raise_if_failed()
                            if self.engine is not None:
                                raise RuntimeError(
                                    "set_initial_weights_path must run before "
                                    "vLLM engine creation"
                                )
                            if self._active_generations:
                                raise RuntimeError(
                                    "set_initial_weights_path requires no active "
                                    "generation"
                                )
                            if (
                                self._lifecycle_state
                                is not VLLMLifecycleState.RUNNING
                            ):
                                raise RuntimeError(
                                    "set_initial_weights_path requires RUNNING, "
                                    f"got {self.lifecycle_state}"
                                )
                            current_shape = self._embedding_shape(self.embed_layer)
                            replacement_shape = self._embedding_shape(
                                prepared_embed,
                            )
                            if replacement_shape != current_shape:
                                raise ValueError(
                                    "Initial checkpoint cannot change the input "
                                    "embedding shape: "
                                    f"current={current_shape}, "
                                    f"new={replacement_shape}"
                                )
                            self._lifecycle_state = VLLMLifecycleState.DRAINING
                            transitioned = True
                            self._lifecycle_condition.notify_all()

                        await self._drop_encoder_locked()
                        self.model_path = inventory["weights_path"]
                        self._engine_kwargs["model"] = inventory["weights_path"]
                        self.embed_layer = prepared_embed
                        await self._set_lifecycle_state(
                            VLLMLifecycleState.RUNNING,
                        )
                    except BaseException as exc:
                        if transitioned:
                            await self._await_cleanup_completion(
                                self._fail_lifecycle(exc),
                            )
                        raise
        return inventory["weights_path"]

    @staticmethod
    def _supported_engine_kwargs(engine_args_cls, kwargs: dict) -> dict:
        accepted = set(inspect.signature(engine_args_cls).parameters)
        essential = {"model", "tensor_parallel_size", "dtype"}
        missing = essential - accepted
        if missing:
            raise RuntimeError(
                f"Installed vLLM AsyncEngineArgs is missing required fields: {sorted(missing)}"
            )
        return {
            key: value
            for key, value in kwargs.items()
            if value is not None and key in accepted
        }

    async def _ensure_engine(self):
        self._raise_if_failed()
        if self.engine is not None:
            return self.engine
        async with self._engine_lock:
            self._raise_if_failed()
            if self.engine is not None:
                return self.engine
            async with self._encoder_quiescence(
                fail_closed_on_timeout=True,
            ):
                self._raise_if_failed()
                if self.engine is not None:
                    return self.engine
                await self._drop_encoder_locked()
                from vllm import AsyncEngineArgs, AsyncLLMEngine

                engine_kwargs = dict(self._engine_kwargs)
                accepted = set(inspect.signature(AsyncEngineArgs).parameters)
                if (
                    self._sleep_mode_enabled
                    and "enable_sleep_mode" not in accepted
                ):
                    raise RuntimeError(
                        "Installed vLLM does not expose "
                        "AsyncEngineArgs.enable_sleep_mode"
                    )
                if self._weight_transfer_backend == "ipc":
                    if "weight_transfer_config" not in accepted:
                        raise RuntimeError(
                            "Installed vLLM does not expose weight_transfer_config"
                        )
                    from vllm.config import WeightTransferConfig

                    engine_kwargs["weight_transfer_config"] = WeightTransferConfig(
                        backend="ipc"
                    )
                kwargs = self._supported_engine_kwargs(
                    AsyncEngineArgs, engine_kwargs,
                )
                try:
                    self.engine = AsyncLLMEngine.from_engine_args(
                        AsyncEngineArgs(**kwargs)
                    )
                    if self._weight_transfer_backend == "ipc":
                        from vllm.distributed.weight_transfer.base import (
                            WeightTransferInitRequest,
                        )

                        await self.engine.init_weight_transfer_engine(
                            WeightTransferInitRequest(init_info={})
                        )
                        self._weight_transfer_initialized = True
                    for name, path in self._adapter_paths.items():
                        await self._load_lora_into_engine(name, path)
                except BaseException as exc:
                    if self._sleep_mode_enabled:
                        await self._await_cleanup_completion(
                            self._fail_lifecycle(exc),
                        )
                    raise
        return self.engine

    def shutdown(self):
        self._drop_encoder_sync()
        engine, self.engine = self.engine, None
        if engine is not None:
            result = engine.shutdown()
            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(result)
                else:
                    loop.create_task(result)


    def _sampling_params(self, overrides: Optional[dict] = None):
        from vllm import SamplingParams

        params = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "min_p": self.min_p,
            "stop": self.stop,
            "seed": self.sampling_seed,
        }
        for key, value in (overrides or {}).items():
            mapped = {"max_new_tokens": "max_tokens", "sampling_seed": "seed"}.get(key, key)
            params[mapped] = value
        return SamplingParams(**{key: value for key, value in params.items() if value is not None})

    async def _render_text_prompt(self, messages: list[dict], tools: Optional[list[dict]]):
        kwargs = {
            "tools": tools or None,
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking is not None:
            kwargs["enable_thinking"] = self.enable_thinking
        try:
            return await asyncio.to_thread(
                self.tokenizer.apply_chat_template, messages, **kwargs,
            )
        except TypeError as exc:
            if "enable_thinking" not in kwargs:
                raise
            kwargs.pop("enable_thinking")
            return await asyncio.to_thread(
                self.tokenizer.apply_chat_template, messages, **kwargs,
            )

    def _dummy_token_id(self) -> int:
        for attr in ("pad_token_id", "eos_token_id", "unk_token_id", "bos_token_id"):
            token_id = getattr(self.tokenizer, attr, None)
            if isinstance(token_id, (list, tuple)):
                token_id = token_id[0] if token_id else None
            if token_id is not None:
                return int(token_id)
        raise ValueError("Tokenizer has no valid placeholder token id")

    async def _build_latent_prompt(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        latent_embeds: torch.Tensor,
        latent_framing_before: str | None,
        latent_framing_after: str | None,
    ) -> dict:
        inject_kwargs = {
            "enable_thinking": self.enable_thinking,
            "add_generation_prompt": True,
            "placeholder_token_id": self._dummy_token_id(),
        }
        if latent_framing_before is not None:
            inject_kwargs["framing_before"] = latent_framing_before
        if latent_framing_after is not None:
            inject_kwargs["framing_after"] = latent_framing_after
        input_ids, positions, _, _ = await asyncio.to_thread(
            build_latent_injected_ids,
            messages,
            tools,
            self.tokenizer,
            int(latent_embeds.shape[0]),
            **inject_kwargs,
        )
        if len(positions) != int(latent_embeds.shape[0]):
            raise ValueError(
                f"Latent slot count {len(positions)} != latent rows {latent_embeds.shape[0]}"
            )
        ids = torch.as_tensor(input_ids, dtype=torch.long)
        with torch.no_grad():
            prompt_embeds = self.embed_layer(ids).detach()
        replacement = latent_embeds.detach().to(
            device="cpu", dtype=prompt_embeds.dtype,
        )
        prompt_embeds[torch.as_tensor(positions, dtype=torch.long)] = replacement
        return {"prompt_embeds": prompt_embeds.contiguous()}

    def _resolve_lora_request(self, name: str | None):
        if name is None:
            return None
        try:
            return self._lora_requests[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown vLLM LoRA {name!r}; loaded={sorted(self._lora_requests)}"
            ) from exc

    async def __call__(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        latent_embeds: Optional[torch.Tensor] = None,
        lora_path: Optional[str] = None,
        latent_framing_before: Optional[str] = None,
        latent_framing_after: Optional[str] = None,
    ) -> dict:
        async with self._generation_lease():
            if latent_embeds is None:
                prompt = await self._render_text_prompt(messages, tools)
            else:
                prompt = await self._build_latent_prompt(
                    messages,
                    tools,
                    latent_embeds,
                    latent_framing_before,
                    latent_framing_after,
                )
            engine = await self._ensure_engine()
            sampling_params = self._sampling_params()
            lora_request = self._resolve_lora_request(lora_path)
            last_exc = None
            for attempt in range(self.max_retries + 1):
                try:
                    request_id = uuid.uuid4().hex
                    final_output = None
                    async for output in engine.generate(
                        prompt,
                        sampling_params,
                        request_id=request_id,
                        lora_request=lora_request,
                    ):
                        final_output = output
                    if final_output is None or not final_output.outputs:
                        raise RuntimeError("vLLM returned no generation output")
                    return _parse_response(final_output.outputs[0].text)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt < self.max_retries:
                        wait = 2**attempt
                        print(
                            f"\n  [vLLM] {type(exc).__name__}: {exc}, "
                            f"retrying in {wait}s..."
                        )
                        await asyncio.sleep(wait)
        raise RuntimeError(
            f"vLLM call failed after {self.max_retries + 1} attempts"
        ) from last_exc




    @staticmethod
    def _resolve_lora_adapter_path(path: str, adapter_name: str) -> str:
        adapter_dir = Path(path)
        if (adapter_dir / "adapter_config.json").exists():
            return str(adapter_dir)
        nested = adapter_dir / adapter_name
        if (nested / "adapter_config.json").exists():
            return str(nested)
        raise FileNotFoundError(
            f"Could not find adapter_config.json in {adapter_dir} or {nested}"
        )

    def _parse_preloaded_loras(self, lora_paths) -> None:
        if not lora_paths:
            return
        if isinstance(lora_paths, dict):
            entries = [
                {"lora_name": name, "lora_path": path}
                for name, path in lora_paths.items()
            ]
        else:
            entries = list(lora_paths)
        for index, entry in enumerate(entries):
            if isinstance(entry, str):
                name, path = Path(entry).name, entry
            else:
                name = entry.get("lora_name") or entry.get("name") or f"adapter_{index}"
                path = entry.get("lora_path") or entry.get("path")
            if not path:
                raise ValueError(f"Invalid vLLM LoRA preload entry: {entry!r}")
            self._adapter_paths[str(name)] = self._resolve_lora_adapter_path(
                str(path), str(name),
            )

    async def _load_lora_into_engine(self, name: str, path: str):
        if name in self._lora_requests:
            return self._lora_requests[name]
        from vllm.lora.request import LoRARequest

        request = LoRARequest(
            lora_name=name,
            lora_int_id=self._next_lora_id,
            lora_path=path,
        )
        self._next_lora_id += 1
        result = await self.engine.add_lora(request)
        if result is False:
            raise RuntimeError(f"vLLM failed to load LoRA {name!r} from {path!r}")
        self._lora_requests[name] = request
        return request

    async def async_load_lora_adapter(
        self, name: str, path: str, pinned: bool = True,
    ) -> str:
        del pinned
        resolved = self._resolve_lora_adapter_path(path, name)
        self._adapter_paths[name] = resolved
        if self.engine is not None:
            await self._load_lora_into_engine(name, resolved)
        return resolved

    async def async_unload_lora_adapter(self, name: str) -> None:
        self._adapter_paths.pop(name, None)
        request = self._lora_requests.pop(name, None)
        if request is not None and self.engine is not None:
            result = await self.engine.remove_lora(request.lora_int_id)
            if result is False:
                raise RuntimeError(f"vLLM failed to unload LoRA {name!r}")


    @staticmethod
    def _torch_dtype(name: str):
        aliases = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
                   "fp16": torch.float16, "float16": torch.float16,
                   "fp32": torch.float32, "float32": torch.float32}
        try:
            return aliases[name.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported encoder dtype: {name}") from exc

    @asynccontextmanager
    async def encoder_phase(self):
        self._raise_if_failed()
        phase_engine = None
        transitioned = False

        async with self._lifecycle_operation_lock:
            self._raise_if_failed()
            phase_engine = self.engine
            if phase_engine is not None:
                self._require_sleep_mode()

            async with self._lifecycle_condition:
                if self._lifecycle_state is not VLLMLifecycleState.RUNNING:
                    raise RuntimeError(
                        "encoder_phase cannot overlap another lifecycle phase; "
                        f"got {self.lifecycle_state}"
                    )
                if phase_engine is None and self._active_generations:
                    raise RuntimeError(
                        "pre-engine encoder_phase requires no active generation"
                    )
                self._lifecycle_state = VLLMLifecycleState.DRAINING
                transitioned = True
                self._lifecycle_condition.notify_all()

            try:
                if phase_engine is not None:
                    async def _wait_for_generation_leases() -> None:
                        async with self._lifecycle_condition:
                            while self._active_generations:
                                await self._lifecycle_condition.wait()

                    await asyncio.wait_for(
                        _wait_for_generation_leases(),
                        timeout=self._lifecycle_drain_timeout_s,
                    )
                    if self.engine is not phase_engine:
                        raise RuntimeError(
                            "vLLM engine changed while entering encoder_phase"
                        )
                    await self._release_encoder()
                    await phase_engine.wait_for_requests_to_drain(
                        drain_timeout=self._lifecycle_drain_timeout_s,
                    )
                    await phase_engine.sleep(level=1, mode="wait")
                await self._set_lifecycle_state(VLLMLifecycleState.ENCODING)
            except BaseException as exc:
                if transitioned:
                    await self._await_cleanup_completion(
                        self._fail_lifecycle(exc),
                    )
                raise

        try:
            yield self
        finally:
            async with self._lifecycle_operation_lock:
                async with self._lifecycle_condition:
                    if self._lifecycle_state is not VLLMLifecycleState.ENCODING:
                        raise RuntimeError(
                            "encoder_phase exit requires ENCODING, got "
                            f"{self.lifecycle_state}"
                        )
                    self._lifecycle_state = VLLMLifecycleState.DRAINING
                    self._lifecycle_condition.notify_all()
                try:
                    async with self._encoder_quiescence(
                        fail_closed_on_timeout=True,
                    ):
                        try:
                            await self._drop_encoder_locked()
                            if self.engine is not phase_engine:
                                raise RuntimeError(
                                    "vLLM engine changed during encoder_phase"
                                )
                            if phase_engine is not None:
                                await phase_engine.wake_up(tags=["weights"])
                                await phase_engine.wake_up(
                                    tags=["kv_cache", "scheduling"],
                                )
                            await self._set_lifecycle_state(
                                VLLMLifecycleState.RUNNING,
                            )
                        except BaseException as exc:
                            await self._await_cleanup_completion(
                                self._fail_lifecycle(exc),
                            )
                            raise
                except BaseException as exc:
                    if self._lifecycle_state is not VLLMLifecycleState.FAILED:
                        await self._await_cleanup_completion(
                            self._fail_lifecycle(exc),
                        )
                    raise

    async def _ensure_encoder(self, lora_name: str | None):
        if (
            self.engine is not None
            and self._lifecycle_state is not VLLMLifecycleState.ENCODING
        ):
            raise RuntimeError(
                "with an initialized vLLM engine, encoder hidden states are "
                "available only inside encoder_phase()"
            )
        if self._lifecycle_state not in (
            VLLMLifecycleState.RUNNING,
            VLLMLifecycleState.ENCODING,
        ):
            raise RuntimeError(
                "encoder hidden states are unavailable during lifecycle state "
                f"{self.lifecycle_state}"
            )
        async with self._encoder_lock:
            if self._encoder_model is None:
                self._encoder_model = await asyncio.to_thread(
                    AutoModelForCausalLM.from_pretrained,
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=self._torch_dtype(self.encoder_dtype),
                    low_cpu_mem_usage=True,
                )
                self._encoder_model.eval().to(self.encoder_device)
            if lora_name is not None and self._encoder_adapter_name != lora_name:
                try:
                    adapter_path = self._adapter_paths[lora_name]
                except KeyError as exc:
                    raise KeyError(f"Encoder LoRA {lora_name!r} has not been registered") from exc
                from peft import PeftModel

                self._encoder_model = PeftModel.from_pretrained(
                    self._encoder_model,
                    adapter_path,
                    adapter_name=lora_name,
                    is_trainable=False,
                )
                self._encoder_model.eval()
                self._encoder_adapter_name = lora_name
        return self._encoder_model

    async def get_hidden_states(
        self,
        text: str,
        lora_path: str | None = None,
        timeout_s: float | None = None,
    ) -> torch.Tensor:
        async def _encode():
            async with self._encoder_activity_lease():
                model = await self._ensure_encoder(lora_path)
                encoded = await asyncio.to_thread(
                    self.tokenizer,
                    text,
                    return_tensors="pt",
                    return_token_type_ids=False,
                )
                encoded = {
                    key: value.to(self.encoder_device)
                    for key, value in encoded.items()
                }
                with torch.inference_mode():
                    output = model(
                        **encoded,
                        output_hidden_states=True,
                        use_cache=False,
                        return_dict=True,
                    )
                return output.hidden_states[-1][0].detach().to("cpu")

        return await asyncio.wait_for(_encode(), timeout_s) if timeout_s else await _encode()

    def _drop_encoder_sync(self):
        if self._encoder_model is not None:
            del self._encoder_model
            self._encoder_model = None
            self._encoder_adapter_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    async def _drop_encoder_locked(self):
        async with self._encoder_lock:
            await self._await_cleanup_completion(
                asyncio.to_thread(self._drop_encoder_sync),
            )

    async def _release_encoder(self):
        async with self._encoder_quiescence(fail_closed_on_timeout=True):
            await self._drop_encoder_locked()


    def build_weight_update_manifest(
        self,
        weights_path: str,
        *,
        session_id: str,
        chunk_count: int,
    ) -> dict[str, Any]:
        self._require_ipc_weight_transfer()
        if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(
                "session_id must match [A-Za-z0-9_.:-]{1,128}"
            )
        if (
            not isinstance(chunk_count, int)
            or isinstance(chunk_count, bool)
            or chunk_count <= 0
        ):
            raise ValueError("chunk_count must be a positive integer")
        inventory = _checkpoint_inventory(weights_path)
        return {
            "schema_version": _UPDATE_MANIFEST_SCHEMA_VERSION,
            "session_id": session_id,
            "backend": "ipc",
            "full_model": True,
            "weights_path": inventory["weights_path"],
            "tensor_count": inventory["tensor_count"],
            "tensor_names_sha256": inventory["tensor_names_sha256"],
            "tensor_specs_sha256": inventory["tensor_specs_sha256"],
            "weight_map_sha256": inventory["weight_map_sha256"],
            "chunk_count": chunk_count,
        }

    def _validate_weight_update_manifest(
        self,
        manifest: dict[str, Any],
        weights_path: str,
    ) -> dict[str, Any]:
        if not isinstance(manifest, dict):
            raise TypeError("weight update manifest must be a dict")
        keys = frozenset(manifest)
        if keys != _UPDATE_MANIFEST_KEYS:
            raise ValueError(
                "weight update manifest keys must match the schema exactly; "
                f"missing={sorted(_UPDATE_MANIFEST_KEYS - keys)}, "
                f"extra={sorted(keys - _UPDATE_MANIFEST_KEYS)}"
            )
        if (
            not isinstance(manifest["schema_version"], int)
            or isinstance(manifest["schema_version"], bool)
        ):
            raise TypeError("manifest schema_version must be an integer")
        if manifest["full_model"] is not True:
            raise ValueError("manifest full_model must be the boolean true")
        if not isinstance(manifest["backend"], str):
            raise TypeError("manifest backend must be a string")
        if not isinstance(manifest["weights_path"], str):
            raise TypeError("manifest weights_path must be a string")
        if (
            not isinstance(manifest["tensor_count"], int)
            or isinstance(manifest["tensor_count"], bool)
        ):
            raise TypeError("manifest tensor_count must be an integer")
        for digest_key in (
            "tensor_names_sha256",
            "tensor_specs_sha256",
            "weight_map_sha256",
        ):
            digest = manifest[digest_key]
            if (
                not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
            ):
                raise ValueError(f"manifest {digest_key} must be lowercase SHA256")
        expected = self.build_weight_update_manifest(
            weights_path,
            session_id=manifest["session_id"],
            chunk_count=manifest["chunk_count"],
        )
        mismatched = {
            key: {"expected": expected[key], "actual": manifest[key]}
            for key in _UPDATE_MANIFEST_KEYS
            if manifest[key] != expected[key]
        }
        if mismatched:
            raise ValueError(f"weight update manifest mismatch: {mismatched}")
        return _checkpoint_inventory(weights_path)

    @staticmethod
    def _embedding_shape(embedding: Any) -> tuple[int, ...]:
        weight = getattr(embedding, "weight", None)
        shape = getattr(weight, "shape", None)
        if shape is None:
            raise TypeError("CPU embedding layer has no weight.shape")
        return tuple(int(value) for value in shape)

    @staticmethod
    def _validate_ipc_update_info(update_info: dict[str, Any]) -> list[str]:
        if not isinstance(update_info, dict):
            raise TypeError("IPC update_info must be a dict")
        keys = frozenset(update_info)
        if keys != _IPC_UPDATE_INFO_KEYS:
            raise ValueError(
                "IPC update_info keys must match vLLM 0.25 packed IPC exactly; "
                f"missing={sorted(_IPC_UPDATE_INFO_KEYS - keys)}, "
                f"extra={sorted(keys - _IPC_UPDATE_INFO_KEYS)}"
            )
        if update_info["ipc_handles_pickled"] is not None:
            raise ValueError(
                "Pickled/HTTP IPC handles are forbidden; pass direct same-node "
                "ipc_handles through the authenticated loopback controller"
            )
        if update_info["packed"] is not True:
            raise ValueError("Only bounded-memory packed IPC chunks are accepted")

        names = update_info["names"]
        dtypes = update_info["dtype_names"]
        shapes = update_info["shapes"]
        tensor_sizes = update_info["tensor_sizes"]
        handles = update_info["ipc_handles"]
        if not isinstance(names, list) or not names:
            raise ValueError("IPC chunk names must be a non-empty list")
        if not all(isinstance(name, str) and name for name in names):
            raise ValueError("IPC chunk contains an invalid tensor name")
        if len(set(names)) != len(names):
            raise ValueError("IPC chunk contains duplicate tensor names")
        if not isinstance(dtypes, list) or len(dtypes) != len(names):
            raise ValueError("dtype_names length must equal names length")
        if not all(isinstance(dtype, str) and dtype for dtype in dtypes):
            raise ValueError("IPC chunk contains an invalid dtype name")
        if not isinstance(shapes, list) or len(shapes) != len(names):
            raise ValueError("shapes length must equal names length")
        if not all(
            isinstance(shape, list)
            and all(
                isinstance(dim, int) and not isinstance(dim, bool) and dim >= 0
                for dim in shape
            )
            for shape in shapes
        ):
            raise ValueError("IPC chunk contains an invalid tensor shape")
        if not isinstance(tensor_sizes, list) or len(tensor_sizes) != len(names):
            raise ValueError("tensor_sizes length must equal names length")
        if not all(
            isinstance(size, int) and not isinstance(size, bool) and size > 0
            for size in tensor_sizes
        ):
            raise ValueError("IPC chunk contains an invalid tensor size")
        if not isinstance(handles, dict) or not handles:
            raise ValueError("packed IPC chunk requires direct per-GPU handles")
        return names

    @classmethod
    def _validate_ipc_update_info_against_specs(
        cls,
        update_info: dict[str, Any],
        expected_specs: dict[str, dict[str, Any]],
    ) -> list[str]:
        names = cls._validate_ipc_update_info(update_info)
        for index, name in enumerate(names):
            expected = expected_specs.get(name)
            if expected is None:
                raise ValueError(f"IPC chunk contains unknown tensor {name!r}")
            actual_dtype = update_info["dtype_names"][index]
            actual_shape = list(update_info["shapes"][index])
            actual_nbytes = update_info["tensor_sizes"][index]
            mismatched: dict[str, dict[str, Any]] = {}
            if actual_dtype != expected["dtype"]:
                mismatched["dtype"] = {
                    "expected": expected["dtype"],
                    "actual": actual_dtype,
                }
            if actual_shape != expected["shape"]:
                mismatched["shape"] = {
                    "expected": expected["shape"],
                    "actual": actual_shape,
                }
            if actual_nbytes != expected["nbytes"]:
                mismatched["nbytes"] = {
                    "expected": expected["nbytes"],
                    "actual": actual_nbytes,
                }
            if mismatched:
                raise ValueError(
                    f"IPC tensor spec mismatch for {name!r}: {mismatched}"
                )
        return names

    async def release_gpu(self) -> None:
        self._require_sleep_mode()
        async with self._lifecycle_operation_lock:
            self._raise_if_failed()
            transitioned = False
            try:
                async with self._lifecycle_condition:
                    if self._lifecycle_state is VLLMLifecycleState.SLEEPING:
                        return
                    if self._lifecycle_state is not VLLMLifecycleState.RUNNING:
                        raise RuntimeError(
                            f"release_gpu requires RUNNING, got {self.lifecycle_state}"
                        )
                    self._lifecycle_state = VLLMLifecycleState.DRAINING
                    transitioned = True
                    self._lifecycle_condition.notify_all()
                async def _wait_for_generation_leases() -> None:
                    async with self._lifecycle_condition:
                        while self._active_generations:
                            await self._lifecycle_condition.wait()

                await asyncio.wait_for(
                    _wait_for_generation_leases(),
                    timeout=self._lifecycle_drain_timeout_s,
                )
                await self._release_encoder()
                if self.engine is not None:
                    await self.engine.wait_for_requests_to_drain(
                        drain_timeout=self._lifecycle_drain_timeout_s,
                    )
                    await self.engine.sleep(level=1, mode="wait")
                await self._set_lifecycle_state(VLLMLifecycleState.SLEEPING)
            except BaseException as exc:
                if transitioned:
                    await self._await_cleanup_completion(
                        self._fail_lifecycle(exc),
                    )
                raise

    async def resume_gpu(self, weights_path: str | None = None) -> None:
        self._require_sleep_mode()
        if weights_path is not None:
            raise NotImplementedError(
                "resume_gpu only supports an unchanged model. Use "
                "start/apply/finish_weight_update_session for full-model updates."
            )
        async with self._lifecycle_operation_lock:
            self._raise_if_failed()
            if self._lifecycle_state is VLLMLifecycleState.RUNNING:
                return
            if self._lifecycle_state is not VLLMLifecycleState.SLEEPING:
                raise RuntimeError(
                    f"resume_gpu requires SLEEPING, got {self.lifecycle_state}"
                )
            try:
                if self.engine is not None:
                    await self.engine.wake_up(tags=["weights"])
                    await self.engine.wake_up(tags=["kv_cache", "scheduling"])
                await self._set_lifecycle_state(VLLMLifecycleState.RUNNING)
            except BaseException as exc:
                await self._await_cleanup_completion(
                    self._fail_lifecycle(exc),
                )
                raise

    async def start_weight_update_session(
        self,
        manifest: dict[str, Any],
        *,
        weights_path: str,
    ) -> str:
        self._require_ipc_weight_transfer()
        self._raise_if_failed()
        inventory = await asyncio.to_thread(
            self._validate_weight_update_manifest, manifest, weights_path,
        )
        prepared_embed = await asyncio.to_thread(
            _load_embed_tokens_cpu, inventory["weights_path"],
        )
        if self._embedding_shape(prepared_embed) != self._embedding_shape(
            self.embed_layer
        ):
            raise ValueError(
                "Hot update cannot change the input embedding shape: "
                f"current={self._embedding_shape(self.embed_layer)}, "
                f"new={self._embedding_shape(prepared_embed)}"
            )

        async with self._lifecycle_operation_lock:
            self._raise_if_failed()
            if self._lifecycle_state is not VLLMLifecycleState.SLEEPING:
                raise RuntimeError(
                    "start_weight_update_session requires a completed release_gpu; "
                    f"got {self.lifecycle_state}"
                )
            if self.engine is None or not self._weight_transfer_initialized:
                raise RuntimeError(
                    "IPC hot update requires an initialized vLLM engine that has "
                    "already served rollouts"
                )
            session = _WeightUpdateSession(
                session_id=manifest["session_id"],
                weights_path=inventory["weights_path"],
                expected_names=frozenset(inventory["names"]),
                expected_specs=inventory["tensor_specs"],
                expected_chunk_count=manifest["chunk_count"],
                prepared_embed_layer=prepared_embed,
                seen_names=set(),
            )
            self._weight_update_session = session
            try:
                await self._set_lifecycle_state(VLLMLifecycleState.UPDATING)
                await self.engine.wake_up(tags=["weights"])
                await self.engine.start_weight_update()
            except BaseException as exc:
                await self._await_cleanup_completion(
                    self._fail_lifecycle(exc),
                )
                raise
            return session.session_id

    async def apply_weight_update_chunk(
        self,
        session_id: str,
        *,
        chunk_index: int,
        update_info: dict[str, Any],
    ) -> None:
        self._require_ipc_weight_transfer()
        async with self._lifecycle_operation_lock:
            self._raise_if_failed()
            session = self._weight_update_session
            if (
                self._lifecycle_state is not VLLMLifecycleState.UPDATING
                or session is None
            ):
                raise RuntimeError(
                    f"No active weight update session; state={self.lifecycle_state}"
                )
            if session_id != session.session_id:
                raise ValueError("weight update session_id mismatch")
            try:
                if (
                    not isinstance(chunk_index, int)
                    or isinstance(chunk_index, bool)
                    or chunk_index != session.next_chunk_index
                ):
                    raise ValueError(
                        f"Expected chunk_index={session.next_chunk_index}, "
                        f"got {chunk_index!r}"
                    )
                if chunk_index >= session.expected_chunk_count:
                    raise ValueError("Received more IPC chunks than declared")
                names = self._validate_ipc_update_info_against_specs(
                    update_info,
                    session.expected_specs,
                )
                unknown = set(names) - session.expected_names
                duplicate = set(names) & session.seen_names
                if unknown:
                    raise ValueError(
                        f"IPC chunk contains unknown tensor names: {sorted(unknown)}"
                    )
                if duplicate:
                    raise ValueError(
                        "IPC chunk repeats tensors from an earlier chunk: "
                        f"{sorted(duplicate)}"
                    )
                await self.engine.collective_rpc(
                    "update_weights",
                    kwargs={"update_info": dict(update_info)},
                )
            except BaseException as exc:
                await self._await_cleanup_completion(
                    self._fail_lifecycle(exc),
                )
                raise
            session.seen_names.update(names)
            session.next_chunk_index += 1

    async def finish_weight_update_session(self, session_id: str) -> None:
        self._require_ipc_weight_transfer()
        async with self._lifecycle_operation_lock:
            self._raise_if_failed()
            session = self._weight_update_session
            if (
                self._lifecycle_state is not VLLMLifecycleState.UPDATING
                or session is None
            ):
                raise RuntimeError(
                    f"No active weight update session; state={self.lifecycle_state}"
                )
            if session_id != session.session_id:
                raise ValueError("weight update session_id mismatch")
            try:
                if session.next_chunk_index != session.expected_chunk_count:
                    raise ValueError(
                        "IPC chunk count mismatch: "
                        f"received={session.next_chunk_index}, "
                        f"expected={session.expected_chunk_count}"
                    )
                if session.seen_names != set(session.expected_names):
                    missing = sorted(session.expected_names - session.seen_names)
                    extra = sorted(session.seen_names - session.expected_names)
                    raise ValueError(
                        f"Full-model tensor coverage mismatch: "
                        f"missing={missing}, extra={extra}"
                    )
                await self.engine.finish_weight_update()

                self.embed_layer = session.prepared_embed_layer
                self.model_path = session.weights_path
                self._engine_kwargs["model"] = session.weights_path

                await self.engine.wake_up(tags=["kv_cache", "scheduling"])
                if self._engine_kwargs.get("enable_prefix_caching", False):
                    reset_ok = await self.engine.reset_prefix_cache()
                    if reset_ok is not True:
                        raise RuntimeError(
                            "vLLM prefix cache reset failed after weight update"
                        )
                self._weight_update_session = None
                await self._set_lifecycle_state(VLLMLifecycleState.RUNNING)
            except BaseException as exc:
                await self._await_cleanup_completion(
                    self._fail_lifecycle(exc),
                )
                raise

    async def fail_weight_update_session(
        self,
        session_id: str,
        error: BaseException | str,
    ) -> None:
        self._require_ipc_weight_transfer()
        async with self._lifecycle_operation_lock:
            session = self._weight_update_session
            if session is None or session.session_id != session_id:
                raise ValueError("weight update session_id mismatch")
            exc = error if isinstance(error, BaseException) else RuntimeError(error)
            await self._await_cleanup_completion(
                self._fail_lifecycle(exc),
            )

    async def update_weights_from_path(self, weights_path: str) -> None:
        del weights_path
        raise NotImplementedError(
            "Direct path reload remains forbidden. Use the strict full-model "
            "packed-IPC weight update session API."
        )


def _parse_response(text: str) -> dict:
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": _extract_tool_calls(text),
    }


def _as_openai_tool_call(obj: dict) -> dict | None:
    function = obj.get("function") if isinstance(obj.get("function"), dict) else obj
    name = function.get("name") or function.get("function_name")
    if not name:
        return None
    args = function.get("arguments") or function.get("parameters") or {}
    return {
        "id": obj.get("id") or f"call_{uuid.uuid4().hex[:8]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args),
        },
    }


_OLMO3_JSON_NAME_LITERALS = {"null": None, "true": True, "false": False}


def _olmo3_ast_literal(node: ast.expr):
    if isinstance(node, ast.Constant):
        value = node.value
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise ValueError("OLMo3 arguments must be JSON-safe literal values")
    if isinstance(node, ast.List):
        return [_olmo3_ast_literal(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        if not all(
            isinstance(key, ast.Constant) and isinstance(key.value, str)
            for key in node.keys
        ):
            raise ValueError("OLMo3 dict keys must be strings")
        return {
            key.value: _olmo3_ast_literal(value)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.Name) and node.id in _OLMO3_JSON_NAME_LITERALS:
        return _OLMO3_JSON_NAME_LITERALS[node.id]
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.USub, ast.UAdd))
        and isinstance(node.operand, ast.Constant)
    ):
        value = _olmo3_ast_literal(node.operand)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return -value if isinstance(node.op, ast.USub) else value
        raise ValueError("OLMo3 numeric arguments must be finite JSON numbers")
    raise ValueError("OLMo3 arguments must be literals")


def _extract_olmo3_pythonic_calls(raw: str) -> list[dict]:
    compact = ", ".join(line.strip() for line in raw.splitlines() if line.strip())
    if not compact:
        return []
    try:
        expression = ast.parse(f"[{compact}]", mode="eval").body
    except SyntaxError:
        return []
    if not isinstance(expression, ast.List):
        return []

    calls: list[dict] = []
    try:
        for item in expression.elts:
            if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name) or item.args:
                return []
            arguments: dict[str, Any] = {}
            for keyword in item.keywords:
                if keyword.arg is None or keyword.arg in arguments:
                    return []
                arguments[keyword.arg] = _olmo3_ast_literal(keyword.value)
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": item.func.id,
                    "arguments": json.dumps(
                        arguments,
                        ensure_ascii=False,
                        allow_nan=False,
                    ),
                },
            })
    except (TypeError, ValueError):
        return []
    return calls


def _extract_tool_calls(text: str) -> list[dict] | None:
    parsed: list[dict] = []
    for raw in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL):
        try:
            obj = json.loads(raw.strip())
        except Exception:
            continue
        call = _as_openai_tool_call(obj) if isinstance(obj, dict) else None
        if call is not None:
            parsed.append(call)

    for raw in re.findall(r"<function_calls>(.*?)</function_calls>", text, re.DOTALL):
        pythonic = _extract_olmo3_pythonic_calls(raw)
        if pythonic:
            parsed.extend(pythonic)
            continue
        try:
            obj = json.loads(raw.strip())
        except Exception:
            continue
        items = obj if isinstance(obj, list) else [obj]
        for item in items:
            call = _as_openai_tool_call(item) if isinstance(item, dict) else None
            if call is not None:
                parsed.append(call)
    return parsed or None
