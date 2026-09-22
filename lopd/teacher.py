from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from memory.compression import compress_hiddens, encode_texts_via_pytorch, latent_cache_key, trajectory_text
from memory.compressor import load_compressor
from memory.lora import attach_lora, base_model_view, iter_lora_named_parameters
from memory.prompt_protocol import resolve_compressor_prompt_protocol
from memory.serialization import IGNORE_INDEX, build_supervised_inputs, inject_latent_tokens_into_embeds


class LatentTeacher:
    def __init__(
        self,
        model_path: str,
        compressor_dir: str,
        *,
        device: str,
        enable_thinking: bool,
        trainable: bool = False,
        cold_start_dir: str | None = None,
        anchor_cache_dir: str | None = None,
    ):
        if trainable and (cold_start_dir is None or anchor_cache_dir is None):
            raise ValueError("a trainable teacher needs cold_start_dir and anchor_cache_dir for the anchor")
        self.device = device
        self.anchor_cache_dir = anchor_cache_dir
        self.enable_thinking = enable_thinking
        self.trainable = trainable

        self.qformer, adapter_dir, manifest = load_compressor(compressor_dir, device=device)
        self.qformer = self.qformer.bfloat16().requires_grad_(trainable)
        self.qformer.train(trainable)
        self.protocol = resolve_compressor_prompt_protocol(manifest["prompt_protocol"])
        self.task_cond = bool(manifest["task_cond"])
        self.adapter_name = manifest["adapter_name"]

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
        model.requires_grad_(False)
        model = attach_lora(model.to(device), adapter_path=adapter_dir, adapter_name=self.adapter_name, is_trainable=trainable)

        self.cold_start_qformer = None
        self.cold_start_adapter = None
        if cold_start_dir is not None:
            self.cold_start_qformer, adapter0_dir, _ = load_compressor(cold_start_dir, device=device)
            self.cold_start_qformer = self.cold_start_qformer.bfloat16().requires_grad_(False).eval()
            self.cold_start_adapter = self.adapter_name + "0"
            model.load_adapter(adapter0_dir, adapter_name=self.cold_start_adapter, is_trainable=False)
            model.set_adapter(self.adapter_name, inference_mode=not trainable)

        if trainable:
            model.train()
            model.gradient_checkpointing_enable()
            model.config.use_cache = False
        else:
            model.eval()
        self.model = model
        self.backbone = base_model_view(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        qformer = [(f"qformer.{name}", p) for name, p in self.qformer.named_parameters() if p.requires_grad]
        lora = [(name, p) for name, p in iter_lora_named_parameters(self.model) if f".{self.adapter_name}." in name]
        return qformer + lora

    @torch.no_grad()
    def topk_log_probs(self, rollout: dict, *, top_k: int, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        log_probs, _, _, _ = self._supervised_log_probs(rollout, max_seq_len)
        values, indices = torch.topk(log_probs, k=top_k, dim=-1, sorted=True)
        return values.cpu(), indices.cpu()

    def privileged_log_probs(self, rollout: dict, support: torch.Tensor, *, max_seq_len: int) -> dict:
        log_probs, next_tokens, latents, latents0 = self._supervised_log_probs(rollout, max_seq_len)
        n = min(log_probs.shape[0], support.shape[0])
        return {
            "lp_at_token": log_probs.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1),
            "support_lp": log_probs[:n].gather(-1, support[:n].to(log_probs.device)),
            "c_phi": latents,
            "c_phi0": latents0,
        }

    def _supervised_log_probs(self, rollout: dict, max_seq_len: int):
        latents, latents0 = self._compose(rollout)
        input_ids, positions, target_ids = build_supervised_inputs(
            rollout["trajectory"], self.tokenizer, enable_thinking=self.enable_thinking, max_seq_len=max_seq_len,
            n_latent=latents.shape[0], protocol=self.protocol,
        )

        ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)
        with torch.no_grad():
            embeds = self.backbone.get_input_embeddings()(ids)
        embeds = inject_latent_tokens_into_embeds(embeds, positions, latents)

        target_ids = target_ids.to(self.device)
        sup_mask = target_ids[1:] != IGNORE_INDEX
        if not sup_mask.any():
            raise ValueError(f"rollout {rollout['task_id']}: no supervised tokens")

        hidden = self.backbone.model(inputs_embeds=embeds, use_cache=False).last_hidden_state[0]
        logits = self.backbone.lm_head(hidden[:-1][sup_mask])
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return log_probs, target_ids[1:][sup_mask], latents, latents0

    def _compose(self, rollout: dict) -> tuple[torch.Tensor, torch.Tensor | None]:
        texts = tuple(trajectory_text(entry, self.tokenizer, enable_thinking=self.enable_thinking) for entry in rollout["retrieved"])
        task_text = rollout["query_text"] if self.task_cond else None
        latents0 = self._anchor_latents(texts, task_text) if self.cold_start_qformer is not None else None
        hiddens = encode_texts_via_pytorch(
            self.model, texts, self.tokenizer, self.device,
            train_encoder_lora=self.trainable, task_text=task_text, protocol=self.protocol,
        )
        return compress_hiddens(self.qformer, hiddens), latents0

    def _anchor_latents(self, texts: tuple[str, ...], task_text: str | None) -> torch.Tensor:
        paths = [os.path.join(self.anchor_cache_dir, latent_cache_key(text, task_text) + ".pt") for text in texts]
        missing = [i for i, path in enumerate(paths) if not os.path.isfile(path)]
        if missing:
            self.model.set_adapter(self.cold_start_adapter, inference_mode=True)
            hiddens0 = encode_texts_via_pytorch(
                self.model, tuple(texts[i] for i in missing), self.tokenizer, self.device,
                train_encoder_lora=False, task_text=task_text, protocol=self.protocol,
            )
            self.model.set_adapter(self.adapter_name, inference_mode=not self.trainable)
            os.makedirs(self.anchor_cache_dir, exist_ok=True)
            with torch.no_grad():
                for i, hidden in zip(missing, hiddens0):
                    latent = self.cold_start_qformer(hidden.to(next(self.cold_start_qformer.parameters()).device)).squeeze(0)
                    torch.save(latent.cpu(), paths[i] + ".tmp")
                    os.replace(paths[i] + ".tmp", paths[i])
        return torch.cat([torch.load(path, map_location=self.device, weights_only=True) for path in paths], dim=0)
