import hashlib
import json
import logging
import os
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MemoryBank:

    def __init__(
        self,
        encoder_model: str | None = None,
        allow_self_retrieval: bool = False,
    ):
        self._encoder_model = encoder_model
        self._encoder: Optional[SentenceTransformer] = None
        self.allow_self_retrieval = allow_self_retrieval
        self.entries: list[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self._query_cache_embs: Optional[np.ndarray] = None
        self._query_cache_index: Optional[dict[str, int]] = None

    @property
    def encoder(self) -> SentenceTransformer:
        if self._encoder is None:
            if self._encoder_model is None:
                raise RuntimeError("encoder_model is required when the query cache is unavailable")
            self._encoder = SentenceTransformer(self._encoder_model)
        return self._encoder

    def add(self, instruction: str, trajectory, success: bool) -> None:
        if not instruction:
            raise ValueError("MemoryBank.add requires a non-empty instruction.")
        self.entries.append({
            "instruction": instruction,
            "trajectory": trajectory,
            "success": success,
        })

    def build_index(self) -> None:
        if not self.entries:
            raise ValueError("No entries to index.")
        texts = [e["instruction"] for e in self.entries]
        self.embeddings = self.encoder.encode(
            texts, show_progress_bar=True, normalize_embeddings=True,
        ).astype("float32")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        logger.info("Built FAISS index: %d vectors, dim=%d", self.index.ntotal, dim)

    def _encode_queries(self, texts: list[str]) -> np.ndarray:
        if self._query_cache_embs is not None:
            dim = self._query_cache_embs.shape[1]
            result = np.empty((len(texts), dim), dtype="float32")
            misses: list[str] = []
            miss_indices: list[int] = []
            for i, text in enumerate(texts):
                row = self._query_cache_index.get(_query_key(text))
                if row is not None:
                    result[i] = self._query_cache_embs[row]
                else:
                    misses.append(text)
                    miss_indices.append(i)
            if misses:
                logger.warning(
                    "Query cache miss for %d/%d queries, falling back to encoder",
                    len(misses), len(texts),
                )
                miss_embs = self.encoder.encode(
                    misses, normalize_embeddings=True,
                ).astype("float32")
                for j, idx in enumerate(miss_indices):
                    result[idx] = miss_embs[j]
            return result
        return self.encoder.encode(
            texts, normalize_embeddings=True,
        ).astype("float32")

    def retrieve_many(self, queries: list[str], k: int = 3) -> list[list[dict]]:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index first.")
        if not queries:
            return []
        query_embs = self._encode_queries(queries)
        fetch_k = min(k + 10, self.index.ntotal)
        scores, indices = self.index.search(query_embs, fetch_k)
        all_results: list[list[dict]] = []
        for query, query_scores, query_indices in zip(queries, scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx == -1:
                    continue
                entry = self.entries[idx]
                if not self.allow_self_retrieval and entry["instruction"] == query:
                    continue
                results.append({**entry, "score": float(score)})
                if len(results) >= k:
                    break
            all_results.append(results)
        return all_results

    def build_query_cache(self, query_texts: list[str], save_dir: str) -> None:
        embs = self.encoder.encode(
            query_texts, show_progress_bar=True, normalize_embeddings=True,
        ).astype("float32")
        index_map: dict[str, int] = {}
        for i, text in enumerate(query_texts):
            index_map[_query_key(text)] = i
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "query_cache.npy"), embs)
        with open(os.path.join(save_dir, "query_cache_index.json"), "w") as f:
            json.dump(index_map, f)
        self._query_cache_embs = embs
        self._query_cache_index = index_map
        logger.info("Built query cache: %d queries, saved to %s", len(query_texts), save_dir)

    def load_query_cache(self, path: str) -> bool:
        cache_emb = os.path.join(path, "query_cache.npy")
        cache_idx = os.path.join(path, "query_cache_index.json")
        if not (os.path.exists(cache_emb) and os.path.exists(cache_idx)):
            return False
        self._query_cache_embs = np.load(cache_emb)
        with open(cache_idx) as f:
            self._query_cache_index = json.load(f)
        logger.info("Loaded query cache: %d entries from %s", len(self._query_cache_index), path)
        return True

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings)
        meta = [
            {"instruction": e["instruction"], "success": e["success"]}
            for e in self.entries
        ]
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False)
        with open(os.path.join(path, "trajectories.jsonl"), "w") as f:
            for e in self.entries:
                f.write(json.dumps(e["trajectory"], ensure_ascii=False) + "\n")
        if self._query_cache_embs is not None:
            np.save(os.path.join(path, "query_cache.npy"), self._query_cache_embs)
            with open(os.path.join(path, "query_cache_index.json"), "w") as f:
                json.dump(self._query_cache_index, f)
        logger.info("Saved memory bank to %s (%d entries)", path, len(self.entries))

    def load(self, path: str) -> None:
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        with open(os.path.join(path, "trajectories.jsonl")) as f:
            trajectories = [json.loads(line) for line in f]
        self.entries = [
            {**m, "trajectory": t} for m, t in zip(meta, trajectories)
        ]
        self.load_query_cache(path)
        logger.info("Loaded memory bank from %s (%d entries)", path, len(self.entries))


def _query_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
