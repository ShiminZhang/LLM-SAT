"""Shared runtime services for embedding and FAISS index access.

This module guarantees one embedding model load and one index repository
instance per Python process via a singleton runtime.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# torch/transformers are imported lazily inside EmbeddingService: they take
# minutes to import on this cluster and are only needed when embeddings are
# actually computed, not for pure record/re-ranking logic or tests.

from .layout import id_map_file_path, index_file_path, records_file_path
from .types import ExperienceRecord, OutcomeLabel, PoolName


def _last_token_pool(last_hidden_states: "Tensor", attention_mask: "Tensor") -> "Tensor":
    """Compute sentence embeddings using Qwen3 last-token pooling.

    Args:
        last_hidden_states: Tensor of shape `[batch, seq_len, hidden_dim]`.
        attention_mask: Tensor of shape `[batch, seq_len]` with nonzero tokens to keep.

    Returns:
        Tensor: Tensor of shape `[batch, hidden_dim]` pooled at the last valid token.

    Notes:
        This matches the official Qwen3 embedding guidance:
        - when inputs are left-padded, use the final token embedding directly
        - otherwise, gather the embedding at each sequence's final non-padding token
    """

    import torch

    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]

    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


class EmbeddingService:
    """Embedding service backed by `Qwen/Qwen3-Embedding-0.6B`.

    The model and tokenizer are loaded once per process when the singleton
    runtime is created.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: Optional[str] = "cpu",
    ) -> None:
        """Initialize embedding model and tokenizer.

        Args:
            model_name: Hugging Face model ID.
            device: Optional torch device override (`cpu`, `cuda`, etc.).
        """

        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Qwen3 embedding guidance recommends left-padding when using last-token pooling.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    def encode(self, texts: List[str], max_length: int = 8192) -> np.ndarray:
        """Encode text into L2-normalized float32 embeddings.

        Args:
            texts: List of input strings.
            max_length: Maximum token length per input.

        Returns:
            np.ndarray: Array of shape `[len(texts), embedding_dim]`, dtype float32.
        """

        import torch
        import torch.nn.functional as F

        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        with torch.inference_mode():
            return self._encode_impl(torch, F, texts, max_length)

    def _encode_impl(self, torch, F, texts: List[str], max_length: int) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        outputs = self.model(**encoded)
        pooled = _last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])
        # Some embedding models may return bfloat16 tensors, which NumPy cannot
        # convert directly on many setups. Cast to float32 first.
        pooled = pooled.to(dtype=torch.float32)
        pooled = F.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu().numpy().astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimensionality.

        Returns:
            int: Hidden dimension of model embeddings.
        """

        return int(self.model.config.hidden_size)


class FaissPartition:
    """In-memory + on-disk FAISS partition for one `(pool, outcome)` pair."""

    def __init__(
        self,
        data_root: str | Path,
        pool_name: PoolName,
        outcome: OutcomeLabel,
        embedding_dim: int,
    ) -> None:
        """Load or initialize FAISS partition state.

        Args:
            data_root: Root storage directory.
            pool_name: Pool namespace.
            outcome: Outcome partition.
            embedding_dim: Embedding vector dimension.
        """

        self.data_root = Path(data_root)
        self.pool_name = pool_name
        self.outcome = outcome
        self.embedding_dim = embedding_dim

        self.index_path = index_file_path(self.data_root, pool_name, outcome)
        self.records_path = records_file_path(self.data_root, pool_name, outcome)
        self.id_map_path = id_map_file_path(self.data_root, pool_name, outcome)

        self._faiss = self._import_faiss()
        self.index = self._load_or_create_index()
        self.id_map, self.records = self._load_metadata()

    @staticmethod
    def _import_faiss():
        """Import FAISS module.

        Returns:
            module: Imported FAISS module.

        Raises:
            ImportError: If FAISS is not installed.
        """

        try:
            import faiss  # type: ignore[import-not-found]

            return faiss
        except ImportError as exc:
            raise ImportError(
                "FAISS is required for experience pool retrieval. Install faiss-cpu."
            ) from exc

    def _load_or_create_index(self):
        """Load existing FAISS index or create a new inner-product index.

        Returns:
            Any: FAISS index object.
        """

        if self.index_path.exists():
            return self._faiss.read_index(str(self.index_path))
        return self._faiss.IndexFlatIP(self.embedding_dim)

    def _load_metadata(self) -> Tuple[List[str], Dict[str, dict]]:
        """Load ID map and records metadata from JSON files.

        Returns:
            tuple[list[str], dict[str, dict]]: (`id_map`, `records_by_id`).
        """

        id_map: List[str] = []
        records: Dict[str, dict] = {}

        if self.id_map_path.exists():
            id_map = json.loads(self.id_map_path.read_text(encoding="utf-8"))

        if self.records_path.exists():
            records = json.loads(self.records_path.read_text(encoding="utf-8"))

        return id_map, records

    def add(self, record_id: str, record: ExperienceRecord, vector: np.ndarray) -> bool:
        """Insert one record/vector into the partition if not duplicated.

        Args:
            record_id: Deterministic unique ID.
            record: Typed record payload.
            vector: Embedding vector with shape `[embedding_dim]`.

        Returns:
            bool: True if inserted; False if duplicate record ID already exists.
        """

        if record_id in self.records:
            return False

        vector_2d = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        self.index.add(vector_2d)
        self.id_map.append(record_id)
        self.records[record_id] = asdict(record)
        return True

    def search(self, query_vector: np.ndarray, top_k: int) -> List[Tuple[str, float, dict]]:
        """Search nearest records by cosine similarity (via normalized inner product).

        Args:
            query_vector: Query embedding vector with shape `[embedding_dim]`.
            top_k: Maximum number of records to return.

        Returns:
            list[tuple[str, float, dict]]: `(record_id, score, record_dict)` results.
        """

        if self.index.ntotal == 0 or top_k <= 0:
            return []

        top_k = min(top_k, int(self.index.ntotal))
        vector_2d = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
        scores, indices = self.index.search(vector_2d, top_k)

        results: List[Tuple[str, float, dict]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            record_id = self.id_map[idx]
            payload = self.records.get(record_id)
            if payload is None:
                continue
            results.append((record_id, float(score), payload))
        return results

    def size(self) -> int:
        """Return number of records in this partition.

        Returns:
            int: Number of indexed records.
        """

        return len(self.id_map)

    def save(self) -> None:
        """Persist index and metadata files to disk.

        Returns:
            None
        """

        self._faiss.write_index(self.index, str(self.index_path))
        self.id_map_path.write_text(json.dumps(self.id_map, indent=2, ensure_ascii=False), encoding="utf-8")
        self.records_path.write_text(json.dumps(self.records, indent=2, ensure_ascii=False), encoding="utf-8")


class FaissIndexRepository:
    """Repository for cached FAISS partitions across all pools/outcomes."""

    def __init__(self, data_root: str | Path, embedding_dim: int) -> None:
        """Initialize repository.

        Args:
            data_root: Root storage directory.
            embedding_dim: Shared embedding dimension.
        """

        self.data_root = Path(data_root)
        self.embedding_dim = embedding_dim
        self._partitions: Dict[Tuple[PoolName, OutcomeLabel], FaissPartition] = {}

    def get_partition(self, pool_name: PoolName, outcome: OutcomeLabel) -> FaissPartition:
        """Return cached partition for `(pool_name, outcome)`.

        Args:
            pool_name: Pool namespace.
            outcome: Outcome partition.

        Returns:
            FaissPartition: Cached or newly-loaded partition instance.
        """

        key = (pool_name, outcome)
        if key not in self._partitions:
            self._partitions[key] = FaissPartition(
                data_root=self.data_root,
                pool_name=pool_name,
                outcome=outcome,
                embedding_dim=self.embedding_dim,
            )
        return self._partitions[key]

    def save_all(self) -> None:
        """Persist all cached partitions to disk.

        Returns:
            None
        """

        for partition in self._partitions.values():
            partition.save()


class SharedRuntime:
    """Process-wide singleton container for embedding and index services."""

    _instance: Optional["SharedRuntime"] = None
    _lock = threading.Lock()

    def __init__(self, data_root: str | Path) -> None:
        """Construct runtime services.

        Args:
            data_root: Root storage directory.
        """

        self.embedding = EmbeddingService()
        self.index_repo = FaissIndexRepository(data_root=data_root, embedding_dim=self.embedding.embedding_dim)

    @classmethod
    def get_instance(cls, data_root: str | Path) -> "SharedRuntime":
        """Get singleton runtime instance.

        Args:
            data_root: Root storage directory for index repository.

        Returns:
            SharedRuntime: Singleton runtime with shared model/index services.
        """

        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(data_root)
        return cls._instance
