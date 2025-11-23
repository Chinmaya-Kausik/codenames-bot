"""
Embedding model utilities with lightweight fallbacks.

This module centralizes loading of SentenceTransformer models while providing
a deterministic hash-based embedding backend for environments where the real
models are unavailable or too heavy (e.g., CI).
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, Sequence, Tuple, Dict

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency elsewhere
    torch = None  # type: ignore


DEFAULT_DIM = 384
_GLOBAL_VOCAB_CACHE: Dict[tuple[str, str], torch.Tensor] = {}


class HashSentenceTransformer:
    """
    Lightweight stand-in for SentenceTransformer that hashes tokens to vectors.

    Produces deterministic unit-length embeddings so agents can run during tests
    without pulling large models from disk or the network.
    """

    def __init__(self, model_name: str = "hash-baseline", dim: int = DEFAULT_DIM):
        self.model_name = model_name
        self.dim = dim
        self.device = None

    def to(self, device):  # noqa: D401 - mirrors SentenceTransformer API
        """Mimic SentenceTransformer.to by storing device preference."""
        self.device = device
        return self

    def eval(self):
        """No-op for API compatibility."""
        return self

    def encode(  # noqa: D401 - mirrors SentenceTransformer API
        self,
        sentences: Sequence[str] | str,
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = False,
        device: str | torch.device | None = None,
    ):
        """
        Encode text into deterministic embeddings.

        Args mirror sentence_transformers.SentenceTransformer.encode, but only
        the commonly used parameters are implemented.
        """
        if isinstance(sentences, str):
            batch = [sentences]
            single_input = True
        else:
            batch = list(sentences)
            single_input = False

        embeddings = self._encode_batch(batch)

        if convert_to_tensor:
            if torch is None:
                raise ImportError("PyTorch is required for tensor outputs.")
            tensor = torch.as_tensor(
                embeddings,
                dtype=torch.float32,
                device=device or self.device
            )
            return tensor[0] if single_input else tensor

        # Default to numpy output just like SentenceTransformer
        embeddings = embeddings.astype(np.float32)
        result = embeddings[0] if single_input else embeddings

        if convert_to_numpy or not single_input:
            return result

        # For single inputs, return 1D array
        return result

    def _encode_batch(self, sentences: Sequence[str]) -> np.ndarray:
        """Hash each sentence deterministically to a vector."""
        vectors = []
        for text in sentences:
            if text is None:
                text = ""
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            base = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)

            # Tile to requested dimension and normalize
            reps = (self.dim + base.size - 1) // base.size
            vec = np.tile(base, reps)[: self.dim]
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            vectors.append(vec)

        return np.stack(vectors, axis=0)


def _should_force_fake() -> bool:
    """Check global flag for forcing fake embeddings."""
    return os.environ.get("CODENAMES_FAKE_EMBEDDINGS", "0") in {"1", "true", "True"}


def load_embedding_model(
    model_name: str,
    device: torch.device | None = None,
) -> tuple[object, bool]:
    """
    Load a SentenceTransformer model with an automatic hash fallback.

    Returns a tuple of (model, is_fake_backend).
    """
    use_fake = _should_force_fake()
    model = None

    if not use_fake:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer(model_name)
        except Exception:
            use_fake = True

    if use_fake or model is None:
        model = HashSentenceTransformer(model_name)

    if device is not None and hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    return model, use_fake


def encode_vocab(
    model,
    tokens: Sequence[str],
    device: torch.device,
    cache_key: tuple[str, str] | None = None
) -> torch.Tensor:
    """
    Encode a list of tokens once and cache them for reuse.

    Args:
        model: SentenceTransformer or HashSentenceTransformer
        tokens: Iterable of strings
        device: target torch device
        cache_key: optional key (> (model_name, identifier)) for reuse
    """
    if cache_key is not None and cache_key in _GLOBAL_VOCAB_CACHE:
        cached = _GLOBAL_VOCAB_CACHE[cache_key]
        if cached.device == device:
            return cached
        return cached.to(device)

    if not tokens:
        empty = torch.zeros(0, getattr(model, "dim", DEFAULT_DIM), device=device)
        if cache_key is not None:
            _GLOBAL_VOCAB_CACHE[cache_key] = empty
        return empty

    embeddings = model.encode(
        list(tokens),
        convert_to_tensor=True,
        device=str(device)
    )
    if embeddings.device != device:
        embeddings = embeddings.to(device)

    if cache_key is not None:
        _GLOBAL_VOCAB_CACHE[cache_key] = embeddings.clone()

    return embeddings
