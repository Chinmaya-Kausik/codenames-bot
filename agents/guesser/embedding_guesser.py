"""
Embedding-based guesser agent using semantic similarity.
"""

from __future__ import annotations

from typing import Optional, Any
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from agents.guesser.base_guesser import BaseGuesser, GuesserParams
from utils.embeddings import load_embedding_model, encode_vocab
from utils.device import get_device, get_device_name


class EmbeddingGuesser(BaseGuesser):
    """
    Guesser that uses semantic embeddings to interpret clues.

    Strategy:
    1. Compute embedding for the clue
    2. Compute embeddings for all unrevealed words
    3. Rank unrevealed words by similarity to clue
    4. Guess the most similar word if above threshold
    5. Consider ending turn early based on confidence
    """

    # Class-level cache for embedding model per (model_name, device)
    _model_cache: dict[tuple[str, str], tuple[object, bool]] = {}
    _fake_warning_printed: bool = False

    def __init__(self, team: str, params: Optional[GuesserParams] = None):
        """
        Initialize embedding guesser.

        Args:
            team: Team this guesser plays for
            params: GuesserParams with configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for accelerated inference. "
                "Install it with: pip install torch"
            )

        super().__init__(team, params)

        # Load embedding model
        self.device = get_device()
        model_name = params.embedding_model_name if params and params.embedding_model_name else "all-MiniLM-L6-v2"
        cache_key = (model_name, str(self.device))
        if cache_key not in EmbeddingGuesser._model_cache:
            EmbeddingGuesser._model_cache[cache_key] = load_embedding_model(model_name, device=self.device)

        self.model, self.using_fake_embeddings = EmbeddingGuesser._model_cache[cache_key]
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        if self.using_fake_embeddings and not EmbeddingGuesser._fake_warning_printed:
            logger.info("EmbeddingGuesser using hash embeddings (set CODENAMES_FAKE_EMBEDDINGS=0 for real models).")
            EmbeddingGuesser._fake_warning_printed = True

        self.word_embeddings_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self.max_cache_size = 1000  # Limit cache size to prevent memory leak

    def reset(self):
        """
        Reset the guesser state between games.

        Clears the word embeddings cache to prevent memory leak.
        """
        self.word_embeddings_cache.clear()

    def get_guess(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Make a guess based on semantic similarity to the clue with PyTorch vectorization.

        Args:
            obs: Guesser observation from environment

        Returns:
            Action dict with tile indices
        """
        # Extract data from observation
        words_list = obs["words"]  # List of word lists
        revealed = torch.as_tensor(obs["revealed"], device=self.device, dtype=torch.bool)  # [B, N]
        current_clues = np.asarray(obs["current_clue"], dtype=object)
        remaining_guesses = torch.as_tensor(obs["remaining_guesses"], device=self.device, dtype=torch.int32)  # [B]
        batch_size = len(words_list)
        board_size = revealed.shape[1]

        # Step 1: Vectorized collection of unrevealed words
        unrevealed = ~revealed  # [B, N]

        # Collect unique clues and words using numpy for deterministic ordering
        clue_unique, clue_indices, clue_inverse = np.unique(
            current_clues,
            return_index=True,
            return_inverse=True
        )
        clue_order = np.argsort(clue_indices)
        clue_unique = clue_unique[clue_order]
        clue_inverse = clue_order[clue_inverse]

        words_array = np.asarray(words_list, dtype=object)
        flat_words = words_array.reshape(-1)
        word_unique, word_indices, word_inverse = np.unique(
            flat_words,
            return_index=True,
            return_inverse=True
        )
        word_order = np.argsort(word_indices)
        word_unique = word_unique[word_order]
        order_map = np.empty_like(word_order)
        order_map[word_order] = np.arange(word_order.size)
        word_inverse = order_map[word_inverse].reshape(batch_size, board_size)

        # Step 2: Batch encode all text and convert to PyTorch tensors
        # Cache board word embeddings to avoid re-encoding same boards
        # Build list of all words needed, checking cache first
        words_to_encode = set(word_unique.tolist())

        # Check cache and collect cached word embeddings
        word_embeddings_dict = {}
        for cached_key, cached_board_embs in self.word_embeddings_cache.items():
            for i, word in enumerate(cached_key):
                if word in words_to_encode and word not in word_embeddings_dict:
                    word_embeddings_dict[word] = cached_board_embs[i]

        # Determine which words still need encoding
        words_needing_encoding = [w for w in word_unique if w not in word_embeddings_dict]

        # Encode clues and uncached words together
        all_text = clue_unique.tolist() + words_needing_encoding
        if len(all_text) == 0:
            return {"tile_index": torch.zeros(batch_size, dtype=torch.int32, device=self.device)}

        # Get embeddings directly as tensors on device (avoid numpy conversion)
        all_embeddings = self.model.encode(
            all_text,
            convert_to_tensor=True,
            device=str(self.device)
        )

        clue_embeddings = all_embeddings[:len(clue_unique)]  # [n_clues, D]
        new_word_embeddings = all_embeddings[len(clue_unique):]  # [n_new_words, D]

        # Add newly encoded words to dict
        for i, word in enumerate(words_needing_encoding):
            word_embeddings_dict[word] = new_word_embeddings[i]

        # Build final word_embeddings in correct order
        word_embeddings = torch.stack([word_embeddings_dict[w] for w in word_unique])

        # Update cache with boards we haven't seen before
        for words in words_array:
            key = tuple(words)
            if key not in self.word_embeddings_cache:
                # Clear cache if it's getting too large
                if len(self.word_embeddings_cache) >= self.max_cache_size:
                    self.word_embeddings_cache.clear()

                # Cache this board's embeddings
                board_embs = torch.stack([word_embeddings_dict[w] for w in words])
                self.word_embeddings_cache[key] = board_embs

        # Step 3: Compute all pairwise similarities [n_clues, n_words]
        clue_norms = torch.nn.functional.normalize(clue_embeddings, dim=1)
        word_norms = torch.nn.functional.normalize(word_embeddings, dim=1)
        all_similarities = clue_norms @ word_norms.T  # [n_clues, n_words]

        # Step 4: Vectorized guess selection
        word_inverse_tensor = torch.from_numpy(word_inverse).to(self.device)
        clue_inverse_tensor = torch.from_numpy(clue_inverse).to(self.device)

        board_sims = all_similarities[clue_inverse_tensor.unsqueeze(1), word_inverse_tensor]  # [B, N]
        masked_sims = board_sims.masked_fill(~unrevealed, float("-inf"))
        sorted_sims, sorted_idx = torch.sort(masked_sims, dim=1, descending=True)

        best_sim = sorted_sims[:, 0]
        best_idx = sorted_idx[:, 0]

        second_sim = sorted_sims[:, 1] if board_size > 1 else torch.full_like(best_sim, float("-inf"))
        remaining = remaining_guesses.to(self.device)

        tile_indices = torch.zeros(batch_size, dtype=torch.int64, device=self.device)
        tile_indices = torch.where(unrevealed.any(dim=1), best_idx, tile_indices)

        tile_positions = torch.arange(board_size, device=self.device).unsqueeze(0).expand(batch_size, -1)
        first_unrevealed = torch.where(
            unrevealed,
            tile_positions,
            torch.full_like(tile_positions, board_size)
        ).amin(dim=1)
        first_unrevealed = torch.clamp(first_unrevealed, max=max(board_size - 1, 0))
        first_unrevealed = torch.where(unrevealed.any(dim=1), first_unrevealed, torch.zeros_like(first_unrevealed))

        low_conf = best_sim < self.params.similarity_threshold
        tile_indices = torch.where(low_conf, first_unrevealed, tile_indices)

        final_guess = remaining <= 1
        cautious = (best_sim < self.params.confidence_threshold) & final_guess
        tile_indices = torch.where(cautious, first_unrevealed, tile_indices)

        multi_candidate = (unrevealed.sum(dim=1) > 1)
        tie_mask = (best_sim - second_sim < 0.1) & (multi_candidate) & (
            final_guess | (best_sim < self.params.confidence_threshold)
        )
        tile_indices = torch.where(tie_mask, first_unrevealed, tile_indices)

        return {"tile_index": tile_indices.to(torch.int32)}
