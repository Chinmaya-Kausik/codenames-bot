"""
Embedding-based spymaster agent using semantic similarity.
"""

from __future__ import annotations

import logging
import random
from typing import Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from agents.spymaster.base_spymaster import BaseSpymaster, SpymasterParams
from utils.embeddings import load_embedding_model, encode_vocab
from utils.wordlist import WORD_POOL
from utils.device import get_device, get_device_name


class EmbeddingSpymaster(BaseSpymaster):
    """
    Spymaster that uses semantic embeddings to generate clues.

    Strategy:
    1. Identify team words, opponent words, neutral words, and assassin
    2. Compute embeddings for all board words and candidate clue words
    3. For each candidate clue, score it based on:
       - Similarity to team words (positive)
       - Dissimilarity to opponent/neutral/assassin words (negative)
    4. Select the best clue and determine how many words it targets
    """

    # Class-level cache for embedding model per (model_name, device)
    _model_cache: dict[tuple[str, str], tuple[object, bool]] = {}
    _fake_warning_printed: bool = False

    def __init__(self, team: str, params: Optional[SpymasterParams] = None):
        """
        Initialize embedding spymaster.

        Args:
            team: Team this spymaster plays for
            params: SpymasterParams with configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for accelerated inference. "
                "Install it with: pip install torch"
            )

        super().__init__(team, params)

        # Load embedding model (with caching)
        self.device = get_device()
        model_name = params.embedding_model_name if params and params.embedding_model_name else "all-MiniLM-L6-v2"
        cache_key = (model_name, str(self.device))
        if cache_key not in EmbeddingSpymaster._model_cache:
            EmbeddingSpymaster._model_cache[cache_key] = load_embedding_model(model_name, device=self.device)

        self.model, self.using_fake_embeddings = EmbeddingSpymaster._model_cache[cache_key]
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        if self.using_fake_embeddings and not EmbeddingSpymaster._fake_warning_printed:
            logger.info("EmbeddingSpymaster using hash embeddings (set CODENAMES_FAKE_EMBEDDINGS=0 for real models).")
            EmbeddingSpymaster._fake_warning_printed = True

        # Load or create clue word pool
        if params and params.clue_word_pool:
            self.clue_word_pool = params.clue_word_pool
        else:
            self.clue_word_pool = self._default_clue_pool()

        self.clue_embeddings = encode_vocab(
            self.model,
            self.clue_word_pool,
            self.device,
            cache_key=(model_name, str(self.device), "clue_pool")
        )
        # Create O(1) lookup dictionary for clue word indices
        self.clue_word_to_index = {word: idx for idx, word in enumerate(self.clue_word_pool)}

        self.board_vocab = WORD_POOL
        self.board_embeddings_cache: dict[tuple[str, ...], torch.Tensor] = {}
        self.max_cache_size = 1000  # Limit cache size to prevent memory leak

        # Random number generator for sampling
        self.rng = random.Random(self.params.seed)

    def reset(self):
        """
        Reset the spymaster state between games.

        Clears the board embeddings cache to prevent memory leak.
        """
        self.board_embeddings_cache.clear()

    def _default_clue_pool(self) -> list[str]:
        """Generate default pool of words for clues."""
        return [
            # Nature & Weather
            "SUNSHINE", "STORM", "OCEAN", "MOUNTAIN", "FOREST", "RIVER", "DESERT",
            "TROPICAL", "ARCTIC", "CLIMATE", "SEASON", "WILDERNESS",
            # Animals & Biology
            "MAMMAL", "REPTILE", "AQUATIC", "FLYING", "PREDATOR", "CREATURE",
            "WILDLIFE", "DOMESTIC", "BEAST", "INSECT",
            # Geography & Places
            "CONTINENT", "ISLAND", "CITY", "CAPITAL", "LANDMARK", "TERRITORY",
            "COASTAL", "INLAND", "REMOTE", "URBAN", "RURAL",
            # Abstract Concepts
            "POWER", "ENERGY", "SPEED", "STRENGTH", "KNOWLEDGE", "WISDOM",
            "FREEDOM", "JUSTICE", "TRUTH", "HARMONY", "CHAOS", "ORDER",
            # Colors & Appearance
            "BRIGHT", "DARK", "SHINY", "TRANSPARENT", "GOLDEN", "SILVER",
            "COLORFUL", "PALE", "VIVID", "METALLIC",
            # Size & Shape
            "HUGE", "TINY", "CIRCULAR", "ANGULAR", "CURVED", "STRAIGHT",
            "MASSIVE", "MINIATURE", "ENORMOUS", "COMPACT",
            # Time & History
            "ANCIENT", "MODERN", "MEDIEVAL", "FUTURE", "HISTORICAL", "CLASSICAL",
            "CONTEMPORARY", "VINTAGE", "ETERNAL", "TIMELESS",
            # Human & Society
            "ROYAL", "MILITARY", "MEDICAL", "LEGAL", "ACADEMIC", "ARTISTIC",
            "ATHLETIC", "MUSICAL", "SCIENTIFIC", "RELIGIOUS",
            # Actions & States
            "MOVING", "STATIC", "FLOWING", "FROZEN", "BURNING", "FLOATING",
            "SPINNING", "FLYING", "SWIMMING", "RUNNING",
            # Materials & Objects
            "WOODEN", "METAL", "FABRIC", "LIQUID", "SOLID", "ELECTRONIC",
            "MECHANICAL", "ORGANIC", "SYNTHETIC", "NATURAL",
        ]

    def get_clue(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        Generate clue using semantic embeddings with full PyTorch vectorization.

        Args:
            obs: Spymaster observation from environment

        Returns:
            Action dict with clue words and numbers
        """
        # Extract batch size and prepare outputs
        words_list = obs["words"]  # List of word lists, one per game
        if not words_list:
            return {
                "clue": ["PASS"],
                "clue_number": np.ones(1, dtype=np.int32)
            }

        revealed = torch.as_tensor(obs["revealed"], device=self.device, dtype=torch.bool)  # [B, N]
        colors = torch.as_tensor(obs["colors"], device=self.device, dtype=torch.int64)  # [B, N]
        batch_size = len(words_list)
        board_size = colors.shape[1]

        team_code = 0 if self.team == "red" else 1
        opponent_code = 1 if self.team == "red" else 0

        # Step 1: Vectorized word categorization across all games
        # Create masks for unrevealed words of each type: [B, N]
        unrevealed = ~revealed
        team_mask = unrevealed & (colors == team_code)
        opponent_mask = unrevealed & (colors == opponent_code)
        neutral_mask = unrevealed & (colors == 2)
        assassin_mask = unrevealed & (colors == 3)

        # Step 2: Sample candidate clues
        n_candidates = self.params.n_candidate_clues
        candidate_clues = self.rng.sample(
            self.clue_word_pool,
            min(n_candidates, len(self.clue_word_pool))
        )
        # O(1) lookup using dictionary instead of O(n) list.index()
        clue_indices = [self.clue_word_to_index[c] for c in candidate_clues]
        candidate_embeddings = self.clue_embeddings[clue_indices]

        # Step 3: Batch encode board words once per game (with caching)
        words_array = np.asarray(words_list, dtype=object)
        board_embeddings = []
        for words in words_array:
            key = tuple(words)
            if key not in self.board_embeddings_cache:
                # Clear cache if it's getting too large
                if len(self.board_embeddings_cache) >= self.max_cache_size:
                    self.board_embeddings_cache.clear()

                embeddings = self.model.encode(
                    words.tolist(),
                    convert_to_tensor=True,
                    device=str(self.device)
                )
                self.board_embeddings_cache[key] = embeddings
            board_embeddings.append(self.board_embeddings_cache[key])
        board_embeddings = torch.stack(board_embeddings, dim=0)

        # Step 4: Compute all pairwise cosine similarities [n_clues, B, N]
        clue_norms = torch.nn.functional.normalize(candidate_embeddings, dim=1)
        word_norms = torch.nn.functional.normalize(board_embeddings, dim=2)
        all_similarities = torch.einsum("cd,bnd->cbn", clue_norms, word_norms)

        # Step 5: Fully vectorized scoring for all games
        clue_words, clue_numbers = self._score_clues_vectorized(
            candidate_clues,
            all_similarities,
            team_mask,
            opponent_mask,
            neutral_mask,
            assassin_mask
        )

        return {
            "clue": clue_words,
            "clue_number": torch.tensor(clue_numbers, dtype=torch.int32)
        }

    def _score_clues_vectorized(
        self,
        candidate_clues: list[str],
        similarities: torch.Tensor,
        team_mask: torch.Tensor,
        opponent_mask: torch.Tensor,
        neutral_mask: torch.Tensor,
        assassin_mask: torch.Tensor
    ) -> tuple[list[str], list[int]]:
        """
        Fully vectorized clue scoring across all games.

        Args:
            candidate_clues: Candidate clue tokens
            similarities: [n_clues, B, N] cosine similarities
            team_mask/opponent_mask/...: [B, N] bool masks

        Returns:
            Tuple of (best clue words list, clue numbers list)
        """
        device = similarities.device
        n_clues, batch_size, board_size = similarities.shape

        if n_clues == 0 or not torch.any(team_mask):
            return ["PASS"] * batch_size, [1] * batch_size

        max_k = min(3, board_size)
        candidate_count = len(candidate_clues)
        if candidate_count == 0:
            return ["PASS"] * batch_size, [1] * batch_size

        def masked_max(mask: torch.Tensor) -> torch.Tensor:
            expanded_mask = mask.unsqueeze(0)  # [1, B, N]
            masked = similarities.masked_fill(~expanded_mask, float("-inf"))
            max_vals = masked.max(dim=2).values  # [n_clues, B]
            max_vals = torch.where(torch.isfinite(max_vals), max_vals, torch.zeros_like(max_vals))
            return max_vals.permute(1, 0)  # [B, n_clues]

        team_mask_exp = team_mask.unsqueeze(0)
        team_scores = similarities.masked_fill(~team_mask_exp, float("-inf"))
        topk_vals, _ = torch.topk(team_scores, k=max_k, dim=2)
        topk_vals = torch.where(torch.isfinite(topk_vals), topk_vals, torch.zeros_like(topk_vals))
        prefix = topk_vals.cumsum(dim=2)

        k_scale = torch.arange(1, max_k + 1, device=device, dtype=similarities.dtype).view(1, 1, max_k)
        avg_topk = prefix / k_scale  # [n_clues, B, max_k]

        avg_topk = avg_topk.permute(1, 0, 2)  # [B, n_clues, max_k]

        # Masks for valid k choices per game
        team_counts = team_mask.sum(dim=1)  # [B]
        k_thresholds = torch.arange(1, max_k + 1, device=device, dtype=team_counts.dtype)
        valid_k = team_counts.unsqueeze(1) >= k_thresholds  # [B, max_k]
        valid_mask = valid_k.unsqueeze(1)  # [B, 1, max_k]

        max_opponent = masked_max(opponent_mask) if torch.any(opponent_mask) else torch.zeros((batch_size, n_clues), device=device)
        max_neutral = masked_max(neutral_mask) if torch.any(neutral_mask) else torch.zeros((batch_size, n_clues), device=device)
        max_assassin = masked_max(assassin_mask) if torch.any(assassin_mask) else torch.zeros((batch_size, n_clues), device=device)

        penalties = (
            max_opponent.unsqueeze(-1) * self.params.opponent_penalty
            + max_neutral.unsqueeze(-1) * self.params.neutral_penalty
            + max_assassin.unsqueeze(-1) * self.params.assassin_penalty
        )

        scores = avg_topk * (k_scale * self.params.risk_tolerance)
        scores = scores - penalties
        scores = scores.masked_fill(~valid_mask, float("-inf"))

        scores_flat = scores.view(batch_size, -1)
        best_vals, best_indices = scores_flat.max(dim=1)

        best_clue_idx = best_indices // max_k
        best_k = (best_indices % max_k) + 1

        has_valid = valid_mask.view(batch_size, -1).any(dim=1) & torch.isfinite(best_vals)

        best_clue_idx_np = best_clue_idx.cpu().numpy()
        best_k_np = best_k.cpu().numpy()
        has_valid_np = has_valid.cpu().numpy()

        candidate_array = np.asarray(candidate_clues, dtype=object)
        safe_indices = np.clip(best_clue_idx_np, 0, len(candidate_clues) - 1)
        selected_words = candidate_array[safe_indices]

        final_words = np.where(has_valid_np, selected_words, "PASS").tolist()
        final_numbers = np.where(has_valid_np, best_k_np, 1).astype(int).tolist()

        return final_words, final_numbers
