"""
Common utilities for batch environments.

Shared helper functions used across VectorBatchEnv and WordBatchEnv
to reduce code duplication.
"""

from __future__ import annotations

from typing import Any, Optional, Union
import torch


def process_spymaster_actions(
    batch_size: int,
    device: torch.device,
    actions_dict: dict[str, Any],
    active_masks: dict[str, torch.Tensor],
    reality_layer: Optional[Any] = None,
    clue_type: str = "vector",
    embedding_dim: int = 384,
    prev_clue_indices: Optional[torch.Tensor] = None,
    prev_clue_numbers: Optional[torch.Tensor] = None,
    prev_clue_outputs: Optional[Union[torch.Tensor, list[str]]] = None
) -> tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, list[str]]]:
    """
    Process spymaster actions and select active clues.

    This helper centralizes the logic for:
    1. Extracting actions from red/blue spymasters
    2. Selecting based on which spymaster is active
    3. Optionally applying reality layer
    4. Returning clue indices, numbers, and final outputs
    5. Preserving previous clues for inactive/finished games

    Args:
        batch_size: Number of parallel games
        device: PyTorch device for tensors
        actions_dict: Dictionary with "red_spy" and "blue_spy" actions
        active_masks: Dictionary with masks for each agent
        reality_layer: Optional reality layer for snapping/mapping clues
        clue_type: Either "vector" or "word" to determine clue format
        embedding_dim: Embedding dimension for vector clues (required for vector mode)
        prev_clue_indices: Optional [B] tensor of previous clue indices (preserved when both spymasters inactive)
        prev_clue_numbers: Optional [B] tensor of previous clue numbers (preserved when both spymasters inactive)
        prev_clue_outputs: Optional previous clue outputs (preserved when both spymasters inactive)

    Returns:
        Tuple of:
            - clue_indices: [B] tensor of clue indices (-1 if no reality layer)
            - clue_numbers: [B] tensor of clue counts
            - clue_outputs: Either [B, D] tensor of vectors or list of word strings
    """
    # Extract actions from both spymasters
    red_spy_actions = actions_dict.get("red_spy", {})
    blue_spy_actions = actions_dict.get("blue_spy", {})

    # Get active masks
    active_red = active_masks["red_spy"]
    active_blue = active_masks["blue_spy"]

    if clue_type == "vector":
        # Vector-based processing
        # Get clue vectors and numbers, normalizing inputs
        red_clue_vecs = red_spy_actions.get("clue_vec", None)
        if red_clue_vecs is None:
            red_clue_vecs = torch.zeros((batch_size, embedding_dim), device=device)
        else:
            red_clue_vecs = torch.as_tensor(red_clue_vecs, device=device, dtype=torch.float32)
            # Ensure shape is correct
            if red_clue_vecs.ndim == 1:
                red_clue_vecs = red_clue_vecs.unsqueeze(0)
            if red_clue_vecs.shape[0] != batch_size or red_clue_vecs.shape[1] != embedding_dim:
                # Raise error instead of silently zeroing
                raise ValueError(
                    f"Red spymaster clue_vec has invalid shape {red_clue_vecs.shape}. "
                    f"Expected ({batch_size}, {embedding_dim})"
                )

        blue_clue_vecs = blue_spy_actions.get("clue_vec", None)
        if blue_clue_vecs is None:
            blue_clue_vecs = torch.zeros((batch_size, embedding_dim), device=device)
        else:
            blue_clue_vecs = torch.as_tensor(blue_clue_vecs, device=device, dtype=torch.float32)
            # Ensure shape is correct
            if blue_clue_vecs.ndim == 1:
                blue_clue_vecs = blue_clue_vecs.unsqueeze(0)
            if blue_clue_vecs.shape[0] != batch_size or blue_clue_vecs.shape[1] != embedding_dim:
                # Raise error instead of silently zeroing
                raise ValueError(
                    f"Blue spymaster clue_vec has invalid shape {blue_clue_vecs.shape}. "
                    f"Expected ({batch_size}, {embedding_dim})"
                )

        # Select based on active mask
        clue_vecs = torch.where(
            active_red.unsqueeze(1),
            red_clue_vecs,
            blue_clue_vecs
        )

        # Apply reality layer if enabled
        if reality_layer is not None:
            # Extract clue numbers for reality layer, normalizing inputs
            red_nums = red_spy_actions.get("clue_number", None)
            if red_nums is None:
                red_nums = torch.zeros(batch_size, dtype=torch.int32, device=device)
            else:
                red_nums = torch.as_tensor(red_nums, device=device, dtype=torch.int32)

            blue_nums = blue_spy_actions.get("clue_number", None)
            if blue_nums is None:
                blue_nums = torch.zeros(batch_size, dtype=torch.int32, device=device)
            else:
                blue_nums = torch.as_tensor(blue_nums, device=device, dtype=torch.int32)

            clue_numbers = torch.where(active_red, red_nums, blue_nums)

            clue_indices, final_vecs, clue_numbers = reality_layer.apply(
                clue_vecs, clue_numbers
            )
            clue_outputs = final_vecs
        else:
            clue_indices = torch.full(
                (batch_size,), -1, dtype=torch.int32, device=device
            )
            clue_outputs = clue_vecs

            # Get clue numbers separately, normalizing inputs
            red_nums = red_spy_actions.get("clue_number", None)
            if red_nums is None:
                red_nums = torch.zeros(batch_size, dtype=torch.int32, device=device)
            else:
                red_nums = torch.as_tensor(red_nums, device=device, dtype=torch.int32)

            blue_nums = blue_spy_actions.get("clue_number", None)
            if blue_nums is None:
                blue_nums = torch.zeros(batch_size, dtype=torch.int32, device=device)
            else:
                blue_nums = torch.as_tensor(blue_nums, device=device, dtype=torch.int32)

            clue_numbers = torch.where(active_red, red_nums, blue_nums)

    elif clue_type == "word":
        # Word-based processing
        def _pad_numbers(values: Any) -> torch.Tensor:
            if isinstance(values, torch.Tensor):
                arr = values.to(torch.int32).reshape(-1).to(device)
            else:
                arr = torch.tensor(
                    values, dtype=torch.int32, device=device
                ).reshape(-1)

            if arr.numel() < batch_size:
                pad_width = batch_size - arr.numel()
                arr = torch.nn.functional.pad(arr, (0, pad_width), value=0)
            elif arr.numel() > batch_size:
                arr = arr[:batch_size]
            return arr

        red_clue_numbers = _pad_numbers(
            red_spy_actions.get(
                "clue_number",
                torch.zeros(batch_size, dtype=torch.int32, device=device)
            )
        )
        blue_clue_numbers = _pad_numbers(
            blue_spy_actions.get(
                "clue_number",
                torch.zeros(batch_size, dtype=torch.int32, device=device)
            )
        )

        clue_numbers = torch.where(active_red, red_clue_numbers, blue_clue_numbers)
        clue_indices = torch.full(
            (batch_size,), -1, dtype=torch.int32, device=device
        )
        clue_words = [""] * batch_size

        # Get token lookup from reality layer if available
        token_lookup = None
        if reality_layer is not None and hasattr(reality_layer, "clue_vocab"):
            if hasattr(reality_layer.clue_vocab, "_token_to_index"):
                token_lookup = reality_layer.clue_vocab._token_to_index

        # Extract word lists
        red_clue_words = red_spy_actions.get("clue", [])
        blue_clue_words = blue_spy_actions.get("clue", [])

        if red_clue_words or blue_clue_words:
            is_active = active_red | active_blue
            active_indices = torch.nonzero(is_active, as_tuple=False).squeeze(-1)

            for idx in active_indices:
                idx_item = idx.item()
                use_red = active_red[idx].item()
                words_list = red_clue_words if use_red else blue_clue_words

                if isinstance(words_list, list) and len(words_list) > idx_item:
                    word = words_list[idx_item]
                    clue_words[idx_item] = word

                    # Map to index if token lookup available
                    if token_lookup is not None:
                        mapped = token_lookup.get(word)
                        if mapped is not None:
                            clue_indices[idx_item] = mapped

        clue_outputs = clue_words

    else:
        raise ValueError(f"Unknown clue_type: {clue_type}")

    # Preserve previous clues for games where neither spymaster is active
    # This prevents overwriting historical clues shown to guessers in finished games
    if prev_clue_indices is not None and prev_clue_numbers is not None and prev_clue_outputs is not None:
        inactive_mask = ~(active_red | active_blue)

        # Preserve indices
        clue_indices = torch.where(inactive_mask, prev_clue_indices, clue_indices)

        # Preserve numbers
        clue_numbers = torch.where(inactive_mask, prev_clue_numbers, clue_numbers)

        # Preserve outputs (vector or word)
        if clue_type == "vector":
            # For vectors, use torch.where with broadcasting
            clue_outputs = torch.where(
                inactive_mask.unsqueeze(1),
                prev_clue_outputs,
                clue_outputs
            )
        else:
            # For word lists, preserve individual words
            clue_outputs = [
                prev_clue_outputs[i] if inactive_mask[i].item() else clue_outputs[i]
                for i in range(batch_size)
            ]

    return clue_indices, clue_numbers, clue_outputs
