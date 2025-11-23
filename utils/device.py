"""
Device management utilities for PyTorch.

Provides centralized GPU detection and device management for CUDA (NVIDIA),
MPS (Apple Silicon), and CPU fallback.
"""

import torch

# Detect best available device
if torch.cuda.is_available():
    DEFAULT_DEVICE = torch.device("cuda")
    DEVICE_NAME = "CUDA"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEFAULT_DEVICE = torch.device("mps")
    DEVICE_NAME = "MPS (Apple Silicon)"
else:
    DEFAULT_DEVICE = torch.device("cpu")
    DEVICE_NAME = "CPU"


def get_device() -> torch.device:
    """
    Get the default compute device.

    Returns:
        torch.device: Best available device (CUDA > MPS > CPU)
    """
    return DEFAULT_DEVICE


def get_device_name() -> str:
    """
    Get human-readable device name.

    Returns:
        str: Device name (e.g., "CUDA", "MPS (Apple Silicon)", "CPU")
    """
    return DEVICE_NAME


def to_numpy(tensor: torch.Tensor) -> "np.ndarray":
    """
    Convert PyTorch tensor to NumPy array.

    Handles device transfers (GPU -> CPU) automatically.

    Args:
        tensor: PyTorch tensor

    Returns:
        NumPy array
    """
    import numpy as np
    return tensor.detach().cpu().numpy()


def to_torch(array: "np.ndarray", device: torch.device = None) -> torch.Tensor:
    """
    Convert NumPy array to PyTorch tensor.

    Args:
        array: NumPy array
        device: Target device (defaults to DEFAULT_DEVICE)

    Returns:
        PyTorch tensor on specified device
    """
    import numpy as np
    if device is None:
        device = DEFAULT_DEVICE
    return torch.from_numpy(array).to(device)


# Device info available via get_device_name() - call explicitly if needed
