"""Device selection: MPS if available, CPU fallback. Never crashes on an
unsupported MPS op — callers should catch RuntimeError around MPS-specific
forward passes if a new op turns out to be unsupported, and fall back
explicitly (see select_device's docstring)."""
from __future__ import annotations

import torch


def select_device(prefer: str = "mps", fallback: str = "cpu") -> torch.device:
    """Return the best available device given a preference, without ever
    silently pretending acceleration happened. Prints what it actually picked
    and why."""
    if prefer == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            print(f"[device] MPS requested but not available on this machine — falling back to {fallback}.")
            return torch.device(fallback)
    return torch.device(prefer)


def safe_to_device(module_or_tensor, device: torch.device):
    """Move a module/tensor to `device`, falling back to CPU with a printed
    warning if an operation is not implemented on MPS (rather than crashing).
    This does NOT silently claim GPU acceleration occurred if it fell back.
    """
    try:
        return module_or_tensor.to(device)
    except (RuntimeError, NotImplementedError) as e:
        if device.type == "mps":
            print(f"[device] Operation unsupported on MPS ({e}); falling back to CPU for this object.")
            return module_or_tensor.to("cpu")
        raise
