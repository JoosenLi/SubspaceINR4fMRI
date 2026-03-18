from __future__ import annotations

from typing import Dict

import torch


BYTES_IN_GB = float(1024**3)


def reset_peak_memory_stats(device: torch.device) -> None:
    if device.type != "cuda":
        return
    torch.cuda.reset_peak_memory_stats(device)


@torch.no_grad()
def get_peak_memory_stats(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {
            "peak_allocated_bytes": 0.0,
            "peak_reserved_bytes": 0.0,
            "peak_allocated_gb": 0.0,
            "peak_reserved_gb": 0.0,
        }
    allocated = float(torch.cuda.max_memory_allocated(device))
    reserved = float(torch.cuda.max_memory_reserved(device))
    return {
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "peak_allocated_gb": allocated / BYTES_IN_GB,
        "peak_reserved_gb": reserved / BYTES_IN_GB,
    }
