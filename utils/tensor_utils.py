from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import torch


Tensor = torch.Tensor


def safe_item(value: Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


@torch.no_grad()
def compute_grad_norm(parameters: Iterable[torch.nn.Parameter], norm_type: float = 2.0) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        param_norm = parameter.grad.data.norm(norm_type)
        total += float(param_norm.item() ** norm_type)
    if total == 0.0:
        return 0.0
    return total ** (1.0 / norm_type)


def assert_finite(name: str, tensor: Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"{name} contains NaN or Inf values.")


@torch.no_grad()
def detach_cpu(tensor: Tensor) -> Tensor:
    return tensor.detach().cpu()


def chunked_forward(
    inputs: Tensor,
    fn: Callable[[Tensor], Tensor],
    chunk_size: Optional[int],
) -> Tensor:
    if chunk_size is None or chunk_size <= 0 or inputs.shape[0] <= chunk_size:
        return fn(inputs)
    outputs = []
    for start in range(0, inputs.shape[0], chunk_size):
        outputs.append(fn(inputs[start : start + chunk_size]))
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def tensor_stats(tensor: Tensor) -> Dict[str, float]:
    tensor = tensor.detach().float()
    return {
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }
