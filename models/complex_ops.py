from __future__ import annotations

import torch



def real_imag_to_complex(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[-1] != 2:
        raise ValueError(f"Expected last dim=2 for real/imag tensor, got {tensor.shape}.")
    return torch.complex(tensor[..., 0], tensor[..., 1])



def complex_to_real_imag(tensor: torch.Tensor) -> torch.Tensor:
    return torch.stack([tensor.real, tensor.imag], dim=-1)



def complex_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(prediction - target) ** 2)



def complex_energy(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sum(tensor**2, dim=-1)
