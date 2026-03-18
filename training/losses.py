from __future__ import annotations

from typing import Tuple

import torch

from ..models.complex_ops import complex_energy



def residual_energy_loss(residual: torch.Tensor) -> torch.Tensor:
    return complex_energy(residual).mean()



def tau_smoothness_loss(tau: torch.Tensor, spatial_shape: Tuple[int, int, int]) -> torch.Tensor:
    nx, ny, nz = spatial_shape
    tau_field = tau.view(nx, ny, nz)
    dx = tau_field[1:, :, :] - tau_field[:-1, :, :]
    dy = tau_field[:, 1:, :] - tau_field[:, :-1, :]
    dz = tau_field[:, :, 1:] - tau_field[:, :, :-1]
    return (dx.square().mean() + dy.square().mean() + dz.square().mean()) / 3.0



def temporal_basis_smoothness_loss(derivative: torch.Tensor) -> torch.Tensor:
    return derivative.square().mean()
