from __future__ import annotations

from typing import Dict, Optional, Tuple

import math

import torch
import torch.nn as nn


class FourierFeatureEncoding(nn.Module):
    def __init__(self, num_frequencies: int):
        super().__init__()
        if num_frequencies > 20:
            raise ValueError(
                "temporal_branch.fourier_num_frequencies is too large for normalized time. "
                f"Got {num_frequencies}, but values above 20 make dg/dt_norm explode because "
                "the encoding uses powers of two frequencies. For this project, use 16 unless "
                "you intentionally redesign the temporal regularization."
            )
        frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.output_dim = 1 + 2 * num_frequencies

    def forward(self, t_norm: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.forward_with_derivative(t_norm)
        return encoded

    def forward_with_derivative(self, t_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # t_norm is the normalized temporal coordinate in [0, 1].
        # The derivative returned here is d(encoding) / dt_norm.
        phases = 2.0 * math.pi * t_norm * self.frequencies.view(1, -1)
        sin_phase = torch.sin(phases)
        cos_phase = torch.cos(phases)
        encoded = torch.cat([t_norm, sin_phase, cos_phase], dim=-1)

        phase_scale = 2.0 * math.pi * self.frequencies.view(1, -1)
        derivative = torch.cat(
            [
                torch.ones_like(t_norm),
                cos_phase * phase_scale,
                -sin_phase * phase_scale,
            ],
            dim=-1,
        )
        return encoded, derivative


class SineLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, w0: float = 30.0, is_first: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.w0 = w0
        self.is_first = is_first
        self.linear = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_dim
            else:
                bound = math.sqrt(6.0 / self.in_dim) / self.w0
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * self.linear(x))

    def forward_with_derivative(self, x: torch.Tensor, dx_dt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        linear_out = self.linear(x)
        scaled = self.w0 * linear_out
        linear_derivative = torch.matmul(dx_dt, self.linear.weight.t())
        output = torch.sin(scaled)
        output_derivative = torch.cos(scaled) * (self.w0 * linear_derivative)
        return output, output_derivative


class TemporalBasisNetwork(nn.Module):
    def __init__(self, config: Dict, rank: int):
        super().__init__()
        num_freq = int(config.get("fourier_num_frequencies", 16))
        hidden_dim = int(config.get("hidden_dim", 64))
        num_hidden_layers = int(config.get("num_hidden_layers", 2))
        self.rank = int(rank)
        self.encoding = FourierFeatureEncoding(num_freq)

        layers = [SineLayer(self.encoding.output_dim, hidden_dim, is_first=True)]
        for _ in range(max(num_hidden_layers - 1, 0)):
            layers.append(SineLayer(hidden_dim, hidden_dim))
        self.hidden = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, 2 * self.rank)
        self.reset_output_parameters(scale=float(config.get("output_init_scale", 1.0e-4)))

    def reset_output_parameters(self, scale: float) -> None:
        with torch.no_grad():
            self.output.weight.uniform_(-scale, scale)
            self.output.bias.zero_()

    def _forward_analytic(
        self,
        t_norm: torch.Tensor,
        need_derivative: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        encoded, derivative = self.encoding.forward_with_derivative(t_norm)
        hidden = encoded
        hidden_derivative = derivative
        for layer in self.hidden:
            hidden, hidden_derivative = layer.forward_with_derivative(hidden, hidden_derivative)

        basis_flat = self.output(hidden)
        basis = basis_flat.view(t_norm.shape[0], self.rank, 2)
        if not need_derivative:
            return basis, None

        derivative_flat = torch.matmul(hidden_derivative, self.output.weight.t())
        derivative_basis = derivative_flat.view(t_norm.shape[0], self.rank, 2)
        return basis, derivative_basis

    def forward_with_autograd_derivative(self, t_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if t_norm.ndim == 1:
            t_norm = t_norm.unsqueeze(-1)
        if t_norm.shape[-1] != 1:
            raise ValueError(f"Expected t_norm shape [B, 1], got {tuple(t_norm.shape)}")

        t_input = t_norm.clone().detach().requires_grad_(True)
        encoded = self.encoding(t_input)
        hidden = encoded
        for layer in self.hidden:
            hidden = layer(hidden)
        basis_flat = self.output(hidden)
        basis = basis_flat.view(t_input.shape[0], self.rank, 2)

        grad_columns = []
        for feature_index in range(basis_flat.shape[-1]):
            grad_column = torch.autograd.grad(
                basis_flat[:, feature_index].sum(),
                t_input,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grad_columns.append(grad_column)
        derivative_flat = torch.cat(grad_columns, dim=-1)
        derivative = derivative_flat.view(t_input.shape[0], self.rank, 2)
        return basis, derivative

    def forward(
        self,
        t_norm: torch.Tensor,
        need_derivative: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if t_norm.ndim == 1:
            t_norm = t_norm.unsqueeze(-1)
        if t_norm.shape[-1] != 1:
            raise ValueError(f"Expected t_norm shape [B, 1], got {tuple(t_norm.shape)}")
        return self._forward_analytic(t_norm, need_derivative=need_derivative)
