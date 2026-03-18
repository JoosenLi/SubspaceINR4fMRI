from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.tensor_utils import chunked_forward
from .heads import TCNNHead
from .spatial_encoder_tcnn import TCNNHashGridEncoder
from .temporal_basis import TemporalBasisNetwork


@dataclass
class SpatialComponents:
    residual: Optional[torch.Tensor]
    coeff: torch.Tensor
    tau: Optional[torch.Tensor]


class SubspaceINR4fMRI(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.rank = int(config["model"]["rank"])
        self.use_residual = bool(config["model"].get("use_residual", False))
        self.use_delay = bool(config["model"]["use_delay"])
        self.cache_spatial_outputs = bool(config["model"].get("cache_spatial_outputs", True))
        self.chunked_spatial_eval = bool(config["model"].get("chunked_spatial_eval", False))
        self.spatial_chunk_size = int(config["model"].get("spatial_chunk_size", 0))

        if self.use_residual and int(config["residual_head"].get("output_dim", 2)) != 2:
            raise ValueError("residual_head.output_dim must be 2 for complex real/imag output.")
        if int(config["coefficient_head"].get("output_dim", self.rank)) != self.rank:
            raise ValueError("coefficient_head.output_dim must match model.rank.")
        if int(config["temporal_branch"].get("output_dim", 2 * self.rank)) != 2 * self.rank:
            raise ValueError("temporal_branch.output_dim must equal 2 * model.rank.")
        if self.use_delay and int(config["delay_head"].get("output_dim", 1)) != 1:
            raise ValueError("delay_head.output_dim must be 1 when delay is enabled.")

        self.total_duration_seconds = float(config["time"]["total_duration_seconds"])
        self.tau_norm_max = float(config["time"]["tau_max_seconds"]) / max(self.total_duration_seconds, 1.0e-6)

        self.spatial_encoder = TCNNHashGridEncoder(config["spatial_encoder"])
        feature_dim = self.spatial_encoder.output_dim

        self.residual_head = None
        if self.use_residual:
            residual_scale = float(config["residual_head"].get("output_init_scale", 1.0e-4))
            self.residual_head = TCNNHead(
                input_dim=feature_dim,
                output_dim=2,
                config=config["residual_head"],
                output_scale=residual_scale,
            )
        self.coefficient_head = TCNNHead(
            input_dim=feature_dim,
            output_dim=self.rank,
            config=config["coefficient_head"],
            output_scale=1.0,
        )
        self.delay_head = None
        if self.use_delay:
            delay_scale = float(config["delay_head"].get("output_init_scale", 1.0e-4))
            self.delay_head = TCNNHead(
                input_dim=feature_dim,
                output_dim=1,
                config=config["delay_head"],
                output_scale=delay_scale,
            )

        self.temporal_basis = TemporalBasisNetwork(config["temporal_branch"], rank=self.rank)

    def _chunk_size(self, override: Optional[int]) -> Optional[int]:
        if override is not None:
            return override
        if self.chunked_spatial_eval:
            return self.spatial_chunk_size
        return None

    def encode_spatial(self, coords: torch.Tensor, chunk_size: Optional[int] = None) -> torch.Tensor:
        return chunked_forward(coords, self.spatial_encoder, self._chunk_size(chunk_size))

    def evaluate_spatial_components(
        self,
        coords: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> SpatialComponents:
        features = self.encode_spatial(coords, chunk_size=chunk_size)
        residual = None
        if self.use_residual and self.residual_head is not None:
            residual = chunked_forward(features, self.residual_head, self._chunk_size(chunk_size))
        coeff = chunked_forward(features, self.coefficient_head, self._chunk_size(chunk_size))
        tau = None
        if self.use_delay and self.delay_head is not None:
            raw_tau = chunked_forward(features, self.delay_head, self._chunk_size(chunk_size))
            tau = self.tau_norm_max * torch.tanh(raw_tau)
        return SpatialComponents(residual=residual, coeff=coeff, tau=tau)

    def evaluate_temporal_basis(
        self,
        t_norm: torch.Tensor,
        need_derivative: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return self.temporal_basis(t_norm, need_derivative=need_derivative)

    def synthesize_batch(
        self,
        init_image_flat: torch.Tensor,
        spatial_components: SpatialComponents,
        basis: torch.Tensor,
        spatial_shape: Tuple[int, int, int],
        basis_derivative: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dynamic = torch.einsum("vm,bmc->bvc", spatial_components.coeff, basis)
        if self.use_delay:
            if spatial_components.tau is None or basis_derivative is None:
                raise ValueError("Delay synthesis requested but tau or basis derivative is missing.")
            delayed_dynamic = torch.einsum("vm,bmc->bvc", spatial_components.coeff, basis_derivative)
            dynamic = dynamic - spatial_components.tau.unsqueeze(0) * delayed_dynamic

        static = init_image_flat.unsqueeze(0)
        if spatial_components.residual is not None:
            static = static + spatial_components.residual.unsqueeze(0)
        prediction = static + dynamic
        nx, ny, nz = spatial_shape
        return prediction.view(basis.shape[0], nx, ny, nz, 2)
