from __future__ import annotations

from typing import Dict

import tinycudann as tcnn
import torch
import torch.nn as nn


class TCNNHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: Dict,
        output_scale: float = 1.0,
    ):
        super().__init__()
        network_type = str(config.get("network_type", "FullyFusedMLP"))
        self.network = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=output_dim,
            network_config={
                "otype": network_type,
                "activation": str(config.get("activation", "ReLU")),
                "output_activation": str(config.get("output_activation", "None")),
                "n_neurons": int(config["hidden_dim"]),
                "n_hidden_layers": int(config["num_hidden_layers"]),
            },
        )
        self.output_scale = float(output_scale)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).float() * self.output_scale
