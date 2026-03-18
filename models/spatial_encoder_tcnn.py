from __future__ import annotations

from typing import Dict

import tinycudann as tcnn
import torch
import torch.nn as nn


class TCNNHashGridEncoder(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.encoding = tcnn.Encoding(
            n_input_dims=int(config.get("input_dim", 3)),
            encoding_config={
                "otype": "HashGrid",
                "n_levels": int(config["num_levels"]),
                "n_features_per_level": int(config["level_dim"]),
                "log2_hashmap_size": int(config.get("log2_hashmap_size", 19)),
                "base_resolution": int(config["base_resolution"]),
                "per_level_scale": float(config["per_level_scale"]),
                "interpolation": str(config.get("interpolation", "Linear")),
            },
        )
        self.output_dim = int(self.encoding.n_output_dims)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.encoding(coords)
