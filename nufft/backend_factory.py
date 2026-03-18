from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .cufinufft_backend import CuFINUFFTBackend
from .gpunufft_backend import GPUNUFFTBackend



def create_nufft_backend(
    config: Dict,
    traj: np.ndarray,
    image_shape: Sequence[int],
    num_coils: int,
    csm: Optional[np.ndarray] = None,
):
    if csm is None and int(num_coils) > 1:
        raise ValueError(
            "Multi-coil reconstruction requires a coil sensitivity map. "
            "Single-coil runs may set csm=None."
        )
    backend_name = str(config.get("backend", "gpunufft")).lower()
    if backend_name == "cufinufft":
        return CuFINUFFTBackend(config, traj, image_shape, num_coils=num_coils, csm=csm)
    if backend_name == "gpunufft":
        return GPUNUFFTBackend(config, traj, image_shape, num_coils=num_coils, csm=csm)
    raise ValueError(f"Unsupported NUFFT backend: {config.get('backend')}")
