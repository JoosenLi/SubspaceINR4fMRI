from __future__ import annotations

from typing import Any, Dict

from mrinufft import get_operator

from .nufft_utils import FramewiseNUFFTBackendBase, to_numpy_samples


class GPUNUFFTBackend(FramewiseNUFFTBackendBase):
    backend_name = "gpunufft"

    def __init__(self, config: Dict[str, Any], traj, image_shape, num_coils: int, csm=None):
        self.operator_cls = get_operator("gpunufft")
        super().__init__(config, traj, image_shape, num_coils, csm=csm)

    def _collect_operator_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = dict(config.get("backend_kwargs", {}))
        kwargs.setdefault("use_gpu_direct", True)
        return kwargs

    def _build_operator(self, samples):
        return self.operator_cls(
            samples=to_numpy_samples(samples),
            shape=self.image_shape,
            density=self.density,
            n_coils=self.num_coils,
            smaps=self.csm,
            squeeze_dims=True,
            **self.operator_kwargs,
        )
