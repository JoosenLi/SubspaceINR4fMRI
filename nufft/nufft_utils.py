from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch



def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)



def cuda_timed_call(device: torch.device, fn, *args, **kwargs):
    _synchronize_if_needed(device)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    _synchronize_if_needed(device)
    elapsed = time.perf_counter() - start
    return result, elapsed



def normalize_kspace_frame(kspace: torch.Tensor, num_coils: int) -> torch.Tensor:
    while kspace.ndim > 2:
        kspace = kspace.squeeze(0)
    if kspace.ndim == 1:
        kspace = kspace.unsqueeze(0)
    if kspace.shape[0] != num_coils and num_coils == 1 and kspace.shape[0] != 1:
        raise ValueError(f"Unexpected single-coil k-space shape: {tuple(kspace.shape)}")
    return kspace



def normalize_image_frame(image: torch.Tensor) -> torch.Tensor:
    while image.ndim > 3:
        image = image.squeeze(0)
    return image



def to_numpy_samples(samples: np.ndarray) -> np.ndarray:
    return np.asarray(samples, dtype=np.float32)


def frame_indices_to_list(frame_indices: torch.Tensor) -> List[int]:
    if frame_indices.device.type == "cpu":
        return [int(index) for index in frame_indices.tolist()]
    return [int(index) for index in frame_indices.detach().cpu().tolist()]


class _FramewiseNUFFTAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, image_batch: torch.Tensor, frame_indices: torch.Tensor, backend):
        ctx.backend = backend
        ctx.frame_indices = frame_indices_to_list(frame_indices)

        outputs = []
        get_operator = backend._get_operator
        num_coils = backend.num_coils
        with torch.no_grad():
            for batch_index, frame_index in enumerate(ctx.frame_indices):
                operator = get_operator(frame_index)
                kspace = operator.op(image_batch[batch_index].contiguous())
                outputs.append(normalize_kspace_frame(kspace, num_coils))
        return torch.stack(outputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_inputs = []
        backend = ctx.backend
        get_operator = backend._get_operator
        with torch.no_grad():
            for batch_index, frame_index in enumerate(ctx.frame_indices):
                operator = get_operator(frame_index)
                image_grad = operator.adj_op(grad_output[batch_index].contiguous())
                grad_inputs.append(normalize_image_frame(image_grad))
        return torch.stack(grad_inputs, dim=0), None, None


class FramewiseNUFFTBackendBase:
    backend_name = "base"

    def __init__(
        self,
        config: Dict[str, Any],
        traj: np.ndarray,
        image_shape: Sequence[int],
        num_coils: int,
        csm: Optional[np.ndarray] = None,
    ):
        self.config = config
        self.traj = np.asarray(traj, dtype=np.float32)
        self.image_shape = tuple(int(dim) for dim in image_shape)
        self.num_coils = int(num_coils)
        self.csm = csm

        self.preload_per_frame_operators = bool(config.get("preload_per_frame_operators", True))
        self.reuse_single_operator = bool(config.get("reuse_single_operator", False))
        if self.preload_per_frame_operators == self.reuse_single_operator:
            raise ValueError(
                "Exactly one NUFFT operator mode must be enabled: preload_per_frame_operators xor reuse_single_operator."
            )

        self.density = config.get("density", True)
        self.operator_kwargs = self._collect_operator_kwargs(config)
        gpu_device_id = int(self.operator_kwargs.get("gpu_device_id", 0))
        self.device = torch.device(f"cuda:{gpu_device_id}" if torch.cuda.is_available() else "cpu")
        self.setup_time_seconds = 0.0
        self.current_frame_index: Optional[int] = None
        self.operators: List[Any] = []
        self.operator = None
        self._initialize_operators()

    def _collect_operator_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = dict(config.get("backend_kwargs", {}))
        return kwargs

    def _build_operator(self, samples: np.ndarray):
        raise NotImplementedError

    def _initialize_operators(self) -> None:
        start = time.perf_counter()
        if self.preload_per_frame_operators:
            self.operators = [self._build_operator(samples) for samples in self.traj]
        else:
            self.operator = self._build_operator(self.traj[0])
            self.current_frame_index = 0
        self.setup_time_seconds = time.perf_counter() - start

    def _get_operator(self, frame_index: int):
        if self.preload_per_frame_operators:
            return self.operators[frame_index]
        if self.operator is None:
            raise RuntimeError("Single-operator NUFFT backend was not initialized.")
        if self.current_frame_index != frame_index:
            self.operator.update_samples(self.traj[frame_index])
            self.current_frame_index = frame_index
        return self.operator

    def forward_direct(self, image_batch: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        outputs = []
        get_operator = self._get_operator
        num_coils = self.num_coils
        for batch_index, frame_index in enumerate(frame_indices_to_list(frame_indices)):
            operator = get_operator(frame_index)
            kspace = operator.op(image_batch[batch_index].contiguous())
            outputs.append(normalize_kspace_frame(kspace, num_coils))
        return torch.stack(outputs, dim=0)

    def forward(self, image_batch: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        if image_batch.requires_grad:
            return _FramewiseNUFFTAutograd.apply(image_batch, frame_indices, self)
        return self.forward_direct(image_batch, frame_indices)

    def adjoint_direct(self, kspace_batch: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        outputs = []
        get_operator = self._get_operator
        for batch_index, frame_index in enumerate(frame_indices_to_list(frame_indices)):
            operator = get_operator(frame_index)
            image = operator.adj_op(kspace_batch[batch_index].contiguous())
            outputs.append(normalize_image_frame(image))
        return torch.stack(outputs, dim=0)

    def adjoint(self, kspace_batch: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
        return self.adjoint_direct(kspace_batch, frame_indices)

    def describe(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "preload_per_frame_operators": self.preload_per_frame_operators,
            "reuse_single_operator": self.reuse_single_operator,
            "setup_time_seconds": self.setup_time_seconds,
        }
