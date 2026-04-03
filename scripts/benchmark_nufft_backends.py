from __future__ import annotations

import argparse
import json
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from subspaceINR4fMRI.data import load_fmri_data_bundle
from subspaceINR4fMRI.models.complex_ops import real_imag_to_complex
from subspaceINR4fMRI.nufft import create_nufft_backend
from subspaceINR4fMRI.nufft.nufft_utils import cuda_timed_call
from subspaceINR4fMRI.training.trainer_utils import prepare_nufft_config, prepare_run_directories, resolve_compute_device
from subspaceINR4fMRI.utils import get_peak_memory_stats, load_yaml_config, reset_peak_memory_stats



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark cufinufft and gpunufft across operator modes.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "benchmark_nufft.yaml"),
        help="Path to the benchmark YAML config.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="both",
        choices=["both", "cufinufft", "gpunufft"],
        help="Which backend(s) to benchmark.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["both", "preload", "reuse"],
        help="Which operator mode(s) to benchmark.",
    )
    return parser.parse_args()



def _select_backends(choice: str) -> List[str]:
    if choice == "both":
        return ["cufinufft", "gpunufft"]
    return [choice]



def _select_modes(choice: str) -> List[Tuple[str, Dict[str, bool]]]:
    if choice == "both":
        return [
            ("preload_per_frame", {"preload_per_frame_operators": True, "reuse_single_operator": False}),
            ("reuse_single_operator", {"preload_per_frame_operators": False, "reuse_single_operator": True}),
        ]
    if choice == "preload":
        return [("preload_per_frame", {"preload_per_frame_operators": True, "reuse_single_operator": False})]
    return [("reuse_single_operator", {"preload_per_frame_operators": False, "reuse_single_operator": True})]



def _make_image_batch(bundle, frame_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    base_image = bundle.background_init.to(device=device, dtype=torch.float32)
    base_complex = real_imag_to_complex(base_image)
    return base_complex.unsqueeze(0).repeat(frame_indices.numel(), 1, 1, 1).to(torch.complex64).contiguous()



def benchmark_backend_mode(config: Dict, bundle, backend_name: str, mode_name: str, mode_flags: Dict[str, bool]) -> Dict:
    benchmark_cfg = config["benchmark"]
    device, gpu_index = resolve_compute_device(config)
    frame_batch_size = int(benchmark_cfg.get("frame_batch_size", config.get("training", {}).get("frame_batch_size", 1)))
    repetitions = int(benchmark_cfg.get("repetitions", 10))
    warmup_repetitions = int(benchmark_cfg.get("warmup_repetitions", 3))
    benchmark_adjoint = bool(benchmark_cfg.get("benchmark_adjoint", True))

    frame_indices = torch.arange(frame_batch_size, dtype=torch.long)
    image_batch = _make_image_batch(bundle, frame_indices, device=device)
    kspace_batch = bundle.kspace[:frame_batch_size].to(device=device, dtype=torch.complex64)

    backend_config = prepare_nufft_config(config, gpu_index)
    backend_config["backend"] = backend_name
    backend_config.update(mode_flags)

    reset_peak_memory_stats(device)
    backend, setup_time = cuda_timed_call(
        device,
        create_nufft_backend,
        backend_config,
        bundle.traj,
        bundle.spatial_shape,
        bundle.num_coils,
        bundle.csm,
    )

    _, first_forward_time = cuda_timed_call(device, backend.forward_direct, image_batch, frame_indices)

    for warmup_index in range(warmup_repetitions):
        start = (warmup_index * frame_batch_size) % max(bundle.n_frames - frame_batch_size + 1, 1)
        current_indices = torch.arange(start, start + frame_batch_size, dtype=torch.long)
        current_images = _make_image_batch(bundle, current_indices, device=device)
        backend.forward_direct(current_images, current_indices)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    forward_times = []
    for repetition_index in range(repetitions):
        start = (repetition_index * frame_batch_size) % max(bundle.n_frames - frame_batch_size + 1, 1)
        current_indices = torch.arange(start, start + frame_batch_size, dtype=torch.long)
        current_images = _make_image_batch(bundle, current_indices, device=device)
        _, elapsed = cuda_timed_call(device, backend.forward_direct, current_images, current_indices)
        forward_times.append(elapsed)

    results = {
        "backend": backend_name,
        "operator_mode": mode_name,
        "setup_time_seconds": setup_time,
        "first_forward_seconds": first_forward_time,
        "forward_mean_seconds": statistics.mean(forward_times),
        "forward_std_seconds": statistics.pstdev(forward_times) if len(forward_times) > 1 else 0.0,
        "peak_memory": get_peak_memory_stats(device),
        "frame_batch_size": frame_batch_size,
        "repetitions": repetitions,
        "warmup_repetitions": warmup_repetitions,
    }

    if benchmark_adjoint:
        _, first_adjoint_time = cuda_timed_call(device, backend.adjoint_direct, kspace_batch, frame_indices)
        adjoint_times = []
        for repetition_index in range(repetitions):
            start = (repetition_index * frame_batch_size) % max(bundle.n_frames - frame_batch_size + 1, 1)
            current_indices = torch.arange(start, start + frame_batch_size, dtype=torch.long)
            current_kspace = bundle.kspace[start : start + frame_batch_size].to(device=device, dtype=torch.complex64)
            _, elapsed = cuda_timed_call(device, backend.adjoint_direct, current_kspace, current_indices)
            adjoint_times.append(elapsed)
        results.update(
            {
                "first_adjoint_seconds": first_adjoint_time,
                "adjoint_mean_seconds": statistics.mean(adjoint_times),
                "adjoint_std_seconds": statistics.pstdev(adjoint_times) if len(adjoint_times) > 1 else 0.0,
            }
        )

    return results



def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    run_dirs = prepare_run_directories(config, PROJECT_ROOT)
    bundle = load_fmri_data_bundle(config, PROJECT_ROOT)

    results = []
    for backend_name in _select_backends(args.backend):
        for mode_name, mode_flags in _select_modes(args.mode):
            result = benchmark_backend_mode(config, bundle, backend_name, mode_name, mode_flags)
            results.append(result)
            print(json.dumps(result, indent=2))

    save_path_cfg = config.get("benchmark", {}).get("save_path")
    if save_path_cfg:
        save_path = Path(save_path_cfg)
        if not save_path.is_absolute():
            save_path = (PROJECT_ROOT / save_path).resolve()
    else:
        save_path = run_dirs["benchmark_dir"] / "benchmark_results.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Saved benchmark results to {save_path}")


if __name__ == "__main__":
    main()
