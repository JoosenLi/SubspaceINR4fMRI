from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from subspaceINR4fMRI.data import create_normalized_spatial_grid, load_fmri_data_bundle
from subspaceINR4fMRI.models import SpatialComponents, SubspaceINR4fMRI, complex_mse, real_imag_to_complex
from subspaceINR4fMRI.nufft import create_nufft_backend
from subspaceINR4fMRI.training.trainer_utils import prepare_nufft_config, resolve_compute_device, set_random_seed
from subspaceINR4fMRI.utils import apply_dotlist_overrides, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test whether I_init absorbs the primary reconstruction energy.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "subspaceINR4fMRI.yaml"),
        help="Path to the YAML config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override such as cuda:0.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "background_init_absorption_test.json"),
        help="Path to save the JSON summary.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional dotted config overrides, e.g. nufft.backend=gpunufft.",
    )
    parser.add_argument(
        "--background-scale",
        type=float,
        default=1.0,
        help="Optional multiplier applied to I_init before the test.",
    )
    return parser.parse_args()


def _spatial_components_to_float32(spatial_components: SpatialComponents) -> SpatialComponents:
    residual = None if spatial_components.residual is None else spatial_components.residual.float()
    tau = None if spatial_components.tau is None else spatial_components.tau.float()
    return SpatialComponents(
        residual=residual,
        coeff=spatial_components.coeff.float(),
        tau=tau,
    )


def _compare_background_to_global_cg(config: Dict, project_root: Path) -> Dict[str, float] | None:
    data_cfg = config.get("data", {})
    bg_path = project_root / str(data_cfg.get("background_init_path", ""))
    case_dir = bg_path.parent if bg_path.name else None
    if case_dir is None:
        return None
    global_cg_path = case_dir / "Global_CG.npy"
    if not global_cg_path.exists():
        return None

    background = np.load(bg_path)
    background_complex = background[..., 0] + 1j * background[..., 1]
    global_cg = np.load(global_cg_path)
    if not np.iscomplexobj(global_cg):
        global_cg = global_cg[..., 0] + 1j * global_cg[..., 1]
    global_cg_mean = global_cg.mean(axis=0)

    alpha = np.vdot(background_complex.reshape(-1), global_cg_mean.reshape(-1)) / np.vdot(
        background_complex.reshape(-1),
        background_complex.reshape(-1),
    )
    magnitude_corr = float(
        np.corrcoef(
            np.abs(background_complex).reshape(-1),
            np.abs(global_cg_mean).reshape(-1),
        )[0, 1]
    )
    relative_residual = float(
        np.linalg.norm(alpha * background_complex - global_cg_mean) / np.linalg.norm(global_cg_mean)
    )
    mean_magnitude_ratio = float(
        np.abs(global_cg_mean).mean() / max(np.abs(background_complex).mean(), 1.0e-12)
    )
    return {
        "global_cg_path": str(global_cg_path),
        "magnitude_correlation": magnitude_corr,
        "mean_magnitude_ratio_global_cg_over_background": mean_magnitude_ratio,
        "best_fit_complex_scale_real": float(np.real(alpha)),
        "best_fit_complex_scale_imag": float(np.imag(alpha)),
        "best_fit_complex_scale_abs": float(np.abs(alpha)),
        "best_fit_complex_scale_phase_deg": float(np.angle(alpha, deg=True)),
        "relative_residual_after_best_scalar_fit": relative_residual,
    }


@torch.no_grad()
def _average_initial_data_loss(
    model: SubspaceINR4fMRI,
    bundle,
    nufft_backend,
    init_image_flat: torch.Tensor,
    spatial_components: SpatialComponents,
    frame_batch_size: int,
) -> float:
    total_loss = 0.0
    num_batches = 0
    full_time = bundle.time_coords.to(init_image_flat.device, dtype=torch.float32)
    for start in range(0, bundle.n_frames, frame_batch_size):
        frame_indices_cpu = torch.arange(start, min(start + frame_batch_size, bundle.n_frames), dtype=torch.long)
        frame_indices_device = frame_indices_cpu.to(device=init_image_flat.device)
        t_batch = full_time.index_select(0, frame_indices_device)
        basis, basis_derivative = model.evaluate_temporal_basis(t_batch, need_derivative=model.use_delay)
        prediction_ri = model.synthesize_batch(
            init_image_flat,
            spatial_components,
            basis.float(),
            bundle.spatial_shape,
            basis_derivative=None if basis_derivative is None else basis_derivative.float(),
        )
        prediction_complex = real_imag_to_complex(prediction_ri).to(torch.complex64).contiguous()
        kspace_prediction = nufft_backend.forward_direct(prediction_complex, frame_indices_cpu)
        kspace_target = bundle.kspace.index_select(0, frame_indices_cpu.to(bundle.kspace.device)).to(
            device=init_image_flat.device,
            dtype=torch.complex64,
            non_blocking=True,
        )
        total_loss += float(complex_mse(kspace_prediction, kspace_target).item())
        num_batches += 1
    return total_loss / max(num_batches, 1)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.device is not None:
        config.setdefault("compute", {})
        if args.device.startswith("cuda"):
            config["compute"]["device"] = "cuda"
            config["compute"]["gpu_index"] = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        else:
            config["compute"]["device"] = args.device
    if args.override:
        config = apply_dotlist_overrides(config, args.override)

    set_random_seed(int(config["experiment"].get("seed", 42)))
    device, gpu_index = resolve_compute_device(config)
    bundle = load_fmri_data_bundle(config, PROJECT_ROOT)
    model = SubspaceINR4fMRI(config).to(device)
    model.eval()

    nufft_backend = create_nufft_backend(
        prepare_nufft_config(config, gpu_index),
        traj=bundle.traj,
        image_shape=bundle.spatial_shape,
        num_coils=bundle.num_coils,
        csm=bundle.csm,
    )

    spatial_coords = create_normalized_spatial_grid(bundle.spatial_shape).to(device)
    init_image_flat = bundle.background_init.to(device=device, dtype=torch.float32).view(-1, 2)
    init_image_flat = init_image_flat * float(args.background_scale)
    zero_init_flat = torch.zeros_like(init_image_flat)

    with torch.no_grad():
        spatial_components = _spatial_components_to_float32(model.evaluate_spatial_components(spatial_coords))

    zero_components = SpatialComponents(
        residual=None if spatial_components.residual is None else torch.zeros_like(spatial_components.residual),
        coeff=torch.zeros_like(spatial_components.coeff),
        tau=None if spatial_components.tau is None else torch.zeros_like(spatial_components.tau),
    )

    loss_with_init = _average_initial_data_loss(
        model=model,
        bundle=bundle,
        nufft_backend=nufft_backend,
        init_image_flat=init_image_flat,
        spatial_components=spatial_components,
        frame_batch_size=int(config.get("training", {}).get("frame_batch_size", 16)),
    )
    loss_without_init = _average_initial_data_loss(
        model=model,
        bundle=bundle,
        nufft_backend=nufft_backend,
        init_image_flat=zero_init_flat,
        spatial_components=spatial_components,
        frame_batch_size=int(config.get("training", {}).get("frame_batch_size", 16)),
    )
    loss_init_only = _average_initial_data_loss(
        model=model,
        bundle=bundle,
        nufft_backend=nufft_backend,
        init_image_flat=init_image_flat,
        spatial_components=zero_components,
        frame_batch_size=int(config.get("training", {}).get("frame_batch_size", 16)),
    )
    loss_zero_all = _average_initial_data_loss(
        model=model,
        bundle=bundle,
        nufft_backend=nufft_backend,
        init_image_flat=zero_init_flat,
        spatial_components=zero_components,
        frame_batch_size=int(config.get("training", {}).get("frame_batch_size", 16)),
    )

    reduction_vs_zero_init = 0.0
    if loss_without_init > 0:
        reduction_vs_zero_init = 100.0 * (loss_without_init - loss_with_init) / loss_without_init

    init_only_reduction_vs_zero = 0.0
    if loss_zero_all > 0:
        init_only_reduction_vs_zero = 100.0 * (loss_zero_all - loss_init_only) / loss_zero_all

    result: Dict[str, object] = {
        "config_path": str(Path(args.config).resolve()),
        "device": str(device),
        "nufft_backend": str(config.get("nufft", {}).get("backend", "unknown")),
        "scale_factor": float(bundle.scale_factor),
        "background_scale": float(args.background_scale),
        "frame_batch_size": int(config.get("training", {}).get("frame_batch_size", 16)),
        "loss_with_I_init_and_random_branches": loss_with_init,
        "loss_without_I_init_and_random_branches": loss_without_init,
        "loss_with_I_init_only": loss_init_only,
        "loss_with_zero_init_and_zero_branches": loss_zero_all,
        "relative_reduction_with_I_init_percent": reduction_vs_zero_init,
        "relative_reduction_I_init_only_percent": init_only_reduction_vs_zero,
        "conclusion": (
            "I_init reduces the initial data-consistency loss, so the fixed background absorbs "
            "a substantial part of the dominant signal energy before training."
            if loss_with_init < loss_without_init
            else "I_init did not reduce the initial data-consistency loss in this test."
        ),
        "notes": [
            "This is an initialization-time test only; it does not prove the final optimization target by itself.",
            "The strongest isolation is the comparison between I_init-only and zero-init/zero-branches.",
            "The random-branch comparison checks the practical starting point seen by training.",
        ],
    }
    global_cg_comparison = _compare_background_to_global_cg(config, PROJECT_ROOT)
    if global_cg_comparison is not None:
        result["comparison_to_global_cg_mean"] = global_cg_comparison

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved background-init absorption test to {output_path}")


if __name__ == "__main__":
    main()
