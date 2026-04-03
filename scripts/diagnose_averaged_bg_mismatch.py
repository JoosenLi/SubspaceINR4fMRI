from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from mrinufft import get_operator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from subspaceINR4fMRI.data import load_fmri_data_bundle
from subspaceINR4fMRI.models.complex_ops import real_imag_to_complex
from subspaceINR4fMRI.nufft import create_nufft_backend
from subspaceINR4fMRI.training.trainer_utils import prepare_nufft_config, resolve_compute_device
from subspaceINR4fMRI.utils import apply_dotlist_overrides, load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose why averagedBG does not lower the initial framewise data loss.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "subspaceINR4fMRI.yaml"),
        help="Path to the YAML config.",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device override such as cuda:0.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "averaged_bg_diagnosis.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Optional dotted config overrides.",
    )
    return parser.parse_args()


def _flatten_complex(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1).to(torch.complex64)


def _complex_scalar_fit(prediction: torch.Tensor, target: torch.Tensor) -> complex:
    pred = _flatten_complex(prediction)
    tgt = _flatten_complex(target)
    denom = torch.vdot(pred, pred)
    if torch.abs(denom) < 1.0e-12:
        return 0.0 + 0.0j
    alpha = torch.vdot(pred, tgt) / denom
    alpha_cpu = alpha.detach().cpu().numpy().item()
    return complex(alpha_cpu)


def _relative_error(prediction: torch.Tensor, target: torch.Tensor) -> float:
    num = torch.linalg.norm(_flatten_complex(prediction - target))
    den = torch.linalg.norm(_flatten_complex(target))
    return float((num / torch.clamp(den, min=1.0e-12)).item())


def _framewise_predictions(background_complex: torch.Tensor, bundle, backend) -> torch.Tensor:
    outputs = []
    for start in range(bundle.n_frames):
        frame_idx = torch.tensor([start], dtype=torch.long)
        pred = backend.forward_direct(background_complex.unsqueeze(0), frame_idx)
        outputs.append(pred)
    return torch.cat(outputs, dim=0)


def _build_stacked_operator(config: Dict[str, Any], bundle, gpu_index: int):
    nufft_cfg = prepare_nufft_config(config, gpu_index)
    backend_name = str(nufft_cfg.get("backend", "cufinufft")).lower()
    backend_kwargs = dict(nufft_cfg.get("backend_kwargs", {}))
    if backend_name == "cufinufft":
        backend_kwargs.pop("use_gpu_direct", None)
    else:
        backend_kwargs.setdefault("use_gpu_direct", True)

    traj_stacked = bundle.traj.reshape(-1, bundle.traj.shape[-1]).astype(np.float32, copy=False)
    operator = get_operator(backend_name)(
        samples=traj_stacked,
        shape=bundle.spatial_shape,
        density=nufft_cfg.get("density", True),
        n_coils=bundle.num_coils,
        smaps=bundle.csm,
        squeeze_dims=True,
        **backend_kwargs,
    )
    return operator


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    config.setdefault("compute", {})
    if args.device.startswith("cuda"):
        config["compute"]["device"] = "cuda"
        config["compute"]["gpu_index"] = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
    else:
        config["compute"]["device"] = args.device
    if args.override:
        config = apply_dotlist_overrides(config, args.override)

    device, gpu_index = resolve_compute_device(config)
    bundle = load_fmri_data_bundle(config, PROJECT_ROOT)
    backend = create_nufft_backend(
        prepare_nufft_config(config, gpu_index),
        traj=bundle.traj,
        image_shape=bundle.spatial_shape,
        num_coils=bundle.num_coils,
        csm=bundle.csm,
    )

    bg_ri = bundle.background_init.to(device=device, dtype=torch.float32)
    background_complex = real_imag_to_complex(bg_ri).to(torch.complex64).contiguous()
    zero_complex = torch.zeros_like(background_complex)
    measured_framewise = bundle.kspace.to(device=device, dtype=torch.complex64)

    with torch.no_grad():
        predicted_framewise = _framewise_predictions(background_complex, bundle, backend)
        zero_framewise = _framewise_predictions(zero_complex, bundle, backend)

    alpha_framewise = _complex_scalar_fit(predicted_framewise, measured_framewise)
    alpha_framewise_tensor = torch.tensor(alpha_framewise, device=device, dtype=torch.complex64)
    fitted_framewise = predicted_framewise * alpha_framewise_tensor

    framewise_alpha_per_frame = []
    for frame in range(bundle.n_frames):
        alpha_t = _complex_scalar_fit(predicted_framewise[frame], measured_framewise[frame])
        framewise_alpha_per_frame.append(alpha_t)

    stacked_operator = _build_stacked_operator(config, bundle, gpu_index)
    kspace_stacked_target = measured_framewise.reshape(bundle.n_frames, bundle.num_coils, -1).permute(1, 0, 2).reshape(bundle.num_coils, -1)
    with torch.no_grad():
        predicted_stacked = stacked_operator.op(background_complex.contiguous())
        zero_stacked = stacked_operator.op(zero_complex.contiguous())

    predicted_stacked_t = torch.as_tensor(predicted_stacked, device=device, dtype=torch.complex64)
    zero_stacked_t = torch.as_tensor(zero_stacked, device=device, dtype=torch.complex64)
    target_stacked_t = kspace_stacked_target.to(device=device, dtype=torch.complex64)

    alpha_stacked = _complex_scalar_fit(predicted_stacked_t, target_stacked_t)
    alpha_stacked_tensor = torch.tensor(alpha_stacked, device=device, dtype=torch.complex64)
    fitted_stacked = predicted_stacked_t * alpha_stacked_tensor

    global_cg_path = (PROJECT_ROOT / config["data"]["background_init_path"]).resolve().parent / "Global_CG.npy"
    cg_diag = {}
    if global_cg_path.exists():
        global_cg = np.load(global_cg_path)
        if not np.iscomplexobj(global_cg):
            global_cg = global_cg[..., 0] + 1j * global_cg[..., 1]
        global_cg_mean = global_cg.mean(axis=0)
        bg_np = background_complex.detach().cpu().numpy()
        alpha_image = np.vdot(bg_np.reshape(-1), global_cg_mean.reshape(-1)) / np.vdot(bg_np.reshape(-1), bg_np.reshape(-1))
        cg_diag = {
            "global_cg_path": str(global_cg_path),
            "best_fit_complex_scale_real": float(np.real(alpha_image)),
            "best_fit_complex_scale_imag": float(np.imag(alpha_image)),
            "best_fit_complex_scale_abs": float(np.abs(alpha_image)),
            "best_fit_complex_scale_phase_deg": float(np.angle(alpha_image, deg=True)),
            "mean_magnitude_ratio_global_cg_over_background": float(np.abs(global_cg_mean).mean() / max(np.abs(bg_np).mean(), 1.0e-12)),
            "magnitude_correlation": float(np.corrcoef(np.abs(global_cg_mean).reshape(-1), np.abs(bg_np).reshape(-1))[0, 1]),
        }

    per_frame_alpha_abs = np.abs(np.asarray(framewise_alpha_per_frame, dtype=np.complex64))
    per_frame_alpha_phase = np.angle(np.asarray(framewise_alpha_per_frame, dtype=np.complex64), deg=True)

    result = {
        "config_path": str(Path(args.config).resolve()),
        "device": str(device),
        "nufft_backend": str(config.get("nufft", {}).get("backend", "unknown")),
        "framewise": {
            "relative_error_background_vs_measured": _relative_error(predicted_framewise, measured_framewise),
            "relative_error_zero_vs_measured": _relative_error(zero_framewise, measured_framewise),
            "best_fit_complex_scale_real": float(np.real(alpha_framewise)),
            "best_fit_complex_scale_imag": float(np.imag(alpha_framewise)),
            "best_fit_complex_scale_abs": float(np.abs(alpha_framewise)),
            "best_fit_complex_scale_phase_deg": float(np.angle(alpha_framewise, deg=True)),
            "relative_error_after_best_scalar_fit": _relative_error(fitted_framewise, measured_framewise),
            "per_frame_scale_abs_mean": float(per_frame_alpha_abs.mean()),
            "per_frame_scale_abs_std": float(per_frame_alpha_abs.std()),
            "per_frame_scale_phase_deg_mean": float(per_frame_alpha_phase.mean()),
            "per_frame_scale_phase_deg_std": float(per_frame_alpha_phase.std()),
        },
        "stacked_operator": {
            "relative_error_background_vs_measured": _relative_error(predicted_stacked_t, target_stacked_t),
            "relative_error_zero_vs_measured": _relative_error(zero_stacked_t, target_stacked_t),
            "best_fit_complex_scale_real": float(np.real(alpha_stacked)),
            "best_fit_complex_scale_imag": float(np.imag(alpha_stacked)),
            "best_fit_complex_scale_abs": float(np.abs(alpha_stacked)),
            "best_fit_complex_scale_phase_deg": float(np.angle(alpha_stacked, deg=True)),
            "relative_error_after_best_scalar_fit": _relative_error(fitted_stacked, target_stacked_t),
        },
        "comparison_to_global_cg_mean": cg_diag,
    }

    frame_scale = result["framewise"]["best_fit_complex_scale_abs"]
    stacked_scale = result["stacked_operator"]["best_fit_complex_scale_abs"]
    frame_error = result["framewise"]["relative_error_background_vs_measured"]
    frame_fit_error = result["framewise"]["relative_error_after_best_scalar_fit"]
    if frame_scale > 1.2 and frame_fit_error < frame_error:
        diagnosis = (
            "The main mismatch is a global amplitude mismatch: averagedBG has the right spatial pattern "
            "but is too small under the current training forward model."
        )
    else:
        diagnosis = (
            "The mismatch is not dominated by a simple global amplitude factor; check operator conventions "
            "or background reconstruction details."
        )
    result["diagnosis"] = diagnosis

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved averagedBG diagnosis to {output_path}")


if __name__ == "__main__":
    main()
