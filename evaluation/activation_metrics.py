from __future__ import annotations

from dataclasses import dataclass
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/hashinr_mplconfig")

from snake.core.handlers import BlockActivationHandler
from snake.core.transform import apply_affine
from snake.mrd_utils import NonCartesianFrameDataLoader

from ..data import create_normalized_spatial_grid, load_fmri_data_bundle
from ..models import SubspaceINR4fMRI
from ..training.trainer_utils import resolve_compute_device
from ..utilis.utilis_plot import cal_z_score, cluster_threshold_3d, get_f2_analysis
from ..utils import resolve_path


@dataclass
class SimulationEvaluationContext:
    sim_conf: Any
    dyn_datas: Sequence[Any]
    activation_handler: BlockActivationHandler
    roi_resampled: np.ndarray
    roi_thresholded: np.ndarray


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        scalar = float(np.asarray(value).reshape(-1)[0])
    except Exception:
        return default
    return scalar if np.isfinite(scalar) else default


def _summarize_stats_results(stats_results: Dict[str, Any]) -> Dict[str, Any]:
    threshold_range = np.asarray(
        stats_results.get("thres_range", stats_results.get("tresh", [])),
        dtype=np.float32,
    ).reshape(-1)
    f2_scores = np.asarray(stats_results.get("f2s", []), dtype=np.float32).reshape(-1)
    precision = np.asarray(stats_results.get("precisions", []), dtype=np.float32).reshape(-1)
    recall = np.asarray(stats_results.get("recalls", []), dtype=np.float32).reshape(-1)
    ious = np.asarray(stats_results.get("ious", []), dtype=np.float32).reshape(-1)

    if f2_scores.size == 0 and {"tp", "fp", "fn"}.issubset(stats_results):
        tp = np.asarray(stats_results.get("tp", []), dtype=np.float32).reshape(-1)
        fp = np.asarray(stats_results.get("fp", []), dtype=np.float32).reshape(-1)
        fn = np.asarray(stats_results.get("fn", []), dtype=np.float32).reshape(-1)
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tp / np.maximum(tp + fn, 1.0)
        beta_sq = 4.0
        f2_scores = (1.0 + beta_sq) * precision * recall / np.maximum(beta_sq * precision + recall, 1.0e-12)
        ious = tp / np.maximum(tp + fp + fn, 1.0)

    best_index = int(np.argmax(f2_scores)) if f2_scores.size > 0 else -1
    fixed_threshold = 3.3
    fixed_index = int(np.argmin(np.abs(threshold_range - fixed_threshold))) if threshold_range.size > 0 else -1

    summary = {
        "num_thresholds": int(threshold_range.size),
        "best_index": best_index,
        "best_threshold": _safe_float(threshold_range[best_index]) if best_index >= 0 else 0.0,
        "best_f2": _safe_float(f2_scores[best_index]) if best_index >= 0 else 0.0,
        "best_precision": _safe_float(precision[best_index]) if best_index >= 0 and precision.size > best_index else 0.0,
        "best_recall": _safe_float(recall[best_index]) if best_index >= 0 and recall.size > best_index else 0.0,
        "best_iou": _safe_float(ious[best_index]) if best_index >= 0 and ious.size > best_index else 0.0,
        "fixed_threshold": fixed_threshold,
        "fixed_index": fixed_index,
        "fixed_f2": _safe_float(f2_scores[fixed_index]) if fixed_index >= 0 else 0.0,
        "fixed_precision": _safe_float(precision[fixed_index]) if fixed_index >= 0 and precision.size > fixed_index else 0.0,
        "fixed_recall": _safe_float(recall[fixed_index]) if fixed_index >= 0 and recall.size > fixed_index else 0.0,
        "fixed_iou": _safe_float(ious[fixed_index]) if fixed_index >= 0 and ious.size > fixed_index else 0.0,
    }
    return summary


@lru_cache(maxsize=8)
def _load_simulation_context_cached(
    dataloader_path_str: str,
    block_off: int,
    block_on: int,
    duration: int,
    atlas: str,
    atlas_label: int,
    phantom_sub_id: int,
    tissue_file: str,
    phantom_output_res: int,
    use_gpu_resample: bool,
) -> SimulationEvaluationContext:
    dataloader_path = Path(dataloader_path_str)
    with NonCartesianFrameDataLoader(str(dataloader_path)) as data_loader:
        sim_conf = data_loader.get_sim_conf()
        dyn_datas = data_loader.get_all_dynamic()
        try:
            phantom = data_loader.get_phantom()
        except Exception:
            from snake.core.phantom import Phantom

            phantom_original = Phantom.from_brainweb(
                sub_id=phantom_sub_id,
                sim_conf=sim_conf,
                tissue_file=tissue_file,
                output_res=phantom_output_res,
            )
            phantom = phantom_original.resample(
                new_affine=sim_conf.fov.affine,
                new_shape=sim_conf.fov.shape,
                use_gpu=use_gpu_resample,
            )

    activation_handler = BlockActivationHandler(
        block_off=block_off,
        block_on=block_on,
        duration=duration,
        atlas=atlas,
        atlas_label=atlas_label,
    )
    if hasattr(phantom, "masks") and hasattr(phantom, "labels_idx") and "ROI" in phantom.labels_idx:
        roi = phantom.masks[phantom.labels_idx["ROI"]]
    else:
        example_phantom = activation_handler.get_static(phantom.copy(), sim_conf)
        roi = example_phantom.masks[example_phantom.labels_idx["ROI"]]
    roi_resampled = apply_affine(
        roi,
        new_affine=sim_conf.fov.affine,
        old_affine=phantom.affine,
        new_shape=sim_conf.fov.shape,
    )
    return SimulationEvaluationContext(
        sim_conf=sim_conf,
        dyn_datas=dyn_datas,
        activation_handler=activation_handler,
        roi_resampled=np.asarray(roi_resampled),
        roi_thresholded=np.asarray(roi_resampled) > 0.05,
    )


def load_simulation_context(config: Dict[str, Any], project_root: Path = PROJECT_ROOT) -> SimulationEvaluationContext:
    evaluation_cfg = config.get("evaluation", {})
    dataloader_path = resolve_path(
        evaluation_cfg.get("dataloader_mrd_path", "dataset/sim3DT/Simulation_Mar16_RotatedSpiral/dataloader.mrd"),
        project_root,
    )
    if dataloader_path is None or not dataloader_path.exists():
        raise FileNotFoundError(f"Could not find evaluation dataloader.mrd at {dataloader_path}")

    sim_cfg = evaluation_cfg.get("simulation", {})
    return _load_simulation_context_cached(
        str(dataloader_path),
        int(sim_cfg.get("block_off", 20)),
        int(sim_cfg.get("block_on", 20)),
        int(sim_cfg.get("duration", 240)),
        str(sim_cfg.get("atlas", "hardvard-oxford__cort-maxprob-thr0-1mm")),
        int(sim_cfg.get("atlas_label", 48)),
        int(sim_cfg.get("phantom_sub_id", 4)),
        str(sim_cfg.get("tissue_file", "tissue_3T")),
        int(sim_cfg.get("phantom_output_res", 1)),
        bool(sim_cfg.get("use_gpu_resample", True)),
    )


def reconstruction_to_magnitude(reconstruction: np.ndarray, scale_factor: float = 1.0, input_is_scaled: bool = True) -> np.ndarray:
    array = np.asarray(reconstruction)
    if np.iscomplexobj(array):
        magnitude = np.abs(array)
    elif array.ndim >= 1 and array.shape[-1] == 2:
        magnitude = np.linalg.norm(array.astype(np.float32), axis=-1)
    else:
        magnitude = np.abs(array.astype(np.float32))

    if input_is_scaled and scale_factor not in (0.0, 1.0):
        magnitude = magnitude / float(scale_factor)
    return magnitude.astype(np.float32, copy=False)


def compute_roi_timeseries(reconstruction: np.ndarray, roi_mask: np.ndarray, tr_seconds: float) -> Tuple[np.ndarray, float]:
    arr = np.abs(reconstruction).astype(np.float32, copy=False)
    roi = np.asarray(roi_mask, dtype=bool)
    if arr.shape[1:] != roi.shape:
        raise ValueError(f"ROI shape {roi.shape} does not match reconstruction spatial shape {arr.shape[1:]}")

    roi_values = arr[..., roi]
    if roi_values.ndim == 1:
        roi_values = roi_values[:, None]
    tsnr = float(np.median(np.mean(np.abs(roi_values), axis=0) / (np.std(np.abs(roi_values), axis=0) + 1.0e-10)))
    centered = (roi_values - np.mean(roi_values, axis=0)) / (np.std(np.abs(roi_values), axis=0) + 1.0e-10)
    ts = np.mean(centered, axis=-1)
    return ts.astype(np.float32, copy=False), tsnr


def compute_expected_signals(context: SimulationEvaluationContext, num_frames: int) -> Dict[str, np.ndarray]:
    waveform_name = f"activation-{context.activation_handler.event_name}"
    matched_dynamic = None
    for dynamic in context.dyn_datas:
        if dynamic.name == waveform_name:
            matched_dynamic = dynamic
            break
    if matched_dynamic is None:
        matched_dynamic = context.dyn_datas[0]

    data = np.asarray(matched_dynamic.data)
    expected_signal = data[0].reshape(num_frames, -1).mean(axis=1)
    activation_event = data[1].reshape(num_frames, -1).mean(axis=1) if data.shape[0] > 1 else np.zeros(num_frames, dtype=np.float32)
    expected_signal = (expected_signal - expected_signal.mean()) / (expected_signal.std() + 1.0e-10)
    time_samples = np.arange(num_frames, dtype=np.float32) * float(context.sim_conf.seq.TR / 1000)
    return {
        "expected_signal": expected_signal.astype(np.float32, copy=False),
        "activation_event": activation_event.astype(np.float32, copy=False),
        "time_samples": time_samples,
    }


def evaluate_magnitude_reconstruction(
    reconstruction_magnitude: np.ndarray,
    config: Dict[str, Any],
    project_root: Path = PROJECT_ROOT,
    output_dir: Optional[Path] = None,
    method_name: str = "reconstruction",
) -> Dict[str, Any]:
    evaluation_cfg = config.get("evaluation", {})
    threshold = float(evaluation_cfg.get("roi_threshold", 0.05))
    target_threshold = float(evaluation_cfg.get("f2_target_threshold", 3.3))
    tr_seconds = float(evaluation_cfg.get("tr_seconds", config.get("time", {}).get("tr_seconds", 1.0)))
    save_arrays = bool(evaluation_cfg.get("save_activation_arrays", False))
    save_filtered = bool(evaluation_cfg.get("save_filtered_activation", False))

    context = load_simulation_context(config, project_root=project_root)
    reconstruction_magnitude = np.asarray(reconstruction_magnitude, dtype=np.float32)
    z_score, stats_results = cal_z_score(
        reconstruction_magnitude,
        activation_handler=context.activation_handler,
        dyn_datas=context.dyn_datas,
        sim_conf=context.sim_conf,
        threshold=threshold,
        TR=tr_seconds,
        roi_resampled=context.roi_resampled,
    )
    f2_metrics = get_f2_analysis(stats_results, target_threshold=target_threshold)

    roi_mask = context.roi_resampled > threshold
    roi_timeseries, tsnr = compute_roi_timeseries(reconstruction_magnitude, roi_mask, tr_seconds=tr_seconds)
    signals = compute_expected_signals(context, reconstruction_magnitude.shape[0])
    roi_corr = float(np.corrcoef(roi_timeseries, signals["expected_signal"][: roi_timeseries.shape[0]])[0, 1])
    stats_summary = _summarize_stats_results(stats_results)

    result = {
        "method_name": method_name,
        "num_frames": int(reconstruction_magnitude.shape[0]),
        "roi_threshold": threshold,
        "f2_target_threshold": target_threshold,
        "f2_fixed": float(f2_metrics["f2_fixed"]),
        "f2_max": float(f2_metrics["f2_max"]),
        "best_thresh": float(f2_metrics["best_thresh"]),
        "actual_thresh": float(f2_metrics["actual_thresh"]),
        "roi_tsnr": tsnr,
        "roi_signal_corr": roi_corr,
        "stats_summary": stats_summary,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (output_dir / "activation_scores.json").write_text(
            json.dumps(_to_serializable(stats_results), indent=2),
            encoding="utf-8",
        )
        if save_arrays:
            np.save(output_dir / "z_score.npy", z_score.astype(np.float32, copy=False))
            np.save(output_dir / "roi_timeseries.npy", roi_timeseries.astype(np.float32, copy=False))
        if save_filtered:
            filtered, info = cluster_threshold_3d(
                act=z_score,
                voxel_thresh=float(evaluation_cfg.get("cluster_voxel_thresh", 3.0)),
                min_cluster_size=int(evaluation_cfg.get("cluster_min_size", 100)),
                connectivity=int(evaluation_cfg.get("cluster_connectivity", 6)),
                two_sided=bool(evaluation_cfg.get("cluster_two_sided", False)),
                use_abs=bool(evaluation_cfg.get("cluster_use_abs", False)),
                return_labels=False,
            )
            np.save(output_dir / "filtered_activation.npy", np.asarray(filtered, dtype=np.float32))
            (output_dir / "filtered_activation_info.json").write_text(
                json.dumps(_to_serializable(info), indent=2),
                encoding="utf-8",
            )
    return result


def _spatial_components_to_float32(spatial_components):
    residual = None if spatial_components.residual is None else spatial_components.residual.float()
    tau = None if spatial_components.tau is None else spatial_components.tau.float()
    return type(spatial_components)(
        residual=residual,
        coeff=spatial_components.coeff.float(),
        tau=tau,
    )


@torch.no_grad()
def reconstruct_checkpoint(
    checkpoint_path: str | Path,
    project_root: Path = PROJECT_ROOT,
    device_override: Optional[str] = None,
    frame_batch_size: Optional[int] = None,
    output_path: Optional[Path] = None,
    unscale_output: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    runtime_config = dict(checkpoint["config"])
    runtime_config.setdefault("compute", {})
    if device_override is not None:
        if device_override.startswith("cuda"):
            runtime_config["compute"]["device"] = "cuda"
            runtime_config["compute"]["gpu_index"] = int(device_override.split(":", 1)[1]) if ":" in device_override else 0
        else:
            runtime_config["compute"]["device"] = device_override
    device, _ = resolve_compute_device(runtime_config)
    bundle = load_fmri_data_bundle(runtime_config, project_root)
    model = SubspaceINR4fMRI(runtime_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    spatial_coords = create_normalized_spatial_grid(bundle.spatial_shape).to(device)
    init_image_flat = bundle.background_init.to(device=device, dtype=torch.float32).view(-1, 2)
    time_coords = bundle.time_coords.to(device=device, dtype=torch.float32)
    batch_size = int(frame_batch_size or runtime_config.get("training", {}).get("frame_batch_size", 16))

    spatial_components = _spatial_components_to_float32(
        model.evaluate_spatial_components(
            spatial_coords,
            chunk_size=model.spatial_chunk_size if model.chunked_spatial_eval else None,
        )
    )

    predictions = []
    for start in range(0, bundle.n_frames, batch_size):
        t_batch = time_coords[start : start + batch_size]
        basis, basis_derivative = model.evaluate_temporal_basis(t_batch, need_derivative=model.use_delay)
        prediction = model.synthesize_batch(
            init_image_flat,
            spatial_components,
            basis.float(),
            bundle.spatial_shape,
            basis_derivative=basis_derivative.float() if basis_derivative is not None else None,
        )
        predictions.append(prediction.cpu().numpy())

    reconstruction = np.concatenate(predictions, axis=0)
    if unscale_output and bundle.scale_factor not in (0.0, 1.0):
        reconstruction = reconstruction / float(bundle.scale_factor)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, reconstruction)
    metadata = {
        "scale_factor": float(bundle.scale_factor),
        "config": runtime_config,
    }
    return reconstruction.astype(np.float32, copy=False), metadata


def evaluate_reconstruction_path(
    reconstruction_path: str | Path,
    config: Dict[str, Any],
    project_root: Path = PROJECT_ROOT,
    output_dir: Optional[Path] = None,
    method_name: Optional[str] = None,
    scale_factor: Optional[float] = None,
    input_is_scaled: bool = True,
) -> Dict[str, Any]:
    reconstruction_path = Path(reconstruction_path)
    reconstruction = np.load(reconstruction_path)
    magnitude = reconstruction_to_magnitude(
        reconstruction,
        scale_factor=float(scale_factor if scale_factor is not None else config.get("data", {}).get("scale_factor", 1.0)),
        input_is_scaled=input_is_scaled,
    )
    return evaluate_magnitude_reconstruction(
        magnitude,
        config=config,
        project_root=project_root,
        output_dir=output_dir,
        method_name=method_name or reconstruction_path.stem,
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    project_root: Path = PROJECT_ROOT,
    output_dir: Optional[Path] = None,
    device_override: Optional[str] = None,
    frame_batch_size: Optional[int] = None,
) -> Dict[str, Any]:
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    config = checkpoint["config"]
    reconstruction, metadata = reconstruct_checkpoint(
        checkpoint_path,
        project_root=project_root,
        device_override=device_override,
        frame_batch_size=frame_batch_size,
        output_path=None,
        unscale_output=True,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "reconstruction_from_checkpoint.npy", reconstruction)
    magnitude = reconstruction_to_magnitude(reconstruction, scale_factor=1.0, input_is_scaled=False)
    result = evaluate_magnitude_reconstruction(
        magnitude,
        config=config,
        project_root=project_root,
        output_dir=output_dir,
        method_name=Path(checkpoint_path).stem,
    )
    result["scale_factor"] = metadata["scale_factor"]
    return result
