from __future__ import annotations

from copy import deepcopy
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from ..data import build_data_bundle_from_arrays, create_normalized_spatial_grid, load_fmri_data_bundle
from ..evaluation import evaluate_magnitude_reconstruction
from ..models import SubspaceINR4fMRI, complex_mse, real_imag_to_complex
from ..nufft import create_nufft_backend
from ..utils import (
    create_summary_writer,
    get_peak_memory_stats,
    reset_peak_memory_stats,
    safe_item,
    save_checkpoint,
)
from ..utils.tensor_utils import compute_grad_norm
from .losses import residual_energy_loss, tau_smoothness_loss, temporal_basis_smoothness_loss
from .schedulers import build_scheduler
from .trainer_utils import (
    build_optimizer,
    maybe_log_histograms,
    prepare_nufft_config,
    prepare_run_directories,
    resolve_compute_device,
    set_random_seed,
    warmup_lambda,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]



def _loss_weights(config: Dict, epoch: int) -> Dict[str, float]:
    losses_cfg = config["losses"]
    warmup_cfg = config.get("warmup", {})

    return {
        "data": float(losses_cfg.get("data", {}).get("weight", 1.0)),
        "residual": warmup_lambda(
            float(losses_cfg.get("residual_energy", {}).get("lambda", 0.0)),
            epoch,
            int(warmup_cfg.get("residual_lambda_warmup_epochs", 0)),
        ),
        "tau": warmup_lambda(
            float(losses_cfg.get("tau_smooth", {}).get("lambda", 0.0)),
            epoch,
            int(warmup_cfg.get("tau_lambda_warmup_epochs", 0)),
        ),
        "temporal": warmup_lambda(
            float(losses_cfg.get("temporal_basis_smooth", {}).get("lambda", 0.0)),
            epoch,
            int(warmup_cfg.get("temporal_smooth_lambda_warmup_epochs", 0)),
        ),
    }



def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)



def _spatial_components_to_float32(spatial_components):
    residual = None if spatial_components.residual is None else spatial_components.residual.float()
    tau = None if spatial_components.tau is None else spatial_components.tau.float()
    return type(spatial_components)(
        residual=residual,
        coeff=spatial_components.coeff.float(),
        tau=tau,
    )



def _magnitude_stats(tensor: torch.Tensor) -> Dict[str, float]:
    detached = tensor.detach()
    if torch.is_complex(detached):
        values = detached.abs().reshape(-1).float()
    elif detached.ndim > 0 and detached.shape[-1] == 2:
        values = torch.linalg.vector_norm(detached.float(), dim=-1).reshape(-1)
    else:
        values = detached.float().abs().reshape(-1)

    if values.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}

    quantile_values = values
    if quantile_values.numel() > 1_000_000:
        stride = max(1, quantile_values.numel() // 1_000_000)
        quantile_values = quantile_values[::stride]

    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
        "p95": float(torch.quantile(quantile_values, 0.95).item()),
    }



def _collect_startup_diagnostics(
    model: SubspaceINR4fMRI,
    bundle,
    spatial_coords: torch.Tensor,
    init_image_flat: torch.Tensor,
    full_time_coords: torch.Tensor,
    device: torch.device,
) -> Dict[str, object]:
    scale_factor = float(bundle.scale_factor)
    diagnostics = {
        "scale_factor": scale_factor,
        "kspace_unscaled_abs": _magnitude_stats(bundle.kspace / max(scale_factor, 1.0e-12)),
        "kspace_scaled_abs": _magnitude_stats(bundle.kspace),
        "background_unscaled_abs": _magnitude_stats(bundle.background_init / max(scale_factor, 1.0e-12)),
        "background_scaled_abs": _magnitude_stats(bundle.background_init),
    }

    model.eval()
    with torch.no_grad():
        spatial_components = _spatial_components_to_float32(model.evaluate_spatial_components(spatial_coords))
        if model.use_delay:
            with torch.enable_grad():
                basis, basis_derivative = model.evaluate_temporal_basis(full_time_coords[:1], need_derivative=True)
            basis = basis.detach()
            basis_derivative = None if basis_derivative is None else basis_derivative.detach()
        else:
            basis, basis_derivative = model.evaluate_temporal_basis(full_time_coords[:1], need_derivative=False)
        prediction = model.synthesize_batch(
            init_image_flat,
            spatial_components,
            basis.float(),
            bundle.spatial_shape,
            basis_derivative=None if basis_derivative is None else basis_derivative.float(),
        )
        diagnostics["initial_prediction_abs"] = _magnitude_stats(prediction)
        diagnostics["residual_branch_enabled"] = bool(model.use_residual)
        if spatial_components.residual is not None:
            diagnostics["initial_residual_abs"] = _magnitude_stats(spatial_components.residual)
        diagnostics["initial_coeff_abs"] = _magnitude_stats(spatial_components.coeff)
        if spatial_components.tau is not None:
            diagnostics["initial_tau_abs"] = _magnitude_stats(spatial_components.tau)
    model.train()
    return diagnostics



def _make_epoch_frame_batches(n_frames: int, batch_size: int, shuffle: bool) -> list[torch.Tensor]:
    if batch_size <= 0:
        raise ValueError(f"frame_batch_size must be positive, got {batch_size}.")
    frame_order = torch.randperm(n_frames, dtype=torch.long) if shuffle else torch.arange(n_frames, dtype=torch.long)
    return [frame_order[start : start + batch_size] for start in range(0, n_frames, batch_size)]



def _select_temporal_regularization_coords(full_time_coords: torch.Tensor, num_samples: int) -> torch.Tensor:
    if num_samples <= 0 or num_samples >= int(full_time_coords.shape[0]):
        return full_time_coords
    sample_positions = torch.linspace(
        0,
        full_time_coords.shape[0] - 1,
        steps=num_samples,
        device=full_time_coords.device,
    ).round().long().unique(sorted=True)
    return full_time_coords.index_select(0, sample_positions)



def _run_gradient_sanity_check(
    model: SubspaceINR4fMRI,
    nufft_backend,
    kspace_source: torch.Tensor,
    spatial_coords: torch.Tensor,
    init_image_flat: torch.Tensor,
    full_time_coords: torch.Tensor,
    spatial_shape,
    frame_batch_size: int,
) -> Dict[str, float]:
    frame_count = max(1, min(frame_batch_size, int(full_time_coords.shape[0]), 2))
    frame_indices_cpu = torch.arange(frame_count, dtype=torch.long)
    frame_indices_device = frame_indices_cpu.to(device=full_time_coords.device)

    model.zero_grad(set_to_none=True)
    spatial_components = _spatial_components_to_float32(model.evaluate_spatial_components(spatial_coords))
    t_batch = full_time_coords.index_select(0, frame_indices_device)
    basis_batch, basis_derivative_batch = model.evaluate_temporal_basis(
        t_batch,
        need_derivative=model.use_delay,
    )
    prediction_ri = model.synthesize_batch(
        init_image_flat,
        spatial_components,
        basis_batch.float(),
        spatial_shape,
        basis_derivative=basis_derivative_batch.float() if basis_derivative_batch is not None else None,
    )
    prediction_complex = real_imag_to_complex(prediction_ri).to(torch.complex64).contiguous()
    prediction_complex.retain_grad()

    kspace_prediction = nufft_backend.forward(prediction_complex, frame_indices_cpu)
    kspace_target = kspace_source.index_select(0, frame_indices_device)
    loss = complex_mse(kspace_prediction, kspace_target)
    loss.backward()

    image_grad = prediction_complex.grad
    image_grad_mean = 0.0 if image_grad is None else float(image_grad.abs().mean().item())
    parameter_grad_norm = compute_grad_norm(model.parameters())
    result = {
        "loss": float(loss.detach().cpu().item()),
        "image_grad_mean_abs": image_grad_mean,
        "parameter_grad_norm": parameter_grad_norm,
    }
    model.zero_grad(set_to_none=True)

    if not np.isfinite(result["loss"]):
        raise RuntimeError("Gradient sanity check produced a non-finite data loss.")
    if result["image_grad_mean_abs"] <= 0.0 or result["parameter_grad_norm"] <= 0.0:
        raise RuntimeError(
            "Gradient sanity check failed: NUFFT data term did not propagate non-zero gradients "
            f"(image_grad_mean_abs={result['image_grad_mean_abs']:.3e}, "
            f"parameter_grad_norm={result['parameter_grad_norm']:.3e})."
        )
    return result




def _reconstruct_full_sequence(
    model: SubspaceINR4fMRI,
    bundle,
    spatial_coords: torch.Tensor,
    init_image_flat: torch.Tensor,
    frame_batch_size: int,
    output_scale_factor: float = 1.0,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        spatial_components = _spatial_components_to_float32(
            model.evaluate_spatial_components(
                spatial_coords,
                chunk_size=model.spatial_chunk_size if model.chunked_spatial_eval else None,
            )
        )

    recon_batches = []
    full_time = bundle.time_coords.to(init_image_flat.device)
    for start in range(0, bundle.n_frames, frame_batch_size):
        t_batch = full_time[start : start + frame_batch_size]
        basis, basis_derivative = model.evaluate_temporal_basis(t_batch, need_derivative=model.use_delay)
        with torch.no_grad():
            prediction = model.synthesize_batch(
                init_image_flat,
                spatial_components,
                basis.float(),
                bundle.spatial_shape,
                basis_derivative=basis_derivative.float() if basis_derivative is not None else None,
            )
        recon_batches.append(prediction.cpu().numpy())

    reconstruction = np.concatenate(recon_batches, axis=0)
    if output_scale_factor not in (0.0, 1.0):
        reconstruction = reconstruction / float(output_scale_factor)
    model.train()
    return reconstruction.astype(np.float32, copy=False)

def _save_reconstruction_snapshot(
    model: SubspaceINR4fMRI,
    bundle,
    spatial_coords: torch.Tensor,
    init_image_flat: torch.Tensor,
    recon_dir: Path,
    epoch: int,
    frame_batch_size: int,
    output_scale_factor: float = 1.0,
) -> Path:
    snapshot = _reconstruct_full_sequence(
        model=model,
        bundle=bundle,
        spatial_coords=spatial_coords,
        init_image_flat=init_image_flat,
        frame_batch_size=frame_batch_size,
        output_scale_factor=output_scale_factor,
    )
    snapshot_path = recon_dir / f"reconstruction_epoch_{epoch:04d}.npy"
    np.save(snapshot_path, snapshot)
    return snapshot_path



def _train_impl(config: Dict, bundle):
    cudnn.benchmark = True
    device, gpu_index = resolve_compute_device(config)
    experiment_dirs = prepare_run_directories(config, PROJECT_ROOT)

    set_random_seed(int(config["experiment"].get("seed", 42)))

    training_cfg = config.get("training", {})
    losses_cfg = config["losses"]
    logging_cfg = config.get("logging", {})
    evaluation_cfg = config.get("evaluation", {})

    spatial_coords = create_normalized_spatial_grid(bundle.spatial_shape).to(device)
    init_image_flat = bundle.background_init.to(device=device, dtype=torch.float32).view(-1, 2)
    full_time_coords = bundle.time_coords.to(device=device, dtype=torch.float32)
    preload_kspace_to_device = bool(training_cfg.get("preload_kspace_to_device", device.type == "cuda"))
    if preload_kspace_to_device:
        kspace_storage = bundle.kspace.to(device=device, dtype=torch.complex64)
    else:
        kspace_storage = bundle.kspace.to(dtype=torch.complex64)

    model = SubspaceINR4fMRI(config).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    use_amp = bool(training_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    nufft_backend = create_nufft_backend(
        prepare_nufft_config(config, gpu_index),
        traj=bundle.traj,
        image_shape=bundle.spatial_shape,
        num_coils=bundle.num_coils,
        csm=bundle.csm,
    )

    writer = create_summary_writer(
        experiment_dirs["log_dir"],
        enabled=bool(logging_cfg.get("tensorboard", False)),
    )

    frame_batch_size = int(training_cfg.get("frame_batch_size", 1))
    shuffle_frames = bool(training_cfg.get("shuffle_frames_each_epoch", True))
    num_epochs = int(training_cfg.get("num_epochs", 1))
    zero_grad_set_to_none = bool(training_cfg.get("zero_grad_set_to_none", True))
    grad_clip = float(training_cfg.get("grad_clip", 0.0))
    detect_anomaly = bool(training_cfg.get("detect_anomaly", False))
    gradient_sanity_check_enabled = bool(training_cfg.get("gradient_sanity_check", False))
    startup_diagnostics_enabled = bool(training_cfg.get("startup_diagnostics", False))

    temporal_reg_cfg = losses_cfg.get("temporal_basis_smooth", {})
    temporal_reg_num_samples = int(temporal_reg_cfg.get("num_time_samples", frame_batch_size))
    temporal_reg_reuse_batch = bool(temporal_reg_cfg.get("reuse_batch_derivative", True))
    temporal_reg_coords = _select_temporal_regularization_coords(full_time_coords, temporal_reg_num_samples)
    profile_first_steps = int(training_cfg.get("profile_first_steps", 0 if device.type == "cuda" else 0))

    log_interval = int(logging_cfg.get("log_interval", 20))
    hist_interval = int(logging_cfg.get("hist_interval", 0))
    save_interval = int(logging_cfg.get("save_interval", 100))
    save_snapshot = bool(logging_cfg.get("save_reconstruction_snapshot", False))
    save_snapshot_every = int(logging_cfg.get("save_reconstruction_every", save_interval))
    log_peak_memory = bool(logging_cfg.get("log_peak_memory", True))

    evaluation_enabled = bool(evaluation_cfg.get("enabled", False))
    evaluation_interval = int(evaluation_cfg.get("every_n_epochs", 0))
    evaluation_primary_metric = str(evaluation_cfg.get("primary_metric", "f2_fixed"))
    save_best_checkpoint = bool(evaluation_cfg.get("save_best_checkpoint", True))
    save_reconstruction_on_eval = bool(evaluation_cfg.get("save_reconstruction_on_eval", False))

    startup_diagnostics = {}
    diagnostics_path = None
    if startup_diagnostics_enabled:
        startup_diagnostics = _collect_startup_diagnostics(
            model=model,
            bundle=bundle,
            spatial_coords=spatial_coords,
            init_image_flat=init_image_flat,
            full_time_coords=full_time_coords,
            device=device,
        )
        diagnostics_path = experiment_dirs["run_dir"] / "startup_diagnostics.json"
        with diagnostics_path.open("w", encoding="utf-8") as handle:
            json.dump(startup_diagnostics, handle, indent=2)
        print(f"Startup diagnostics saved to {diagnostics_path}")
        print(json.dumps(startup_diagnostics, indent=2))

    if gradient_sanity_check_enabled:
        sanity_result = _run_gradient_sanity_check(
            model=model,
            nufft_backend=nufft_backend,
            kspace_source=kspace_storage if preload_kspace_to_device else bundle.kspace.to(device=device, dtype=torch.complex64),
            spatial_coords=spatial_coords,
            init_image_flat=init_image_flat,
            full_time_coords=full_time_coords,
            spatial_shape=bundle.spatial_shape,
            frame_batch_size=frame_batch_size,
        )
        sanity_path = experiment_dirs["run_dir"] / "gradient_sanity_check.json"
        with sanity_path.open("w", encoding="utf-8") as handle:
            json.dump(sanity_result, handle, indent=2)
        print(f"Gradient sanity check passed: {json.dumps(sanity_result, indent=2)}")
    else:
        sanity_result = {}
        sanity_path = None

    global_step = 0
    final_metrics = {}
    latest_evaluation = {}
    best_eval_metric = float("-inf")
    best_eval_summary = {}
    profile_totals = {
        "spatial_seconds": 0.0,
        "temporal_seconds": 0.0,
        "nufft_seconds": 0.0,
        "backward_seconds": 0.0,
        "step_seconds": 0.0,
        "profiled_steps": 0,
    }

    with torch.autograd.set_detect_anomaly(detect_anomaly):
        for epoch in range(num_epochs):
            model.train()
            if log_peak_memory:
                reset_peak_memory_stats(device)

            epoch_sums = {
                "loss_total": 0.0,
                "loss_data": 0.0,
                "loss_residual": 0.0,
                "loss_tau": 0.0,
                "loss_temporal": 0.0,
                "grad_norm": 0.0,
            }
            num_batches = 0
            weights = _loss_weights(config, epoch)
            frame_batches = _make_epoch_frame_batches(bundle.n_frames, frame_batch_size, shuffle_frames)

            for frame_indices_cpu in frame_batches:
                num_batches += 1
                global_step += 1
                profile_this_step = profile_totals["profiled_steps"] < profile_first_steps
                if profile_this_step:
                    _sync_if_needed(device)
                    step_start = time.perf_counter()

                frame_indices_device = frame_indices_cpu.to(device=device)
                if preload_kspace_to_device:
                    kspace_target = kspace_storage.index_select(0, frame_indices_device)
                else:
                    kspace_target = bundle.kspace.index_select(0, frame_indices_cpu).to(
                        device=device,
                        dtype=torch.complex64,
                        non_blocking=True,
                    )
                t_norm_batch = full_time_coords.index_select(0, frame_indices_device)

                optimizer.zero_grad(set_to_none=zero_grad_set_to_none)

                if profile_this_step:
                    _sync_if_needed(device)
                    spatial_start = time.perf_counter()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    spatial_components = model.evaluate_spatial_components(spatial_coords)
                spatial_components = _spatial_components_to_float32(spatial_components)
                if profile_this_step:
                    _sync_if_needed(device)
                    profile_totals["spatial_seconds"] += time.perf_counter() - spatial_start

                if profile_this_step:
                    _sync_if_needed(device)
                    temporal_start = time.perf_counter()
                basis_batch, basis_derivative_batch = model.evaluate_temporal_basis(
                    t_norm_batch,
                    need_derivative=model.use_delay,
                )
                prediction_ri = model.synthesize_batch(
                    init_image_flat,
                    spatial_components,
                    basis_batch.float(),
                    bundle.spatial_shape,
                    basis_derivative=basis_derivative_batch.float() if basis_derivative_batch is not None else None,
                )
                prediction_complex = real_imag_to_complex(prediction_ri).to(torch.complex64).contiguous()
                if profile_this_step:
                    _sync_if_needed(device)
                    profile_totals["temporal_seconds"] += time.perf_counter() - temporal_start

                if profile_this_step:
                    _sync_if_needed(device)
                    nufft_start = time.perf_counter()
                kspace_prediction = nufft_backend.forward(prediction_complex, frame_indices_cpu)
                loss_data = weights["data"] * complex_mse(kspace_prediction, kspace_target)
                if profile_this_step:
                    _sync_if_needed(device)
                    profile_totals["nufft_seconds"] += time.perf_counter() - nufft_start

                residual_enabled = (
                    model.use_residual
                    and spatial_components.residual is not None
                    and bool(losses_cfg.get("residual_energy", {}).get("enabled", True))
                )
                if residual_enabled:
                    loss_residual = residual_energy_loss(spatial_components.residual)
                else:
                    loss_residual = torch.zeros((), device=device)

                tau_enabled = (
                    model.use_delay
                    and bool(losses_cfg.get("tau_smooth", {}).get("enabled", True))
                    and spatial_components.tau is not None
                )
                if tau_enabled:
                    loss_tau = tau_smoothness_loss(spatial_components.tau, bundle.spatial_shape)
                else:
                    loss_tau = torch.zeros((), device=device)

                temporal_smooth_enabled = bool(temporal_reg_cfg.get("enabled", True))
                if temporal_smooth_enabled:
                    if temporal_reg_reuse_batch and basis_derivative_batch is not None:
                        temporal_derivative = basis_derivative_batch
                    else:
                        _, temporal_derivative = model.evaluate_temporal_basis(temporal_reg_coords, need_derivative=True)
                    if temporal_derivative is None:
                        raise RuntimeError("Temporal derivative is required for temporal basis smoothness loss.")
                    loss_temporal = temporal_basis_smoothness_loss(temporal_derivative.float())
                else:
                    loss_temporal = torch.zeros((), device=device)

                total_loss = loss_data + weights["residual"] * loss_residual
                if tau_enabled:
                    total_loss = total_loss + weights["tau"] * loss_tau
                if temporal_smooth_enabled:
                    total_loss = total_loss + weights["temporal"] * loss_temporal

                if profile_this_step:
                    _sync_if_needed(device)
                    backward_start = time.perf_counter()
                if use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.unscale_(optimizer)
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    grad_norm = compute_grad_norm(model.parameters())
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    grad_norm = compute_grad_norm(model.parameters())
                    optimizer.step()
                if profile_this_step:
                    _sync_if_needed(device)
                    profile_totals["backward_seconds"] += time.perf_counter() - backward_start
                    profile_totals["step_seconds"] += time.perf_counter() - step_start
                    profile_totals["profiled_steps"] += 1

                epoch_sums["loss_total"] += safe_item(total_loss)
                epoch_sums["loss_data"] += safe_item(loss_data)
                epoch_sums["loss_residual"] += safe_item(loss_residual)
                epoch_sums["loss_tau"] += safe_item(loss_tau)
                epoch_sums["loss_temporal"] += safe_item(loss_temporal)
                epoch_sums["grad_norm"] += grad_norm

            if scheduler is not None:
                scheduler.step()

            averages = {key: value / max(num_batches, 1) for key, value in epoch_sums.items()}
            learning_rate = optimizer.param_groups[0]["lr"]
            memory_stats = get_peak_memory_stats(device) if log_peak_memory else {}
            final_metrics = {
                "epoch": epoch,
                "learning_rate": learning_rate,
                "loss_total": averages["loss_total"],
                "loss_data": averages["loss_data"],
                "loss_residual": averages["loss_residual"],
                "loss_tau": averages["loss_tau"],
                "loss_temporal": averages["loss_temporal"],
                "grad_norm": averages["grad_norm"],
            }
            if log_peak_memory:
                final_metrics.update(memory_stats)

            writer.add_scalar("train/total_loss", averages["loss_total"], epoch)
            writer.add_scalar("train/data_loss", averages["loss_data"], epoch)
            writer.add_scalar("train/residual_loss", averages["loss_residual"], epoch)
            writer.add_scalar("train/tau_smooth_loss", averages["loss_tau"], epoch)
            writer.add_scalar("train/temporal_smooth_loss", averages["loss_temporal"], epoch)
            writer.add_scalar("train/grad_norm", averages["grad_norm"], epoch)
            writer.add_scalar("train/lr", learning_rate, epoch)
            writer.add_scalar("weights/residual", weights["residual"], epoch)
            writer.add_scalar("weights/tau", weights["tau"], epoch)
            writer.add_scalar("weights/temporal", weights["temporal"], epoch)
            if use_amp:
                writer.add_scalar("train/amp_scale", float(scaler.get_scale()), epoch)
            if log_peak_memory:
                writer.add_scalar("memory/peak_allocated_gb", memory_stats["peak_allocated_gb"], epoch)
                writer.add_scalar("memory/peak_reserved_gb", memory_stats["peak_reserved_gb"], epoch)
            if profile_totals["profiled_steps"] > 0:
                denom = float(profile_totals["profiled_steps"])
                writer.add_scalar("profile/spatial_ms", 1000.0 * profile_totals["spatial_seconds"] / denom, epoch)
                writer.add_scalar("profile/temporal_ms", 1000.0 * profile_totals["temporal_seconds"] / denom, epoch)
                writer.add_scalar("profile/nufft_ms", 1000.0 * profile_totals["nufft_seconds"] / denom, epoch)
                writer.add_scalar("profile/backward_ms", 1000.0 * profile_totals["backward_seconds"] / denom, epoch)
                writer.add_scalar("profile/step_ms", 1000.0 * profile_totals["step_seconds"] / denom, epoch)

            maybe_log_histograms(writer, model, epoch, hist_interval)

            should_save = ((epoch + 1) % max(save_interval, 1) == 0) or epoch == num_epochs - 1
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None if scheduler is None else scheduler.state_dict(),
                "epoch": epoch,
                "config": config,
                "nufft": nufft_backend.describe(),
                "startup_diagnostics": startup_diagnostics,
                "gradient_sanity_check": sanity_result,
                "profile_summary": profile_totals,
            }
            if should_save:
                save_checkpoint(
                    checkpoint,
                    experiment_dirs["checkpoint_dir"],
                    epoch=epoch,
                    keep_last_k=int(config.get("checkpointing", {}).get("keep_last_k", 5)),
                )

            should_snapshot = save_snapshot and (((epoch + 1) % max(save_snapshot_every, 1) == 0) or epoch == num_epochs - 1)
            if should_snapshot:
                _save_reconstruction_snapshot(
                    model=model,
                    bundle=bundle,
                    spatial_coords=spatial_coords,
                    init_image_flat=init_image_flat,
                    recon_dir=experiment_dirs["recon_dir"],
                    epoch=epoch,
                    frame_batch_size=frame_batch_size,
                    output_scale_factor=bundle.scale_factor,
                )

            should_evaluate = evaluation_enabled and (
                epoch == num_epochs - 1
                or (evaluation_interval > 0 and (epoch + 1) % evaluation_interval == 0)
            )
            if should_evaluate:
                reconstruction = _reconstruct_full_sequence(
                    model=model,
                    bundle=bundle,
                    spatial_coords=spatial_coords,
                    init_image_flat=init_image_flat,
                    frame_batch_size=frame_batch_size,
                    output_scale_factor=bundle.scale_factor,
                )
                reconstruction_path = None
                if save_reconstruction_on_eval:
                    reconstruction_path = experiment_dirs["evaluation_dir"] / f"epoch_{epoch:04d}" / "reconstruction.npy"
                    reconstruction_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(reconstruction_path, reconstruction)
                evaluation_output_dir = experiment_dirs["evaluation_dir"] / f"epoch_{epoch:04d}"
                latest_evaluation = evaluate_magnitude_reconstruction(
                    np.linalg.norm(reconstruction, axis=-1),
                    config=config,
                    project_root=PROJECT_ROOT,
                    output_dir=evaluation_output_dir,
                    method_name=f"epoch_{epoch:04d}",
                )
                writer.add_scalar("eval/f2_fixed", latest_evaluation["f2_fixed"], epoch)
                writer.add_scalar("eval/f2_max", latest_evaluation["f2_max"], epoch)
                writer.add_scalar("eval/roi_signal_corr", latest_evaluation["roi_signal_corr"], epoch)
                writer.add_scalar("eval/roi_tsnr", latest_evaluation["roi_tsnr"], epoch)
                current_metric = float(latest_evaluation.get(evaluation_primary_metric, float("-inf")))
                if current_metric > best_eval_metric:
                    best_eval_metric = current_metric
                    best_eval_summary = dict(latest_evaluation)
                    if save_best_checkpoint:
                        torch.save(checkpoint, experiment_dirs["checkpoint_dir"] / f"best_{evaluation_primary_metric}.pt")
                final_metrics[evaluation_primary_metric] = current_metric

            if epoch % max(log_interval, 1) == 0 or epoch == num_epochs - 1:
                message = (
                    f"[epoch {epoch:04d}] total={averages['loss_total']:.4e} "
                    f"data={averages['loss_data']:.4e} res={averages['loss_residual']:.4e} "
                    f"tau={averages['loss_tau']:.4e} g={averages['loss_temporal']:.4e} "
                    f"lr={learning_rate:.3e} grad={averages['grad_norm']:.3e}"
                )
                if evaluation_enabled and latest_evaluation:
                    message += (
                        f" f2={latest_evaluation['f2_fixed']:.4f}"
                        f" f2max={latest_evaluation['f2_max']:.4f}"
                    )
                if log_peak_memory:
                    message += (
                        f" peak_alloc={memory_stats['peak_allocated_gb']:.3f}GB"
                        f" peak_reserved={memory_stats['peak_reserved_gb']:.3f}GB"
                    )
                if profile_totals["profiled_steps"] > 0:
                    denom = float(profile_totals["profiled_steps"])
                    message += (
                        f" spatial_ms={1000.0 * profile_totals['spatial_seconds'] / denom:.2f}"
                        f" temporal_ms={1000.0 * profile_totals['temporal_seconds'] / denom:.2f}"
                        f" nufft_ms={1000.0 * profile_totals['nufft_seconds'] / denom:.2f}"
                        f" backward_ms={1000.0 * profile_totals['backward_seconds'] / denom:.2f}"
                        f" step_ms={1000.0 * profile_totals['step_seconds'] / denom:.2f}"
                    )
                print(message)

    profile_summary = dict(profile_totals)
    if profile_summary["profiled_steps"] > 0:
        denom = float(profile_summary["profiled_steps"])
        profile_summary["spatial_ms"] = 1000.0 * profile_summary["spatial_seconds"] / denom
        profile_summary["temporal_ms"] = 1000.0 * profile_summary["temporal_seconds"] / denom
        profile_summary["nufft_ms"] = 1000.0 * profile_summary["nufft_seconds"] / denom
        profile_summary["backward_ms"] = 1000.0 * profile_summary["backward_seconds"] / denom
        profile_summary["step_ms"] = 1000.0 * profile_summary["step_seconds"] / denom
        profile_summary["cpu_overhead_ms"] = max(
            0.0,
            profile_summary["step_ms"]
            - profile_summary["spatial_ms"]
            - profile_summary["temporal_ms"]
            - profile_summary["nufft_ms"]
            - profile_summary["backward_ms"],
        )
    profile_path = experiment_dirs["run_dir"] / "profile_summary.json"
    with profile_path.open("w", encoding="utf-8") as handle:
        json.dump(profile_summary, handle, indent=2)

    summary = {
        "run_dir": str(experiment_dirs["run_dir"]),
        "checkpoint_dir": str(experiment_dirs["checkpoint_dir"]),
        "log_dir": str(experiment_dirs["log_dir"]),
        "recon_dir": str(experiment_dirs["recon_dir"]),
        "evaluation_dir": str(experiment_dirs["evaluation_dir"]),
        "diagnostics_path": None if diagnostics_path is None else str(diagnostics_path),
        "gradient_sanity_check_path": None if sanity_path is None else str(sanity_path),
        "profile_summary_path": str(profile_path),
        "nufft": nufft_backend.describe(),
        "startup_diagnostics": startup_diagnostics,
        "gradient_sanity_check": sanity_result,
        "profile_summary": profile_summary,
        "final_metrics": final_metrics,
        "latest_evaluation": latest_evaluation,
        "best_evaluation": best_eval_summary,
        "device": str(device),
    }
    summary_path = experiment_dirs["run_dir"] / "metrics.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    writer.close()
    return summary


def train_subspace_inr_3d_fmri(config: Dict):
    bundle = load_fmri_data_bundle(config, PROJECT_ROOT)
    return _train_impl(config, bundle)



def train_subspace_inr_from_arrays(
    kspaces: np.ndarray,
    trajs: np.ndarray,
    back_img: np.ndarray,
    config: Dict,
    case_name: Optional[str] = None,
    csm: Optional[np.ndarray] = None,
):
    runtime_config = deepcopy(config)
    if case_name is not None:
        experiment_cfg = runtime_config.setdefault("experiment", {})
        previous_case_name = experiment_cfg.get("case_name")
        previous_run_name = experiment_cfg.get("run_name")
        experiment_cfg["case_name"] = case_name
        default_previous_run_name = (
            None
            if previous_case_name is None
            else f"{experiment_cfg.get('name', 'subspaceINR4fMRI')}_{previous_case_name}"
        )
        if previous_run_name in (None, default_previous_run_name):
            experiment_cfg["run_name"] = f"{experiment_cfg.get('name', 'subspaceINR4fMRI')}_{case_name}"
    bundle = build_data_bundle_from_arrays(
        kspace=kspaces,
        traj=trajs,
        background_init=back_img,
        csm=csm,
        img_size=runtime_config["data"]["img_size"],
        scale_factor=float(runtime_config["data"].get("scale_factor", 1.0)),
    )
    return _train_impl(runtime_config, bundle)
