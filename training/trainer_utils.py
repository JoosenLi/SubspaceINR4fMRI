from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch



def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_compute_device(config: Dict[str, Any]) -> Tuple[torch.device, int]:
    compute_cfg = config.get("compute", {})
    requested_device = str(compute_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")).lower()
    gpu_index = int(compute_cfg.get("gpu_index", 0))

    if requested_device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        return torch.device(f"cuda:{gpu_index}"), gpu_index

    return torch.device("cpu"), -1


def prepare_nufft_config(config: Dict[str, Any], gpu_index: int) -> Dict[str, Any]:
    nufft_cfg = dict(config["nufft"])
    backend_kwargs = dict(nufft_cfg.get("backend_kwargs", {}))
    if gpu_index >= 0:
        backend_kwargs.setdefault("gpu_device_id", gpu_index)
    nufft_cfg["backend_kwargs"] = backend_kwargs
    return nufft_cfg



def build_optimizer(model: torch.nn.Module, config: Dict) -> torch.optim.Optimizer:
    optimizer_cfg = config["optimizer"]
    optimizer_type = str(optimizer_cfg.get("type", "Adam")).lower()
    if optimizer_type != "adam":
        raise ValueError(f"Unsupported optimizer type: {optimizer_cfg.get('type')}")
    return torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 2.0e-4)),
        betas=(float(optimizer_cfg.get("beta1", 0.9)), float(optimizer_cfg.get("beta2", 0.99))),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )



def warmup_lambda(base_value: float, epoch: int, warmup_epochs: int) -> float:
    if base_value <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return base_value
    factor = min(1.0, float(epoch + 1) / float(warmup_epochs))
    return base_value * factor



def prepare_run_directories(config: Dict, project_root: Path) -> Dict[str, Path]:
    experiment_cfg = config["experiment"]
    output_root = Path(experiment_cfg.get("output_root", "./outputs"))
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    run_name = experiment_cfg.get("run_name") or f"{experiment_cfg['name']}_{experiment_cfg['case_name']}"
    run_dir = output_root / run_name
    log_dir = run_dir / "logs"
    checkpoint_dir = run_dir / "checkpoints"
    recon_dir = run_dir / "reconstructions"
    benchmark_dir = run_dir / "benchmark"
    evaluation_dir = run_dir / "evaluation"

    for path in [run_dir, log_dir, checkpoint_dir, recon_dir, benchmark_dir, evaluation_dir]:
        path.mkdir(parents=True, exist_ok=True)

    return {
        "output_root": output_root,
        "run_dir": run_dir,
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
        "recon_dir": recon_dir,
        "benchmark_dir": benchmark_dir,
        "evaluation_dir": evaluation_dir,
    }



def maybe_log_histograms(writer, model: torch.nn.Module, step: int, every: int) -> None:
    if every <= 0 or step % every != 0:
        return
    for name, parameter in model.named_parameters():
        try:
            writer.add_histogram(f"param/{name}", parameter.detach().cpu().numpy(), step)
            if parameter.grad is not None:
                writer.add_histogram(f"grad/{name}", parameter.grad.detach().cpu().numpy(), step)
        except Exception:
            continue
