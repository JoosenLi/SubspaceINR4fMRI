from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from subspaceINR4fMRI.utils import load_yaml_config


TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_subspace_inr_3d_fmri.py"
EVAL_SCRIPT = PROJECT_ROOT / "scripts" / "evaluate_reconstruction_f2.py"
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_nufft_backends.py"
PROFILE_SCRIPT = PROJECT_ROOT / "scripts" / "profile_training_efficiency.py"


@dataclass
class Trial:
    round_name: str
    trial_name: str
    overrides: List[str]
    epochs: int


CAPACITY_PRESETS: Dict[str, List[str]] = {
    "cap_base": [],
    "cap_hash_large": [
        "spatial_encoder.num_levels=20",
        "spatial_encoder.level_dim=2",
        "spatial_encoder.log2_hashmap_size=21",
    ],
    "cap_hash_wide": [
        "spatial_encoder.num_levels=16",
        "spatial_encoder.level_dim=4",
        "spatial_encoder.log2_hashmap_size=21",
    ],
    "cap_heads_large": [
        "residual_head.hidden_dim=128",
        "coefficient_head.hidden_dim=128",
        "delay_head.hidden_dim=128",
        "residual_head.num_hidden_layers=2",
        "coefficient_head.num_hidden_layers=2",
        "delay_head.num_hidden_layers=2",
    ],
    "cap_temporal_large": [
        "temporal_branch.fourier_num_frequencies=16",
        "temporal_branch.hidden_dim=128",
        "temporal_branch.num_hidden_layers=2",
    ],
    "cap_all_large": [
        "spatial_encoder.num_levels=20",
        "spatial_encoder.level_dim=2",
        "spatial_encoder.log2_hashmap_size=21",
        "residual_head.hidden_dim=128",
        "coefficient_head.hidden_dim=128",
        "delay_head.hidden_dim=128",
        "residual_head.num_hidden_layers=2",
        "coefficient_head.num_hidden_layers=2",
        "delay_head.num_hidden_layers=2",
        "temporal_branch.fourier_num_frequencies=20",
        "temporal_branch.hidden_dim=128",
        "temporal_branch.num_hidden_layers=3",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the staged two-GPU subspaceINR hyperparameter search.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "search_subspaceINR4fMRI.yaml"),
        help="Path to the search YAML config.",
    )
    parser.add_argument(
        "--round",
        type=str,
        default="all",
        choices=["all", "round0", "round1", "round2", "round3", "round4", "round5"],
        help="Run a single round or the full staged search.",
    )
    return parser.parse_args()


def _search_root(config: Dict[str, Any]) -> Path:
    root = Path(config["search"].get("output_root", PROJECT_ROOT / "outputs" / "search_subspaceINR4fMRI"))
    if not root.is_absolute():
        root = (PROJECT_ROOT / root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _slugify(text: str) -> str:
    allowed = []
    for char in text:
        if char.isalnum() or char in {"-", "_"}:
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed)


def _trial_run_name(round_name: str, trial_name: str) -> str:
    return _slugify(f"search_{round_name}_{trial_name}")


def _result_key(record: Dict[str, Any], key: str, default: float = float("-inf")) -> float:
    if key in record:
        return float(record[key])
    if "best_evaluation" in record and key in record["best_evaluation"]:
        return float(record["best_evaluation"][key])
    if "latest_evaluation" in record and key in record["latest_evaluation"]:
        return float(record["latest_evaluation"][key])
    if "final_metrics" in record and key in record["final_metrics"]:
        return float(record["final_metrics"][key])
    return default


def _load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for record in records for key in record.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _flatten_summary(round_name: str, trial: Trial, metrics: Dict[str, Any]) -> Dict[str, Any]:
    record = {
        "round": round_name,
        "trial_name": trial.trial_name,
        "run_dir": metrics.get("run_dir"),
        "device": metrics.get("device"),
        "f2_fixed": _result_key(metrics, "f2_fixed"),
        "f2_max": _result_key(metrics, "f2_max"),
        "roi_signal_corr": _result_key(metrics, "roi_signal_corr"),
        "roi_tsnr": _result_key(metrics, "roi_tsnr"),
        "loss_data": _result_key(metrics, "loss_data"),
        "step_ms": metrics.get("profile_summary", {}).get("step_ms", float("inf")),
    }
    for override in trial.overrides:
        if "=" in override:
            key, value = override.split("=", 1)
            record[key] = value
    return record


def _build_train_command(base_config_path: str, device: int, trial: Trial, search_root: Path) -> List[str]:
    output_root = search_root / trial.round_name
    return [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        base_config_path,
        "--device",
        f"cuda:{device}",
        "--override",
        f"experiment.output_root={str(output_root)}",
        "--override",
        f"experiment.run_name={_trial_run_name(trial.round_name, trial.trial_name)}",
        "--override",
        f"training.num_epochs={trial.epochs}",
        "--override",
        "training.gradient_sanity_check=false",
        "--override",
        "training.startup_diagnostics=false",
        "--override",
        "training.profile_first_steps=0",
        "--override",
        "logging.tensorboard=false",
        "--override",
        "logging.save_reconstruction_snapshot=false",
        "--override",
        "evaluation.enabled=true",
        "--override",
        f"evaluation.every_n_epochs={trial.epochs}",
        "--override",
        "evaluation.save_best_checkpoint=true",
    ] + [item for override in trial.overrides for item in ("--override", override)]


def _run_trials(
    base_config_path: str,
    search_root: Path,
    trials: Sequence[Trial],
    gpus: Sequence[int],
    resume: bool,
    max_parallel_trials: int | None = None,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    pending = list(trials)
    active: Dict[int, Dict[str, Any]] = {}
    parallel_limit = len(gpus) if max_parallel_trials is None else max(1, min(int(max_parallel_trials), len(gpus)))

    while pending or active:
        while pending and len(active) < parallel_limit:
            gpu = next(g for g in gpus if g not in active)
            trial = pending.pop(0)
            run_name = _trial_run_name(trial.round_name, trial.trial_name)
            run_dir = search_root / trial.round_name / run_name
            metrics_path = run_dir / "metrics.json"
            if resume and metrics_path.exists():
                metrics = _load_metrics(metrics_path)
                records.append(_flatten_summary(trial.round_name, trial, metrics))
                continue

            log_dir = search_root / trial.round_name / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = log_dir / f"{run_name}.log"
            stdout_handle = stdout_path.open("w", encoding="utf-8")
            command = _build_train_command(base_config_path, gpu, trial, search_root)
            process = subprocess.Popen(command, stdout=stdout_handle, stderr=subprocess.STDOUT, text=True)
            active[gpu] = {
                "trial": trial,
                "process": process,
                "stdout": stdout_handle,
                "run_dir": run_dir,
            }
            print(f"Launched {trial.round_name}/{trial.trial_name} on GPU{gpu}")

        finished_gpus = []
        for gpu, info in active.items():
            process = info["process"]
            if process.poll() is None:
                continue
            info["stdout"].close()
            trial = info["trial"]
            run_dir = info["run_dir"]
            metrics_path = run_dir / "metrics.json"
            if process.returncode != 0 or not metrics_path.exists():
                print(f"Trial failed: {trial.round_name}/{trial.trial_name} on GPU{gpu}")
            else:
                metrics = _load_metrics(metrics_path)
                records.append(_flatten_summary(trial.round_name, trial, metrics))
                print(f"Completed {trial.round_name}/{trial.trial_name} on GPU{gpu}")
            finished_gpus.append(gpu)
        for gpu in finished_gpus:
            active.pop(gpu)
        if active:
            time.sleep(2.0)
    return records


def _update_global_search_summary(search_root: Path) -> None:
    round_names = ["round1", "round2", "round3", "round4a", "round4b", "round4c", "round4d", "round5"]
    all_records: List[Dict[str, Any]] = []
    for round_name in round_names:
        all_records.extend(_read_round_records(search_root, round_name))
    _write_json(search_root / "search_results.json", all_records)
    _write_csv(search_root / "search_results.csv", all_records)


def _select_top(records: Sequence[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    sorted_records = sorted(
        records,
        key=lambda item: (
            -float(item.get("f2_fixed", float("-inf"))),
            -float(item.get("f2_max", float("-inf"))),
            float(item.get("loss_data", float("inf"))),
        ),
    )
    return sorted_records[:k]


def _run_baselines(config: Dict[str, Any], search_root: Path) -> Dict[str, Any]:
    base_config_path = str(Path(config["search"]["base_config"]).expanduser().resolve())
    baseline_dir = search_root / "round0" / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    base_config = load_yaml_config(base_config_path)
    dataset_root = PROJECT_ROOT / "dataset" / "sim3DT" / base_config["experiment"].get("case_name", "Simulation_Mar16_RotatedSpiral")
    baselines = {
        "Global_CG": dataset_root / "Global_CG.npy",
        "NUFFT_recon": dataset_root / "NUFFT_recon.npy",
    }
    existing_run = config["search"].get("existing_subspace_run")
    if existing_run:
        baselines["existing_subspace_recon"] = Path(existing_run)

    results = {}
    for name, path in baselines.items():
        if not path.exists():
            continue
        output_dir = baseline_dir / name
        command = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--config",
            base_config_path,
            "--reconstruction",
            str(path),
            "--input-is-unscaled",
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(command, check=True)
        results[name] = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    _write_json(baseline_dir / "baseline_metrics.json", results)
    return results


def _run_round0(config: Dict[str, Any], search_root: Path) -> Dict[str, Any]:
    base_config_path = str(Path(config["search"]["base_config"]).expanduser().resolve())
    round_root = search_root / "round0"
    round_root.mkdir(parents=True, exist_ok=True)

    profile_command = [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--config",
        base_config_path,
        "--epochs",
        str(int(config["search"].get("round0_profile_epochs", 4))),
        "--output",
        str(round_root / "efficiency" / "profile.json"),
    ]
    subprocess.run(profile_command, check=True)

    benchmark_command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--config",
        str(PROJECT_ROOT / "configs" / "benchmark_nufft.yaml"),
        "--backend",
        "both",
        "--mode",
        "both",
    ]
    subprocess.run(benchmark_command, check=True)

    baselines = _run_baselines(config, search_root)
    return {
        "profile_path": str(round_root / "efficiency" / "profile.json"),
        "baseline_metrics": baselines,
    }


def _build_round1_trials() -> List[Trial]:
    trials = []
    for backend, batch_size, lr in product(["cufinufft", "gpunufft"], [16, 24, 32, 48], [None]):
        lr_value = {16: 2e-4, 24: 3e-4, 32: 4e-4, 48: 6e-4}[batch_size]
        trials.append(
            Trial(
                round_name="round1",
                trial_name=f"{backend}_bs{batch_size}_lr{lr_value:.0e}",
                epochs=20,
                overrides=[
                    f"nufft.backend={backend}",
                    "nufft.preload_per_frame_operators=true",
                    "nufft.reuse_single_operator=false",
                    f"training.frame_batch_size={batch_size}",
                    f"optimizer.lr={lr_value}",
                    "training.amp=false",
                    "model.use_delay=true",
                    "model.use_residual=false",
                    "model.rank=28",
                    "coefficient_head.output_dim=28",
                    "temporal_branch.output_dim=56",
                    "data.scale_factor=500.0",
                ],
            )
        )
    return trials


def _filter_round1(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    viable = [record for record in records if record.get("f2_fixed", float("-inf")) > float("-inf")]
    if not viable:
        return []
    best_f2 = max(float(record["f2_fixed"]) for record in viable)
    threshold = best_f2 * 0.98
    candidates = [record for record in viable if float(record["f2_fixed"]) >= threshold]
    return sorted(candidates, key=lambda item: float(item.get("step_ms", float("inf"))))[:1]


def _build_round2_trials(system_record: Dict[str, Any]) -> List[Trial]:
    backend = system_record.get("nufft.backend", "cufinufft")
    batch_size = int(float(system_record.get("training.frame_batch_size", 16)))
    lr = float(system_record.get("optimizer.lr", 2e-4))
    trials = []
    for scale, rank, use_delay, use_residual in product([50, 100, 200, 500, 1000, 2000], [8, 18, 28], [False, True], [False, True]):
        trial_name = f"scale{scale}_rank{rank}_delay{int(use_delay)}_res{int(use_residual)}"
        trials.append(
            Trial(
                round_name="round2",
                trial_name=trial_name,
                epochs=40,
                overrides=[
                    f"nufft.backend={backend}",
                    f"training.frame_batch_size={batch_size}",
                    f"optimizer.lr={lr}",
                    f"data.scale_factor={float(scale)}",
                    f"model.rank={rank}",
                    f"coefficient_head.output_dim={rank}",
                    f"temporal_branch.output_dim={2 * rank}",
                    f"model.use_delay={str(use_delay).lower()}",
                    f"delay_head.enabled={str(use_delay).lower()}",
                    f"model.use_residual={str(use_residual).lower()}",
                    f"losses.residual_energy.enabled={str(use_residual).lower()}",
                ],
            )
        )
    return trials


def _build_round3_trials(top_round2: Sequence[Dict[str, Any]]) -> List[Trial]:
    trials = []
    for record in top_round2:
        inherited = [
            f"nufft.backend={record.get('nufft.backend', 'cufinufft')}",
            f"training.frame_batch_size={record.get('training.frame_batch_size', 16)}",
            f"optimizer.lr={record.get('optimizer.lr', 2e-4)}",
            f"data.scale_factor={record.get('data.scale_factor', 500)}",
            f"model.rank={record.get('model.rank', 28)}",
            f"coefficient_head.output_dim={record.get('model.rank', 28)}",
            f"temporal_branch.output_dim={2 * int(float(record.get('model.rank', 28)))}",
            f"model.use_delay={record.get('model.use_delay', 'true')}",
            f"delay_head.enabled={record.get('model.use_delay', 'true')}",
            f"model.use_residual={record.get('model.use_residual', 'false')}",
            f"losses.residual_energy.enabled={record.get('model.use_residual', 'false')}",
        ]
        base_name = record["trial_name"]
        for preset_name, preset_overrides in CAPACITY_PRESETS.items():
            trials.append(
                Trial(
                    round_name="round3",
                    trial_name=f"{base_name}_{preset_name}",
                    epochs=60,
                    overrides=inherited + preset_overrides,
                )
            )
    return trials


def _build_lr_trials(top_round3: Sequence[Dict[str, Any]]) -> List[Trial]:
    trials = []
    for record in top_round3:
        base_overrides = [f"{key}={value}" for key, value in record.items() if key in {
            'nufft.backend', 'training.frame_batch_size', 'data.scale_factor', 'model.rank', 'model.use_delay', 'model.use_residual',
            'spatial_encoder.num_levels', 'spatial_encoder.level_dim', 'spatial_encoder.log2_hashmap_size',
            'residual_head.hidden_dim', 'coefficient_head.hidden_dim', 'delay_head.hidden_dim',
            'residual_head.num_hidden_layers', 'coefficient_head.num_hidden_layers', 'delay_head.num_hidden_layers',
            'temporal_branch.fourier_num_frequencies', 'temporal_branch.hidden_dim', 'temporal_branch.num_hidden_layers'
        }]
        base_overrides.extend([
            f"coefficient_head.output_dim={record.get('model.rank', 28)}",
            f"temporal_branch.output_dim={2 * int(float(record.get('model.rank', 28)))}",
            f"delay_head.enabled={record.get('model.use_delay', 'true')}",
            f"losses.residual_energy.enabled={record.get('model.use_residual', 'false')}",
        ])
        for lr in [1e-4, 2e-4, 4e-4]:
            trials.append(
                Trial(
                    round_name="round4a",
                    trial_name=f"{record['trial_name']}_lr{lr:.0e}",
                    epochs=60,
                    overrides=base_overrides + [f"optimizer.lr={lr}"],
                )
            )
    return trials


def _build_lambda_trials(top_records: Sequence[Dict[str, Any]], round_name: str, key: str, values: Sequence[float]) -> List[Trial]:
    trials = []
    for record in top_records:
        base_overrides = [f"{k}={v}" for k, v in record.items() if k in {
            'nufft.backend', 'training.frame_batch_size', 'data.scale_factor', 'model.rank', 'model.use_delay', 'model.use_residual',
            'spatial_encoder.num_levels', 'spatial_encoder.level_dim', 'spatial_encoder.log2_hashmap_size',
            'residual_head.hidden_dim', 'coefficient_head.hidden_dim', 'delay_head.hidden_dim',
            'residual_head.num_hidden_layers', 'coefficient_head.num_hidden_layers', 'delay_head.num_hidden_layers',
            'temporal_branch.fourier_num_frequencies', 'temporal_branch.hidden_dim', 'temporal_branch.num_hidden_layers',
            'optimizer.lr'
        }]
        base_overrides.extend([
            f"coefficient_head.output_dim={record.get('model.rank', 28)}",
            f"temporal_branch.output_dim={2 * int(float(record.get('model.rank', 28)))}",
            f"delay_head.enabled={record.get('model.use_delay', 'true')}",
            f"losses.residual_energy.enabled={record.get('model.use_residual', 'false')}",
        ])
        for value in values:
            trials.append(
                Trial(
                    round_name=round_name,
                    trial_name=f"{record['trial_name']}_{key.split('.')[-1]}{value:.0e}",
                    epochs=60,
                    overrides=base_overrides + [f"{key}={value}"],
                )
            )
    return trials


def _build_round5_trials(top_records: Sequence[Dict[str, Any]]) -> List[Trial]:
    trials = []
    seeds = [42, 123, 2026]
    for index, record in enumerate(top_records):
        for seed in (seeds if index == 0 else [42]):
            base_overrides = [f"{k}={v}" for k, v in record.items() if k not in {
                'round', 'trial_name', 'run_dir', 'device', 'f2_fixed', 'f2_max', 'roi_signal_corr', 'roi_tsnr', 'loss_data', 'step_ms'
            }]
            base_overrides.extend([
                f"training.frame_batch_size={record.get('training.frame_batch_size', 16)}",
                "evaluation.enabled=true",
                "evaluation.every_n_epochs=25",
                "evaluation.save_best_checkpoint=true",
                "evaluation.save_reconstruction_on_eval=true",
                "logging.save_reconstruction_snapshot=false",
                f"experiment.seed={seed}",
            ])
            trials.append(
                Trial(
                    round_name="round5",
                    trial_name=f"{record['trial_name']}_seed{seed}",
                    epochs=300,
                    overrides=base_overrides,
                )
            )
    return trials


def _read_round_records(search_root: Path, round_name: str) -> List[Dict[str, Any]]:
    summary_path = search_root / round_name / f"{round_name}_summary.json"
    if not summary_path.exists():
        return []
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _save_round_summary(search_root: Path, round_name: str, records: Sequence[Dict[str, Any]]) -> None:
    round_root = search_root / round_name
    _write_json(round_root / f"{round_name}_summary.json", list(records))
    _write_csv(round_root / f"{round_name}_summary.csv", list(records))


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    search_root = _search_root(config)
    base_config_path = str(Path(config["search"]["base_config"]).expanduser().resolve())
    gpus = [int(value) for value in config["search"].get("gpus", [0, 1])]
    resume = bool(config["search"].get("resume", True))
    max_parallel_trials = int(config["search"].get("max_parallel_trials", len(gpus)))

    if args.round in {"all", "round0"}:
        round0_result = _run_round0(config, search_root)
        _write_json(search_root / "round0" / "round0_summary.json", round0_result)
        if args.round == "round0":
            return

    if args.round in {"all", "round1"}:
        round1_records = _run_trials(base_config_path, search_root, _build_round1_trials(), gpus, resume, max_parallel_trials)
        _save_round_summary(search_root, "round1", round1_records)
        _update_global_search_summary(search_root)
        if args.round == "round1":
            return
    else:
        round1_records = _read_round_records(search_root, "round1")

    system_default = _filter_round1(round1_records)
    if not system_default:
        raise SystemExit("Round 1 did not produce a viable system configuration.")

    if args.round in {"all", "round2"}:
        round2_trials = _build_round2_trials(system_default[0])
        round2_records = _run_trials(base_config_path, search_root, round2_trials, gpus, resume, max_parallel_trials)
        round2_top = _select_top(round2_records, 10)
        _save_round_summary(search_root, "round2", round2_records)
        _write_json(search_root / "round2" / "round2_top10.json", round2_top)
        _update_global_search_summary(search_root)
        if args.round == "round2":
            return
    else:
        round2_records = _read_round_records(search_root, "round2")
        round2_top = _select_top(round2_records, 10)

    if args.round in {"all", "round3"}:
        round3_trials = _build_round3_trials(round2_top)
        round3_records = _run_trials(base_config_path, search_root, round3_trials, gpus, resume, max_parallel_trials)
        round3_top = _select_top(round3_records, 6)
        _save_round_summary(search_root, "round3", round3_records)
        _write_json(search_root / "round3" / "round3_top6.json", round3_top)
        _update_global_search_summary(search_root)
        if args.round == "round3":
            return
    else:
        round3_records = _read_round_records(search_root, "round3")
        round3_top = _select_top(round3_records, 6)

    if args.round in {"all", "round4"}:
        round4a_records = _run_trials(base_config_path, search_root, _build_lr_trials(round3_top), gpus, resume, max_parallel_trials)
        _save_round_summary(search_root, "round4a", round4a_records)
        _update_global_search_summary(search_root)
        round4a_top = _select_top(round4a_records, 3)
        _write_json(search_root / "round4a" / "round4a_top3.json", round4a_top)

        round4b_records = _run_trials(base_config_path, search_root, _build_lambda_trials(round4a_top, "round4b", "losses.temporal_basis_smooth.lambda", [1e-6, 1e-5, 1e-4]), gpus, resume, max_parallel_trials)
        _save_round_summary(search_root, "round4b", round4b_records)
        _update_global_search_summary(search_root)
        round4b_top = _select_top(round4b_records, 3)

        delay_candidates = [record for record in round4b_top if str(record.get("model.use_delay", "true")).lower() == "true"]
        if delay_candidates:
            round4c_records = _run_trials(base_config_path, search_root, _build_lambda_trials(delay_candidates, "round4c", "losses.tau_smooth.lambda", [0.0, 1e-6, 1e-5, 1e-4]), gpus, resume, max_parallel_trials)
            _save_round_summary(search_root, "round4c", round4c_records)
            _update_global_search_summary(search_root)
            round4c_top = _select_top(round4c_records, 3)
        else:
            round4c_top = round4b_top

        residual_candidates = [record for record in round4c_top if str(record.get("model.use_residual", "false")).lower() == "true"]
        if residual_candidates:
            round4d_records = _run_trials(base_config_path, search_root, _build_lambda_trials(residual_candidates, "round4d", "losses.residual_energy.lambda", [0.0, 1e-6, 1e-5, 1e-4]), gpus, resume, max_parallel_trials)
            _save_round_summary(search_root, "round4d", round4d_records)
            _update_global_search_summary(search_root)
            final_top = _select_top(round4d_records, 3)
        else:
            final_top = round4c_top

        _write_json(search_root / "round4" / "round4_top3.json", final_top)
        if args.round == "round4":
            return
    else:
        final_top = json.loads((search_root / "round4" / "round4_top3.json").read_text(encoding="utf-8"))

    if args.round in {"all", "round5"}:
        round5_records = _run_trials(base_config_path, search_root, _build_round5_trials(final_top), gpus, resume, max_parallel_trials)
        _save_round_summary(search_root, "round5", round5_records)
        _update_global_search_summary(search_root)
        _write_json(search_root / "round5" / "best_final.json", _select_top(round5_records, 1))


if __name__ == "__main__":
    main()
