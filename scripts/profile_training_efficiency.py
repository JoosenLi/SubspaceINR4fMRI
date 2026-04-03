from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from subspaceINR4fMRI.training import train_subspace_inr_3d_fmri
from subspaceINR4fMRI.utils import apply_dotlist_overrides, load_yaml_config


class GPUSampler:
    def __init__(self, gpu_index: int, interval_seconds: float = 0.5):
        self.gpu_index = gpu_index
        self.interval_seconds = interval_seconds
        self.samples: List[Dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        query = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=timestamp,power.draw,utilization.gpu,utilization.memory,memory.used",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(query, text=True).strip()
                if output:
                    timestamp, power, util_gpu, util_mem, mem_used = [part.strip() for part in output.split(",")]
                    self.samples.append(
                        {
                            "timestamp": timestamp,
                            "power_draw_w": float(power),
                            "utilization_gpu_pct": float(util_gpu),
                            "utilization_memory_pct": float(util_mem),
                            "memory_used_mb": float(mem_used),
                        }
                    )
            except Exception:
                pass
            time.sleep(self.interval_seconds)

    def summary(self) -> Dict[str, float]:
        if not self.samples:
            return {}
        def _avg(key: str) -> float:
            return sum(sample[key] for sample in self.samples) / len(self.samples)
        def _max(key: str) -> float:
            return max(sample[key] for sample in self.samples)
        return {
            "num_samples": len(self.samples),
            "power_draw_w_mean": _avg("power_draw_w"),
            "power_draw_w_max": _max("power_draw_w"),
            "utilization_gpu_pct_mean": _avg("utilization_gpu_pct"),
            "utilization_gpu_pct_max": _max("utilization_gpu_pct"),
            "utilization_memory_pct_mean": _avg("utilization_memory_pct"),
            "memory_used_mb_max": _max("memory_used_mb"),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile steady-state HashINR training efficiency.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "subspaceINR4fMRI.yaml"))
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--profile-steps", type=int, default=6)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    overrides = [
        f"training.num_epochs={args.epochs}",
        f"training.profile_first_steps={args.profile_steps}",
        "training.gradient_sanity_check=false",
        "training.startup_diagnostics=false",
        "logging.tensorboard=false",
        "logging.save_reconstruction_snapshot=false",
        "evaluation.enabled=false",
    ]
    if args.device is not None:
        if args.device.startswith("cuda"):
            overrides.extend(["compute.device=cuda", f"compute.gpu_index={int(args.device.split(':', 1)[1]) if ':' in args.device else 0}"])
        else:
            overrides.append(f"compute.device={args.device}")
    overrides.extend(args.override)
    config = apply_dotlist_overrides(config, overrides)

    gpu_index = int(config.get("compute", {}).get("gpu_index", 0)) if config.get("compute", {}).get("device", "cuda") == "cuda" else -1
    sampler = GPUSampler(gpu_index=gpu_index, interval_seconds=args.sample_interval) if gpu_index >= 0 else None

    if sampler is not None:
        sampler.start()
    try:
        result = train_subspace_inr_3d_fmri(config)
    finally:
        if sampler is not None:
            sampler.stop()

    profile = {
        "config_path": str(Path(args.config).resolve()),
        "device": result.get("device"),
        "profile_summary": result.get("profile_summary", {}),
        "final_metrics": result.get("final_metrics", {}),
        "gpu_telemetry": {} if sampler is None else sampler.summary(),
    }

    output_path = Path(args.output).expanduser().resolve() if args.output else Path(result["run_dir"]) / "efficiency_profile.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(json.dumps(profile, indent=2))
    print(f"Saved efficiency profile to {output_path}")


if __name__ == "__main__":
    main()
