from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from subspaceINR4fMRI.evaluation import evaluate_checkpoint, evaluate_reconstruction_path
from subspaceINR4fMRI.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a reconstruction or checkpoint with notebook-equivalent F2 metrics.")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs" / "subspaceINR4fMRI.yaml"))
    parser.add_argument("--reconstruction", type=str, default=None, help="Path to a reconstruction .npy file.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a training checkpoint .pt file.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save evaluation outputs.")
    parser.add_argument("--device", type=str, default=None, help="Device override for checkpoint reconstruction, e.g. cuda:0.")
    parser.add_argument("--scale-factor", type=float, default=None, help="Optional scale factor for raw reconstruction files.")
    parser.add_argument(
        "--input-is-scaled",
        action="store_true",
        help="Treat the provided reconstruction file as scaled by data.scale_factor (default true for HashINR outputs).",
    )
    parser.add_argument(
        "--input-is-unscaled",
        action="store_true",
        help="Treat the provided reconstruction file as already unscaled.",
    )
    parser.add_argument(
        "--print-full",
        action="store_true",
        help="Print the full evaluation JSON instead of a compact summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.reconstruction) == bool(args.checkpoint):
        raise SystemExit("Provide exactly one of --reconstruction or --checkpoint.")

    config = load_yaml_config(args.config)
    output_dir = None if args.output_dir is None else Path(args.output_dir).expanduser().resolve()

    if args.checkpoint:
        result = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
            device_override=args.device,
        )
    else:
        input_is_scaled = not args.input_is_unscaled
        if args.input_is_scaled:
            input_is_scaled = True
        result = evaluate_reconstruction_path(
            reconstruction_path=args.reconstruction,
            config=config,
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
            scale_factor=args.scale_factor,
            input_is_scaled=input_is_scaled,
        )

    if args.print_full:
        payload = result
    else:
        payload = {
            "method_name": result.get("method_name"),
            "num_frames": result.get("num_frames"),
            "f2_fixed": result.get("f2_fixed"),
            "f2_max": result.get("f2_max"),
            "best_thresh": result.get("best_thresh"),
            "actual_thresh": result.get("actual_thresh"),
            "roi_signal_corr": result.get("roi_signal_corr"),
            "roi_tsnr": result.get("roi_tsnr"),
            "metrics_path": None if output_dir is None else str(output_dir / "metrics.json"),
            "activation_scores_path": None if output_dir is None else str(output_dir / "activation_scores.json"),
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
