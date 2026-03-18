from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from HashINR.training import train_subspace_inr_3d_fmri
from HashINR.utils import apply_dotlist_overrides, load_yaml_config



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train subspaceINR4fMRI.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "subspaceINR4fMRI.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config override such as training.num_epochs=20. Can be passed multiple times.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.device is not None:
        if args.device.startswith("cuda"):
            config.setdefault("compute", {})
            config["compute"]["device"] = "cuda"
            config["compute"]["gpu_index"] = int(args.device.split(":", 1)[1]) if ":" in args.device else 0
        else:
            config.setdefault("compute", {})
            config["compute"]["device"] = args.device
    if args.override:
        config = apply_dotlist_overrides(config, args.override)
    result = train_subspace_inr_3d_fmri(config)
    print("Training finished.")
    print(result)


if __name__ == "__main__":
    main()
