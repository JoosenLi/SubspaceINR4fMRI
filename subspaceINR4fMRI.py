from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from HashINR.training import train_subspace_inr_3d_fmri, train_subspace_inr_from_arrays
from HashINR.utils import apply_dotlist_overrides, load_yaml_config


def dynamic_Recon3D(
    kspaces: np.ndarray,
    trajs: np.ndarray,
    back_img: np.ndarray,
    config: Dict,
    case_name: Optional[str] = None,
    csm: Optional[np.ndarray] = None,
):
    return train_subspace_inr_from_arrays(
        kspaces=kspaces,
        trajs=trajs,
        back_img=back_img,
        config=config,
        case_name=case_name,
        csm=csm,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="subspaceINR4fMRI training entrypoint.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "subspaceINR4fMRI.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override such as cuda:0 or cpu.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config override, e.g. training.num_epochs=20. Can be provided multiple times.",
    )
    return parser.parse_args()



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
    result = train_subspace_inr_3d_fmri(config)
    print("Finished subspaceINR4fMRI training.")
    print(result)


if __name__ == "__main__":
    main()
