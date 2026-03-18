from __future__ import annotations

import argparse
import pprint
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from HashINR.utils import load_yaml_config



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a subspaceINR4fMRI YAML config.")
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "configs" / "subspaceINR4fMRI.yaml"),
        help="Path to the YAML config file.",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    pprint.pprint(config, sort_dicts=False)


if __name__ == "__main__":
    main()
