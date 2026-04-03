from __future__ import annotations

import argparse
import pprint
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    pprint.pprint(config, sort_dicts=False)


if __name__ == "__main__":
    main()
