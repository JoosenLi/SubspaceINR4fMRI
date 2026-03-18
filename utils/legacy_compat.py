from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml



def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged



def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)



def resolve_path(path_value: str | None, base_dir: Path) -> Path | None:
    if path_value in (None, "", "null"):
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def parse_override_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        pass
    try:
        if "." in raw_value or "e" in lowered:
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def apply_dotlist_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    updated = deepcopy(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override {override!r}. Expected dotted.path=value.")
        dotted_key, raw_value = override.split("=", 1)
        value = parse_override_value(raw_value)
        cursor = updated
        parts = dotted_key.split(".")
        for key in parts[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[parts[-1]] = value
    return updated
