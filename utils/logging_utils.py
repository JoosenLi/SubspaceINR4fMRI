from __future__ import annotations

from pathlib import Path
from typing import Any


class NullSummaryWriter:
    def add_scalar(self, *args: Any, **kwargs: Any) -> None:
        return

    def add_histogram(self, *args: Any, **kwargs: Any) -> None:
        return

    def close(self) -> None:
        return


try:
    from tensorboardX import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - optional dependency fallback
    SummaryWriter = None



def create_summary_writer(log_dir: Path, enabled: bool = True):
    if not enabled or SummaryWriter is None:
        return NullSummaryWriter()
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(log_dir))
