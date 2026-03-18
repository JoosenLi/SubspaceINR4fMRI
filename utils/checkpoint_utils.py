from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch



def save_checkpoint(
    checkpoint: Dict,
    checkpoint_dir: Path,
    epoch: int,
    keep_last_k: int = 5,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"
    torch.save(checkpoint, checkpoint_path)

    if keep_last_k > 0:
        checkpoints = sorted(checkpoint_dir.glob("epoch_*.pt"))
        for stale_path in checkpoints[:-keep_last_k]:
            stale_path.unlink(missing_ok=True)

    latest_path = checkpoint_dir / "latest.pt"
    torch.save(checkpoint, latest_path)
    return checkpoint_path
