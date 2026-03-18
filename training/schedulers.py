from __future__ import annotations

from typing import Dict, Optional

import torch



def build_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    scheduler_cfg = config.get("scheduler", {})
    scheduler_type = str(scheduler_cfg.get("type", "none")).lower()
    if scheduler_type in {"none", "null", "disabled"}:
        return None
    if scheduler_type == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scheduler_cfg.get("t_max", 1000)),
            eta_min=float(scheduler_cfg.get("eta_min", 1.0e-7)),
        )
    raise ValueError(f"Unsupported scheduler type: {scheduler_cfg.get('type')}")
