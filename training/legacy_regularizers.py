from __future__ import annotations

import torch


# Legacy DD-INR regularizers are intentionally kept out of the new main training path.
# They remain available here for future ablations without coupling them to subspaceINR4fMRI.


def charbonnier(x: torch.Tensor, eps: float = 1.0e-6) -> torch.Tensor:
    return torch.sqrt(x.square() + eps)



def lp_penalty(x: torch.Tensor, p: float, eps: float = 1.0e-12) -> torch.Tensor:
    return torch.pow(torch.abs(x) + eps, p)
