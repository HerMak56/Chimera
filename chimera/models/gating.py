from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import torch
from torch import nn

@dataclass
class GatingConfig:
    dim: int
    hidden: int = 0
    bias: bool = True
    temperature: float = 1.0  # <1 — острее σ

class GatingModule(nn.Module):
    r"""Trainable gate splitting embedding z into common/unique parts.

    α = σ((Wz+b)/τ);  z_c = α ⊙ z;  z_u = (1-α) ⊙ z
    """
    def __init__(self, cfg: GatingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(cfg.dim, cfg.hidden, bias=cfg.bias),
                nn.ReLU(),
                nn.Linear(cfg.hidden, cfg.dim, bias=cfg.bias),
            )
        else:
            self.net = nn.Linear(cfg.dim, cfg.dim, bias=cfg.bias)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.net(z) / max(self.cfg.temperature, 1e-6)
        alpha = torch.sigmoid(logits)
        z_c = alpha * z
        z_u = (1.0 - alpha) * z
        return z_c, z_u, alpha