from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
from torch import nn
import pytorch_lightning as pl

from chimera.models.gating import GatingModule, GatingConfig
from chimera.losses.split_losses import VICRegLoss

class SimpleEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2*dim), nn.ReLU(),
            nn.Linear(2*dim, dim)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class VICRegPLModule(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)

        dim = cfg["model"]["dim"]
        self.encoder = SimpleEncoder(dim)
        gcfg = GatingConfig(
            dim=dim,
            hidden=cfg["gating"]["hidden"],
            temperature=cfg["gating"]["temperature"],
        )
        self.gate = GatingModule(gcfg)
        self.crit = VICRegLoss(**cfg["losses"]["vicreg"])

        self.lr = cfg["optim"]["lr"]
        self.wd = cfg["optim"]["wd"]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        z1_raw, z2_raw = batch
        e1, e2 = self.encoder(z1_raw), self.encoder(z2_raw)
        z1_c, _, a1 = self.gate(e1)
        z2_c, _, a2 = self.gate(e2)

        loss = self.crit(z1_c, z2_c)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/alpha_mean", 0.5*(a1.mean()+a2.mean()), prog_bar=False, on_step=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.gate.parameters()),
            lr=self.lr, weight_decay=self.wd
        )
        return opt