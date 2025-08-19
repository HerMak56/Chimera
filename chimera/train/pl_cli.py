from __future__ import annotations
import argparse, yaml
from typing import Any, Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from chimera.utils.reproducibility import set_seed
from chimera.train.pl_module import VICRegPLModule

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Chimera Lightning Trainer")
    p.add_argument("--config", type=str, default="configs/vicreg_pl.yaml")
    p.add_argument("--clearml", action="store_true", help="enable ClearML logger")
    p.add_argument("--project", type=str, default="Chimera", help="ClearML project name")
    p.add_argument("--task", type=str, default="vicreg-baseline", help="ClearML task name")
    return p.parse_args()

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


class TwoViewNoiseDataset(Dataset):
    def __init__(self, n: int, dim: int, noise: float = 0.05, device: str = "cpu"):
        self.base = torch.randn(n, dim, device=device)
        self.noise = noise
        self.device = device
    def __len__(self) -> int:
        return self.base.shape[0]
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.base[idx]
        z1 = x + self.noise * torch.randn_like(x)
        z2 = x + self.noise * torch.randn_like(x)
        return z1, z2

def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    set_seed(cfg.get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset / DataLoader
    n_train = cfg["data"]["n_train"]
    dim = cfg["model"]["dim"]
    ds = TwoViewNoiseDataset(n_train, dim, noise=cfg["data"]["noise"], device=device)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)

    # LightningModule
    module = VICRegPLModule(cfg)

    # Logger: ClearML (по флагу)
    loggers = []
    if args.clearml:
        try:
            from clearml import Task
            task = Task.init(project_name=args.project, task_name=args.task, reuse_last_task_id=False)
            # Lightning сам подцепит std логгер через ClearML auto-logging
        except Exception as e:
            print(f"[warn] ClearML init failed: {e}")

    # Trainer
    trainer = pl.Trainer(
        max_steps=cfg["train"]["steps"],
        accelerator="gpu" if device == "cuda" else "cpu",
        devices=1,
        log_every_n_steps=cfg["train"].get("log_every", 50),
        enable_checkpointing=False,
        enable_model_summary=True,
    )

    trainer.fit(module, dl)

if __name__ == "__main__":
    main()