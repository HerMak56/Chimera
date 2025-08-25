from __future__ import annotations
from typing import Optional, Dict, Any
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from chimera.data.gnm_dataset import GNM2SocialNavDataset
from chimera.data.collate import socialnav_collate

class GNMDataModule(pl.LightningDataModule):
    """
    LightningDataModule для GNM2SocialNavDataset.
    Использует time-major collate под твой Learner (VANP-стиль).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = cfg
        self.ds_train = None
        self.ds_val = None

    def setup(self, stage: Optional[str] = None) -> None:
        data_cfg = self.cfg["data"]["gnm"]
        # train
        self.ds_train = GNM2SocialNavDataset(
            obs_len=data_cfg["obs_len"],
            pred_len=data_cfg["pred_len"],
            use_yaw=data_cfg.get("use_yaw", False),
            train=True,
            resize=tuple(data_cfg["resize"]),
            use_mask=data_cfg.get("use_mask", False),
            data_path=data_cfg["data_path"],
            mask_root=data_cfg.get("mask_root", data_cfg["data_path"]),
            seed=self.cfg.get("seed", 42),
        )
        # val
        self.ds_val = GNM2SocialNavDataset(
            obs_len=data_cfg["obs_len"],
            pred_len=data_cfg["pred_len"],
            use_yaw=data_cfg.get("use_yaw", False),
            train=False,
            resize=tuple(data_cfg["resize"]),
            use_mask=data_cfg.get("use_mask", False),
            data_path=data_cfg["data_path"],
            mask_root=data_cfg.get("mask_root", data_cfg["data_path"]),
            seed=self.cfg.get("seed", 42),
        )

    def train_dataloader(self) -> DataLoader:
        lcfg = self.cfg["loader"]
        return DataLoader(
            self.ds_train,
            batch_size=lcfg["batch_size"],
            shuffle=True,
            num_workers=lcfg.get("num_workers", 4),
            pin_memory=lcfg.get("pin_memory", True),
            persistent_workers=lcfg.get("persistent_workers", True),
            prefetch_factor=lcfg.get("prefetch_factor", 2),
            drop_last=lcfg.get("drop_last", True),
            collate_fn=socialnav_collate,
        )

    def val_dataloader(self) -> DataLoader:
        lcfg = self.cfg["loader"]
        return DataLoader(
            self.ds_val,
            batch_size=lcfg["batch_size"],
            shuffle=False,
            num_workers=lcfg.get("num_workers", 4),
            pin_memory=lcfg.get("pin_memory", True),
            persistent_workers=lcfg.get("persistent_workers", True),
            prefetch_factor=lcfg.get("prefetch_factor", 2),
            drop_last=False,
            collate_fn=socialnav_collate,
        )