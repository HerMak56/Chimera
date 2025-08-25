# train_vanp.py
from __future__ import annotations
import os
import yaml
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
from clearml import Task
from chimera.train.vanp_lighting.vanp_predtext_datamodule import VANPPretextModule
from chimera.data.gnm_datamodule import GNMDataModule  # импорт твоего DataModule
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train VANP pretext (Lightning).")
    parser.add_argument(
        "--conf",
        type=str,
        default="./configs/default.yaml",
        help="Path to YAML config.",
    )
    return parser.parse_args(argv)

def main():
    args = parse_args()
    with open(args.conf, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg['logger'].get("clearml", {}).get("use", True):
        task = Task.init(
            project_name=cfg['logger']["clearml"].get("project", "chimera"),
            task_name=cfg['logger']["clearml"].get("task", "vanp_pretext"),
            reuse_last_task_id=False,
        )
        # это прикрепит YAML в “Configuration”
        task.connect(cfg)
        # опционально теги
        for t in cfg['logger']["clearml"].get("tags", []):
            task.add_tags(t)
    # устройство
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    # seed
    pl.seed_everything(cfg.get("seed", 42), workers=True)

    # модули
    dm = GNMDataModule(cfg)
    model = VANPPretextModule(cfg)

    save_dir = cfg.get("save_dir", "./runs")
    os.makedirs(save_dir, exist_ok=True)
    ckpt = ModelCheckpoint(
        dirpath=os.path.join(save_dir, "ckpts"),
        filename="vanp-{epoch:03d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )
    lr_mon = LearningRateMonitor(logging_interval="step")
    progress = RichProgressBar()

    csv_logger = CSVLogger(save_dir, name="csv")
    tb_logger = TensorBoardLogger(save_dir, name="tb")

    # тренер
    trainer = pl.Trainer(
        max_epochs=cfg["train_params"].get("epochs", 100),
        accelerator="gpu" if "cuda" in device and torch.cuda.is_available() else "cpu",
        devices=cfg["train_params"].get("devices", 1),
        precision=cfg["train_params"].get("precision", 32),
        gradient_clip_val=cfg["train_params"].get("grad_clip", 0.0),
        accumulate_grad_batches=cfg["train_params"].get("accumulation_steps", 1),
        log_every_n_steps=cfg["train_params"].get("log_every_n_steps", 10),
        val_check_interval=cfg["train_params"].get("val_check_interval", 1.0),  # каждая эпоха
        check_val_every_n_epoch=cfg["train_params"].get("check_val_every_n_epoch", 1),
        callbacks=[ckpt, lr_mon, progress],
        logger=[csv_logger, tb_logger],
        deterministic=cfg["train_params"].get("deterministic", False),
    )

    # запуск
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.get("resume_from", None))

if __name__ == "__main__":
    main()