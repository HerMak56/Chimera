from __future__ import annotations
from typing import Any, Dict, List
import torch
import torch.nn as nn
import pytorch_lightning as pl
from clearml import Task
from chimera.models.vanp.build_pred_text import build_pretext_model
from chimera.losses.split_losses import VICRegLoss

class VANPPretextModule(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = build_pretext_model(cfg=cfg['model'], device=cfg['device'])
        self.criterion = VICRegLoss(
            inv=self.cfg['train_params']['vicreg_loss'].get("sim_coeff", 25),
            var=self.cfg['train_params']['vicreg_loss'].get("std_coeff", 0.5),
            cov=self.cfg['train_params']['vicreg_loss'].get("cov_coeff", 25),
        )
        self.loss_lambda = float(cfg['train_params']['vicreg_loss'].get("loss_lambda", 0.5))
    def training_step(self, batch, batch_idx):
        frames = batch["past_frames"]
        future_frame = batch["future_frame"]
        future_actions = batch['future_positions']

        _, img_z, action_z, future_frame_z = self.model(frames, future_frame, future_actions)

        loss_v, m_v = self.criterion(img_z, future_frame_z)
        loss_a, m_a = self.criterion(img_z, action_z)

        total = self.loss_lambda * loss_v + (1.0 - self.loss_lambda) * loss_a

        self.log_dict({
            "train/loss": total,
            "train/loss_vision": loss_v,
            "train/loss_action": loss_a,
            "train/repr": m_v["repr"] + m_a["repr"],
            "train/std":  m_v["std"]  + m_a["std"],
            "train/cov":  m_v["cov"]  + m_a["cov"],
        }, prog_bar=True, on_step=True, on_epoch=True, batch_size=future_frame.size(0))


        log_every = int(self.cfg.get("clearml", {}).get("log_every_n_steps", 10))
        if (self.global_step % log_every) == 0:
            self._report_to_clearml({
                "loss": total,
                "loss_vision": loss_v,
                "loss_action": loss_a,
                "repr": (m_v["repr"] + m_a["repr"]),
                "std": (m_v["std"]  + m_a["std"]),
                "cov": m_v["cov"]  + m_a["cov"],
            }, title="train", iteration=self.global_step)
        return total
    def validation_step(self, batch, batch_idx):
        frames = batch["past_frames"]
        future_frame = batch["future_frame"]
        future_actions = batch['future_positions']

        _, img_z, action_z, future_frame_z = self.model(frames, future_frame, future_actions)


        loss_v, m_v = self.criterion(img_z, future_frame_z)
        loss_a, m_a = self.criterion(img_z, action_z)
        total = self.loss_lambda * loss_v + (1.0 - self.loss_lambda) * loss_a

        self.log_dict({
            "val/loss": total,
            "val/loss_vision": loss_v,
            "val/loss_action": loss_a,
            "val/repr": m_v["repr"] + m_a["repr"],
            "val/std": m_v["std"] + m_a["std"],
            "val/cov": m_v["cov"] + m_a["cov"],
        }, prog_bar=True, on_step=False, on_epoch=True, batch_size=future_frame_z.size(0))

        log_every = int(self.cfg.get("clearml", {}).get("log_every_n_steps", 10))
        if (self.global_step % log_every) == 0:
            self._report_to_clearml({
                "loss": total,
                "loss_vision": loss_v,
                "loss_action": loss_a,
                "repr": (m_v["repr"] + m_a["repr"]),
                "std": (m_v["std"]  + m_a["std"]),
                "cov": m_v["cov"]  + m_a["cov"],
            }, title="val", iteration=self.global_step)
        return total
    
    def configure_optimizers(self):
        opt_cfg = self.cfg["optim"]["adamw"]  # например: {"lr":3e-4,"weight_decay":1e-4, "betas":[0.9,0.999]}
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=float(opt_cfg.get("lr", 3e-4)),
            weight_decay=float(opt_cfg.get("weight_decay", 1e-4)),
            betas=tuple(opt_cfg.get("betas", (0.9, 0.999))),
        )
        sch_cfg = self.cfg["optim"].get("cosine", {"use": False})
        if sch_cfg.get("use", False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=sch_cfg.get("t_max", 100)
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
    
    def _report_to_clearml(self, metrics: dict, title: str, iteration: int):
        task = Task.current_task()
        if not task:
            return
        # Логгер ClearML
        clr_logger = task.get_logger()
        # Отправляем каждую метрику как отдельную серию
        for k, v in metrics.items():
            # приведи к числу
            if isinstance(v, torch.Tensor):
                v = v.detach().float().item()
            clr_logger.report_scalar(
                title=title,     # логическая группа графиков, например "train" или "val"
                series=k,        # имя конкретной кривой, например "loss", "loss_vision", ...
                value=v,
                iteration=iteration
            )
