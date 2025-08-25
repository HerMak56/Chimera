from __future__ import annotations
import argparse
import yaml
import pytorch_lightning as pl

from chimera.data.gnm_datamodule import GNMDataModule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, default="configs/default.yaml")
    parser.add_argument("--fast", action="store_true", help="fast_dev_run=1")
    args = parser.parse_args()

    with open(args.conf, "r") as f:
        cfg = yaml.safe_load(f)

    pl.seed_everything(cfg.get("seed", 42), workers=True)

    dm = GNMDataModule(cfg)

    trainer = pl.Trainer(
        fast_dev_run=1 if args.fast else False,
        accelerator="auto",
        devices=1,
        precision="32-true",
        log_every_n_steps=5,
    )

    # Smoke: просто прогоним sanity steps на валидации
    # (или можно trainer.fit(model, datamodule=dm), если уже есть PLModule)
    trainer.validate(model=None, datamodule=dm)

if __name__ == "__main__":
    main()