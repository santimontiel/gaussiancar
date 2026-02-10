import logging

import hydra
import lightning as L
import torch
import rootutils
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

from gaussiancar.utils.config import register_new_resolvers, set_seed


def instantiate_trainer(cfg: DictConfig) -> L.Trainer:
    """Instantiate the Lightning Trainer."""

    # Instantiate the loggers.
    loggers = []
    if cfg.get("loggers", None):
        for logger_cfg in cfg.loggers:
            loggers.append(hydra.utils.instantiate(logger_cfg))
    
    # Instantiate the callbacks.
    callbacks = []
    if cfg.get("callbacks", None):
        for callback_cfg in cfg.callbacks:
            callbacks.append(hydra.utils.instantiate(callback_cfg))


    return L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        logger=loggers,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        accumulate_grad_batches=cfg.trainer.effective_batch_size
            // (cfg.trainer.batch_size * cfg.trainer.devices),
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to run the training process.
    """
    torch.set_float32_matmul_precision('high')
    set_seed(cfg.seed)
    register_new_resolvers()

    log.info(f"Loading datamodule <{cfg.data._target_}>...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    log.info(f"Loading module <{cfg.module._target_}>...")
    module: L.LightningModule = hydra.utils.instantiate(cfg.module, cfg=cfg)

    log.info("Instantiating the trainer...")
    trainer: L.Trainer = instantiate_trainer(cfg)

    log.info("Starting training...")
    trainer.fit(module, datamodule)

    log.info("Training completed. ✅")


if __name__ == "__main__":
    main()