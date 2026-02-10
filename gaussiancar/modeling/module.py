from typing import Any, Dict, Literal

import lightning as L
import torch
import torch.nn as nn


class GaussianCaRModule(L.LightningModule):
    def __init__(
        self,
        cfg: Any = None,
        model: nn.Module = None,
        losses: Any = None,
        metrics: Any = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.losses = losses
        self.metrics = torch.nn.ModuleDict(metrics)
        

    def forward(self, batch: Dict[str, Any]):
        return self.model(batch)


    def common_step(
        self,
        batch: Dict[str, Any],
        stage: Literal["train", "val"] = "train",
    ) -> Dict[str, Any]:

        # Move batch to device.
        if self.cfg.trainer.precision == "bf16":
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(dtype=torch.bfloat16)
                elif isinstance(v, list):
                    batch[k] = [
                        item.to(dtype=torch.bfloat16)
                        if isinstance(item, torch.Tensor)
                        else item
                        for item in v
                    ]

        # Get batch size.
        B = batch["image"].shape[0]

        # Forward pass.
        outputs = self(batch)

        # Compute losses.
        loss, loss_details, weights = self.losses(outputs["output"], batch)
        aux_loss, aux_loss_details, aux_weights = self.losses(outputs["aux_output"], batch)

        # Update metrics.
        for k in self.metrics.keys():
            self.metrics[k].update(outputs["output"], batch)

        # Log losses and metrics.
        self.log(
            f"{stage}/loss",
            loss.detach(), on_step=False, on_epoch=True,
            logger=True, batch_size=B, 
        )
        self.log_dict(
            {f"{stage}/loss/{k}": v.detach() for k, v in loss_details.items()},
            on_step=False, on_epoch=True,
            logger=True, batch_size=B, 
        )
        self.log(
            f"{stage}/aux_loss",
            aux_loss.detach(), on_step=False, on_epoch=True,
            logger=True, batch_size=B, 
        )
        self.log_dict(
            {f"{stage}/aux_loss/{k}": v.detach() for k, v in aux_loss_details.items()},
            on_step=False, on_epoch=True,
            logger=True, batch_size=B, 
        )

        if "num_gaussians_cam" in outputs:
            self.log(
                f"{stage}/num_gaussians_cam",
                outputs["num_gaussians_cam"],
                on_step=False, on_epoch=True,
                logger=True, batch_size=B, 
            )
        if "num_gaussians_radar" in outputs:
            self.log(
                f"{stage}/num_gaussians_radar",
                outputs["num_gaussians_radar"],
                on_step=False, on_epoch=True,
                logger=True, batch_size=B, 
            )

        return {
            "loss": loss + aux_loss,
        }

    
    def training_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, Any]:
        return self.common_step(batch, stage="train")


    def validation_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> Dict[str, Any]:
        return self.common_step(batch, stage="val")


    def log_per_epoch_metrics(
        self,
        stage: Literal["train", "val"] = "train",
    ) -> None:
        ious = list()
        for k in self.metrics.keys():
            res = self.metrics[k].compute()
            self.log(f"{stage}/metrics/{k}", res[k], on_epoch=True, logger=True, )
            self.log(f"{stage}/metrics/{k}_max_threshold", res["max_threshold"], on_epoch=True, logger=True, )
            ious.append(res[k])
            self.metrics[k].reset()
        self.log(f'{stage}/metrics/mIoU', torch.stack(ious).mean(), on_epoch=True, logger=True, )


    def on_validation_start(self) -> None:
        self.log_per_epoch_metrics(stage="train")


    def on_validation_epoch_end(self) -> None:
        self.log_per_epoch_metrics(stage="val")


    def configure_optimizers(self) -> Dict[str, Any]:

        if self.cfg.optimizer.name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.optimizer.lr,
                weight_decay=self.cfg.optimizer.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.cfg.optimizer.name}")
        
        if self.cfg.scheduler.name == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.scheduler.max_lr,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.cfg.scheduler.pct_start,
                anneal_strategy=self.cfg.scheduler.anneal_strategy,
                div_factor=self.cfg.scheduler.div_factor,
                final_div_factor=self.cfg.scheduler.final_div_factor,
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.cfg.scheduler.name}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
