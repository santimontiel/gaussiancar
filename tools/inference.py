from functools import partial

import logging
import hydra
import lightning as L
import rootutils
import torch
from nuscenes.nuscenes import NuScenes
from tqdm.auto import tqdm
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

from gaussiancar.utils.config import register_new_resolvers
from gaussiancar.utils.consts import VALIDATION_DRN_SPLITS


def load_from_checkpoint(
    module: L.LightningModule,
    checkpoint_path: str,
    device: str = "cpu",
) -> L.LightningModule:
    """Load model weights from a checkpoint file.

    Args:
        module (L.LightningModule): The LightningModule instance to load weights into.
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        L.LightningModule: The LightningModule instance with loaded weights.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    module.load_state_dict(checkpoint["state_dict"])
    module.to(device)
    return module


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    """Main function to make a prediction in a selected sample using a trained model.
    """
    register_new_resolvers()

    log.info(f"Loading datamodule <{cfg.data._target_}>...")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    dataset = datamodule.val_dataset

    log.info(f"Loading module <{cfg.module._target_}> with weights from <{cfg.checkpoint_path}>...")
    module: L.LightningModule = hydra.utils.instantiate(cfg.module, cfg=cfg)
    if cfg.checkpoint_path:
        module = load_from_checkpoint(module, cfg.checkpoint_path, cfg.device)
        module.eval()
    module.to(cfg.device).eval()

    SAMPLE_INDEX = -1
    log.info(f"Running inference on sample index {SAMPLE_INDEX}...")
    sample = dataset[SAMPLE_INDEX]

    for k, v in sample.items():
        if k.startswith("radar_points") or k.startswith("lidar_points"):
            sample[k] = [v.to(cfg.device)]
        elif torch.is_tensor(v):
            sample[k] = v.to(cfg.device).unsqueeze(0)
        elif isinstance(v, list) and torch.is_tensor(v[0]):
            sample[k] = [item.to(cfg.device) for item in v]
        else:
            continue

    # Forward pass through the model.
    with torch.no_grad():
        outputs = module(sample)

    log.info("Inference completed. ✅")



if __name__ == "__main__":
    main()