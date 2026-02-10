import logging
from pathlib import Path

import json
import hydra
import numpy as np
import rootutils
import torch
from tqdm.auto import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
log = logging.getLogger(__name__)

from gaussiancar.utils.config import register_new_resolvers


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg):
    """Entry point for the data creation script.

    Creates the following dataset structure:

    cfg.data.data_config.labels_dir/
        {scene_name}.json
        {scene_name}/
            gt_box_{scene_token}.npz
            TODO: radar_points_{scene_token}.npz
    """

    register_new_resolvers()

    # Settings for the create data process.
    cfg.data.dataset = cfg.data.dataset.replace("_generated", "")
    cfg.data.data_config.augment_bev = False
    cfg.data.data_config.augment_img = False

    cfg.data.loader_config.batch_size = 1
    cfg.data.loader_config.num_workers = 0
    cfg.data.loader_config.prefetch_factor = None

    # Instantiate the datamodule.
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")

    # Create the output directories.
    labels_dir = Path(cfg.data.data_config.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over the dataset and save the labels and point clouds.
    log.info(f"Creating data in {labels_dir}")
    for split in ["train", "val"]:
        log.info(f"Processing {split} split")
        data = getattr(datamodule, f"{split}_data")

        for scene in tqdm(data, desc=f"Processing {split} scenes...", leave=False):
            scene_dir = labels_dir / scene.scene_name
            scene_dir.mkdir(parents=True, exist_ok=True)

            info = []
            loader = torch.utils.data.DataLoader(
                scene,
                collate_fn=list,
                **datamodule.loader_config,
            )

            for batch in tqdm(loader, desc=f"Processing {split} {scene.scene_name}..."):
                info.extend(batch)

            # Write the scene info file.
            scene_info_path = labels_dir / f"{scene.scene_name}.json"
            with open(scene_info_path, "w") as f:
                json.dump(info, f, indent=4)
            

if __name__ == "__main__":
    main()