import json
import torch
import numpy as np
import rootutils

from pathlib import Path
from .common import get_split
from typing import Dict, Optional
from nuscenes.nuscenes import NuScenes

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from gaussiancar.data.dataset_refactor import NuScenesDataset
from gaussiancar.data.transforms.augmentations import RandomTransformBev, RandomTransformImage
from gaussiancar.data.transforms.nuscenes.loading import ImageDataLoader, RadarDataLoader, LidarDataLoader
from gaussiancar.data.transforms.nuscenes.labels import LoadSegmentationLabels, LoadMapLabels


def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    image=None,                         # image config
    **dataset_kwargs
):
    out = []
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    # Override augment if not training
    training = True if split == 'train' else False
    
    # Arrange transformations.
    nusc = NuScenes(version=version, dataroot=str(dataset_dir), verbose=False)
    transforms = list()
    transforms += [
        RandomTransformImage(
            dataset_kwargs["img_params"],
            training=training,
            orig_img_size=(
                dataset_kwargs["img_params"]["W"],
                dataset_kwargs["img_params"]["H"],
            ),
        ),
        RandomTransformBev(dataset_kwargs["bev_aug_conf"], training=training),
        ImageDataLoader(dataset_kwargs["img_params"], str(dataset_dir)),
        RadarDataLoader(nusc, num_sweeps=7),
        # LidarDataLoader(nusc, num_sweeps=1),
    ]
    if dataset_kwargs.get("vehicle", False):
        transforms.append(LoadSegmentationLabels(labels_dir, mode="vehicle"))
    if dataset_kwargs.get("map_layers", False):
        transforms.append(LoadMapLabels(dataset_dir))

    # Format the split name
    split = f'mini_{split}' if version == 'v1.0-mini' else split
    split_scenes = get_split(split, 'nuscenes')

    for s in split_scenes:
        tmp_dataset = NuScenesDataset(s, labels_dir, transforms=transforms)
        out.append(tmp_dataset)
    return out
