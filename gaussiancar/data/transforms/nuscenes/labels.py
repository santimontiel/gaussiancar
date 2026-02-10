import pathlib
from typing import Any, Dict, List, Literal

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rootutils
from nuscenes.utils.data_classes import Box
from torchvision.transforms import ToTensor

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from gaussiancar.data.common import sincos2quaternion, INTERPOLATION
from gaussiancar.utils.nuscenes_map_api import FixedNuScenesMap


class LoadSegmentationLabels:
    def __init__(
        self,
        labels_dir,
        mode: Literal["vehicle", "pedestrian"] = "vehicle",
        bev_conf: Dict[str, Any] = {"h": 200, "w": 200},
    ) -> None:
        self.labels_dir = pathlib.Path(labels_dir)
        self.mode = mode
        self.bev_conf = bev_conf
        self.bev_H = bev_conf["h"]
        self.bev_W = bev_conf["w"]
        self.to_tensor = ToTensor()

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:

        # Assert necessary keys are present in the data dictionary.
        assert "bev_augm" in data_dict, "BEV augmentation matrix not found in data_dict"
        assert "gt_box" in data_dict, "Ground truth box path not found in data_dict"
        assert "scene" in data_dict, "Scene name not found in data_dict"
        assert "view" in data_dict, "View matrix not found in data_dict"
        bev_augm = data_dict["bev_augm"]
        gt_box = data_dict["gt_box"]
        scene = data_dict["scene"]
        view = np.array(data_dict["view"])

        # Load boxes data.
        scene_dir = self.labels_dir / scene
        gt_box = np.load(scene_dir / gt_box, allow_pickle=True)['gt_box']

        # Arrange template tensors.
        bev = np.zeros((self.bev_H, self.bev_W), dtype=np.uint8)
        center_score = np.zeros((self.bev_H, self.bev_W), dtype=np.float32)
        center_offset = np.zeros((self.bev_H, self.bev_W, 2), dtype=np.float32)
        height_map = np.zeros((self.bev_H, self.bev_W), dtype=np.float32)
        dimensions = np.zeros((self.bev_H, self.bev_W, 3), dtype=np.float32)
        angle_map = np.zeros((self.bev_H, self.bev_W, 2), dtype=np.float32)
        visibility = np.full((self.bev_H, self.bev_W), 255, dtype=np.uint8)
        buf = np.zeros((self.bev_H, self.bev_W), dtype=np.uint8)
        coords = np.stack(np.meshgrid(np.arange(self.bev_W), np.arange(self.bev_H)), -1).astype(np.float32)
        sigma = 1

        # Iterate over boxes and fill the tensors.
        for box_data in gt_box:

            # Skip box if none.
            if len(box_data) == 0:
                continue

            # Select class (vehicle / pedestrian).
            class_idx = int(box_data[7])
            if class_idx == 5 and self.mode == 'vehicle':
                continue
            elif class_idx != 5 and self.mode == 'pedestrian':
                continue

            # Extract box parameters.
            translation = [box_data[0], box_data[1], box_data[4]]
            size = [box_data[2], box_data[3], box_data[5]]
            yaw = -box_data[6] - np.pi / 2
            visibility_token = box_data[8]
            box = Box(translation, size, sincos2quaternion(np.sin(yaw),np.cos(yaw)))
            points = box.bottom_corners()
            center = points.mean(-1)[:, None]

            # Project boxes to BEV plane: segmentation.
            homog_points = np.ones((4, 4))
            homog_points[:3, :] = points
            homog_points[-1, :] = 1
            points = self._prepare_augmented_boxes(bev_augm, homog_points)
            points[2] = 1 # add 1 for next matrix matmul
            points = (view @ points)[:2]
            cv2.fillPoly(bev, [points.round().astype(np.int32).T], 1, INTERPOLATION)

            # Project boxes to BEV plane: center, offsets, height.
            homog_points = np.ones((4, 1))
            homog_points[:3, :] = center
            homog_points[-1, :] = 1
            center = self._prepare_augmented_boxes(bev_augm, homog_points).astype(np.float32)
            center[2] = 1 # add 1 for next matrix matmul
            center = (view @ center)[:2, 0].astype(np.float32) # squeeze 1

            buf.fill(0)
            cv2.fillPoly(buf, [points.round().astype(np.int32).T], 1, INTERPOLATION)
            mask = buf > 0
            center_off = center[None] - coords
            center_offset[mask] = center_off[mask]
            g = np.exp(-(center_off ** 2).sum(-1) / (2 * sigma ** 2))
            center_score = np.maximum(center_score, g)
            object_height = translation[2]
            height_map[mask] = object_height
            object_dimensions = np.array(size, dtype=np.float32)
            dimensions[mask] = object_dimensions
            angle_map[mask, 0] = np.sin(yaw)
            angle_map[mask, 1] = np.cos(yaw)

            # Project boxes to BEV plane: visibility.
            visibility[mask] = visibility_token

        bev = self.to_tensor(255 * bev)
        center_score = self.to_tensor(center_score)
        center_offset = self.to_tensor(center_offset)
        height_map = self.to_tensor(height_map)
        dimensions = self.to_tensor(dimensions)
        angle_map = self.to_tensor(angle_map)
        visibility = torch.from_numpy(visibility)

        data_dict.update({
            f"{self.mode}": bev,
            f"{self.mode}_center": center_score,
            f"{self.mode}_offset": center_offset,
            f"{self.mode}_visibility": visibility,
            f"{self.mode}_height": height_map,
            f"{self.mode}_dimensions": dimensions,
            f"{self.mode}_angle": angle_map,
        })
        return data_dict

    # copied from PointBEV
    def _prepare_augmented_boxes(self, bev_aug, points, inverse=True):
        points_in = np.copy(points)
        Rquery = np.zeros((3, 3))

        if isinstance(bev_aug, torch.Tensor):
            bev_aug = bev_aug.cpu().numpy()

        if inverse:
            # Inverse query aug:
            # Ex: when tx=10, the query is 10/res meters front,
            # so points are fictivelly 10/res meters back.
            Rquery[:3, :3] = bev_aug[:3, :3].T
            tquery = np.array([-1, -1, 1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]

            # Rquery @ (X + tquery)
            points_out = (Rquery @ (points_in[:3, :] + tquery))
        else:
            Rquery[:3, :3] = bev_aug[:3, :3]
            tquery = np.array([1, 1, -1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]

            # Rquery @ X + tquery
            points_out = ((Rquery @ points_in[:3, :]) + tquery)

        return points_out


class LoadMapLabels:
    nusc_map_name: List[str] = [
        'boston-seaport',
        'singapore-onenorth',
        'singapore-hollandvillage',
        'singapore-queenstown'
    ]
    map_layers: List[str] = [
        "lane",
        "road_segment",
        "road_divider",
        "lane_divider",
        "ped_crossing",
        "walkway",
        "carpark_area",
    ]

    def __init__(
        self,
        dataset_dir,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.to_tensor = ToTensor()
        self.nusc_map = {}
        for map_name in self.nusc_map_name:
            self.nusc_map[map_name] = FixedNuScenesMap(dataroot=self.dataset_dir, map_name=map_name)

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:

        # Assert necessary keys are present in the data dictionary.
        assert "map_name" in data_dict, "Map name not found in data_dict"
        assert "pose" in data_dict, "Pose not found in data_dict"
        assert "view" in data_dict, "View not found in data_dict"
        assert "bev_augm" in data_dict, "BEV augmentation matrix not found in data_dict"
        bev_augm = data_dict["bev_augm"]
        map_name = data_dict["map_name"]
        pose = data_dict["pose"] @ bev_augm
        view = np.array(data_dict["view"])
        H, W = 200, 200

        # Geometrical parameters.
        S = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
        lidar2global = (view @ S @ np.linalg.inv(pose))
        rotation_lidar2global = lidar2global[:3, :3]
        v = np.dot(rotation_lidar2global, np.array([1, 0, 0]))
        yaw = np.arctan2(v[1], v[0])
        angle = (yaw / np.pi * 180)

        # Get the map layers.
        map_mask = self.nusc_map[map_name].get_map_mask(
            (pose[0][-1], pose[1][-1], 100, 100), angle, self.map_layers, (H, W)
        )
        for i, m in enumerate(map_mask):
            data_dict[self.map_layers[i]] = self.to_tensor(255 * np.flipud(m)[..., None])
        return data_dict

