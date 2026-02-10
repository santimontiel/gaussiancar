import os
from PIL import Image
from PIL.ImageTransform import AffineTransform
from pyquaternion import Quaternion
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchvision
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix


class ImageDataLoader:
    def __init__(
        self,
        img_params: Dict[str, Any],
        dataset_dir: str,
    ) -> None:
        self.img_params = img_params
        self.dataset_dir = dataset_dir
        self.to_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=img_params.get("mean", [0.485, 0.456, 0.406]),
                std=img_params.get("std", [0.229, 0.224, 0.225]),
            ),
        ])
        
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:

        # Load images and camera parameters from data_dict info.
        images: List[Image.Image] = []
        intrinsics: List[np.ndarray] = []
        extrinsics: List[torch.Tensor] = []

        for i, (img_path, intr, extr) in enumerate(zip(
            data_dict["images"],
            data_dict["intrinsics"],
            data_dict["extrinsics"],
        )):
            this_image = self._load_image(img_path)
            this_intr = np.array(intr, dtype=np.float32)
            this_extr = torch.tensor(extr, dtype=torch.float32)

            # Update intrinsics and preprocess image with augmentation matrix if available.
            if "ida_mat" in data_dict:
                ida_mat = data_dict["ida_mat"][i]
                this_intr = ida_mat @ this_intr
                final_dims = tuple(self.img_params["final_dim"][::-1])
                this_image = self._pil_preprocess_from_affine_mat(
                    this_image, ida_mat, final_dims
                )
            else:
                this_image = self.to_tensor(this_image)

            images.append(this_image)
            intrinsics.append(this_intr)
            extrinsics.append(this_extr)

        data_dict["image"] = torch.stack(images, 0)
        data_dict["intrinsics"] = intrinsics
        data_dict["extrinsics"] = extrinsics

        # Compute lidar to image transformation matrices.
        lidar2img: List[torch.Tensor] = []
        for intr, extr in zip(intrinsics, extrinsics):
            viewpad = torch.eye(4, dtype=torch.float32)
            viewpad[:intr.shape[0], :intr.shape[1]] = intr
            lidar2img.append(viewpad @ extr)
        data_dict["lidar2img"] = torch.stack(lidar2img, 0)

        # Update lidar2img and extrinsics according to bev augmentation if available.
        if "bev_augm" in data_dict:
            bev_augm = torch.from_numpy(data_dict["bev_augm"]).float()
            data_dict["extrinsics"] = torch.stack(data_dict["extrinsics"], 0) @ bev_augm
            data_dict["lidar2img"] = data_dict["lidar2img"] @ bev_augm

        return data_dict
    
    def _load_image(self, rel_image_path: str) -> Image.Image:
        path = os.path.join(self.dataset_dir, rel_image_path)
        return Image.open(path).convert("RGB")
    
    def _pil_preprocess_from_affine_mat(self, img, affine_mat, final_dims):
        inv_mat = np.linalg.inv(affine_mat)
        img = img.transform(
            size=tuple(final_dims),
            method=AffineTransform(inv_mat[:2].ravel()),
            resample=Image.BILINEAR,
        )
        return self.to_tensor(img)
        

class RadarDataLoader:
    def __init__(
        self,
        nusc,
        num_sweeps: int = 7,
        lidar_name: str = "LIDAR_TOP",
        radar_names: Optional[list] = [
            "RADAR_BACK_RIGHT",
            "RADAR_BACK_LEFT",
            "RADAR_FRONT",
            "RADAR_FRONT_LEFT",
            "RADAR_FRONT_RIGHT",
        ],
    ) -> None:
        self.nusc = nusc
        self.num_sweeps = num_sweeps
        self.lidar_name = lidar_name
        self.radar_names = radar_names

    def __call__(self, data_dict: Dict[str, any]) -> Dict[str, any]:

        # Assert necesary keys are present.
        assert "bev_augm" in data_dict, "Key 'bev_augm' not found in data_dict"
        assert "token" in data_dict, "Key 'token' not found in data_dict"
        bev_augm = data_dict["bev_augm"]

        # Extract sample information.
        sample_token = data_dict["token"]
        sample_rec = self.nusc.get("sample", sample_token)
        ref_token = sample_rec["data"][self.lidar_name]
        ref_sd_record = self.nusc.get("sample_data", ref_token)

        # Iterate through radar sensors and load points.
        radar_points = []
        for radar_name in self.radar_names:

            radar_token = sample_rec["data"][radar_name]
            sd_record = self.nusc.get("sample_data", radar_token)

            # Points are loaded in LiDAR coordinates.
            pc, times = RadarPointCloud.from_file_multisweep(
                nusc=self.nusc,
                sample_rec=sample_rec,
                chan=radar_name,
                ref_chan=self.lidar_name,
                nsweeps=self.num_sweeps,
                min_distance=1.0,
            )

            # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
            # point cloud.
            radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
            ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
            ego_pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
            velocities = pc.points[8:10, :]  # Compensated velocity
            velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
            velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
            velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
            velocities = np.dot(Quaternion(ego_pose_record['rotation']).rotation_matrix, velocities)
            velocities = velocities[:2, :].T  # Keep only x and y components

            # Transform points from LiDAR to ego coordinates.
            sensor_to_ego = transform_matrix(
                ref_cs_record["translation"],
                Quaternion(ref_cs_record["rotation"]),
            )
            homog_points = np.concatenate([pc.points[:3, :], np.ones((1, pc.points.shape[1]))], axis=0)
            homog_points = sensor_to_ego @ homog_points # Shape: (4, N)
            homog_points = self._augment_boxes(bev_augm, homog_points)
            xyz_ego = homog_points[:3, :].T  # Shape: (N, 3)

            # Perform encoding of RADAR variables.
            features = self.encode_radar_features(pc.points.T, xyz_ego, velocities, times.T, bev_augm)

            radar_points.append(features)

        # Stack all radar points into a single array
        if len(radar_points) > 0:
            radar_points = np.vstack(radar_points)

            # Mask the points that are outside the BEV grid.
            mask = (
                (radar_points[:, 0] >= -50.0) & (radar_points[:, 0] < 50.0) &
                (radar_points[:, 1] >= -50.0) & (radar_points[:, 1] < 50.0) &
                (radar_points[:, 2] >= -10.0) & (radar_points[:, 2] < 10.0)
            )
            radar_points = radar_points[mask]

        data_dict["radar_points"] = torch.from_numpy(radar_points).float()
        return data_dict
    
    def encode_radar_features(
        self,
        points: np.ndarray,
        xyz_ego: np.ndarray,
        velocities_ego: np.ndarray,
        times: np.ndarray,
        bev_augm: np.ndarray = None,
    ):
        """Perform encoding of RADAR variables from raw point clouds in
        the NuScenes dataset.

        Original features:
        - x, y, z [0, 1, 2]: Cartesian coordinates of the point with
            respect to the sensor.
        - dyn_prop [3]: Dynamic property of the point.
        - id [4]: Unique ID of the point.
        - rcs [5]: Radar Cross Section of the point.
        - vx, vy [6, 7]: Velocity of the point in x and y directions
            with respect to the sensor.
        - vx_comp, vy_comp [8, 9]: Compensated velocity of the point
            in x and y directions with respect to the sensor.
        - is_quality_valid [10]: Boolean flag indicating whether the
            quality of the point is valid.
        - ambig_state [11]: State of Doppler (radial velocity)
            ambiguity solution.
        - x_rms, y_rms [12, 13]: Root Mean Square of the point
            in x and y directions with respect to the sensor.
        - invalid_state [14]: State of the cluster validity.
        - pdh0 [15]: False alarm probability of the point. Probabilty
            of being an artifact caused by multipath or similar.
        - vx_rms, vy_rms [16, 17]: Root Mean Square of the
            compensated velocity of the point in x and y directions
            with respect to the sensor.

        We add two additional features:
        - times [18]: Delta of time between the present time and the
            capture time of the point.
        - nusc_filter [19]: Boolean flag indicating whether the point
            passes the NuScenes filter.

        Transformations are applied as follows:
        - x, y, z [0, 1, 2]: Move to ego coordinates. WARNING: This is
            precomputed and passes as xyz_ego.
        - dyn_prop [3]: One-hot encoded with 8 classes.
        - vx_comp, vy_comp [6, 7]: Move to ego coordinates and
            compensate multi-sweeps. TODO.
        - ambig_state [11]. One-hot encoded with 5 classes.
        - invalid_state [14]. One-hot encoded with 18 classes.
        - pdh0 [15]. Ordinal encoding with 8 possible values.
        - nusc_filter [19].

        Args:
            points (N, 18): Raw RADAR point cloud data.
            xyz_ego (N, 3): Cartesian coordinates of the point in ego coordinates.
            velocities_ego (N, 2): Compensated velocity of the point in ego coordinates.
            times (N, 1): Time delta of the point.
            bev_augm (4, 4): BEV augmentation matrix.
        Returns:
            features (N, F): Encoded RADAR features.
        """

        xyz = points[:, :3]  # Shape: (N, 3)
        dyn_prop = points[:, 3]  # Shape: (N,)
        ids = points[:, 4]  # Shape: (N,)
        rcs = points[:, 5]  # Shape: (N,)
        vxy = points[:, 6:8]  # Shape: (N, 2)
        vxy_comp = points[:, 8:10]  # Shape: (N, 2)
        is_quality_valid = points[:, 10]  # Shape: (N,)
        ambig_state = points[:, 11]  # Shape: (N,)
        xy_rms = points[:, 12:14]  # Shape: (N, 2)
        invalid_state = points[:, 14]  # Shape: (N,)
        pdh0 = points[:, 15]  # Shape: (N,)
        vxy_rms = points[:, 16:18]  # Shape: (N, 2)

        # One-hot encode dynamic property.
        dyn_prop_one_hot = np.zeros((xyz.shape[0], 8), dtype=np.float32)
        dyn_prop_one_hot[np.arange(xyz.shape[0]), np.rint(dyn_prop).astype(int)] = 1.0

        # Move compensated velocities to ego coordinates.

        # One-hot encode ambiguity state.
        ambig_state_one_hot = np.zeros((xyz.shape[0], 5), dtype=np.float32)
        ambig_state_one_hot[np.arange(xyz.shape[0]), np.rint(ambig_state).astype(int)] = 1.0

        # One-hot encode invalid state.
        invalid_state_one_hot = np.zeros((xyz.shape[0], 18), dtype=np.float32)
        invalid_state_one_hot[np.arange(xyz.shape[0]), np.rint(invalid_state).astype(int)] = 1.0

        # Ordinal encode pdh0.
        pdh0_encoded = np.zeros((xyz.shape[0], 7), dtype=np.float32)
        for i in range(7):
            pdh0_encoded[:, i] = (np.rint(pdh0) > i).astype(np.float32)

        # Calculate nusc_filter.
        nusc_filter = np.zeros(xyz.shape[0], dtype=np.float32)
        mask1 = (invalid_state == 0)
        mask2 = (dyn_prop < 7)
        mask3 = (ambig_state == 3)
        nusc_filter[mask1 & mask2 & mask3] = 1.0

        # Concatenate all features.
        features = np.concatenate(
            [
                xyz_ego,  # (N, 3)
                dyn_prop_one_hot,  # (N, 8)
                ids[:, None],  # (N, 1)
                rcs[:, None],  # (N, 1)
                vxy,  # (N, 2)
                velocities_ego,  # (N, 2)
                is_quality_valid[:, None],  # (N, 1)
                ambig_state_one_hot,  # (N, 5)
                xy_rms,  # (N, 2)
                invalid_state_one_hot,  # (N, 18)
                pdh0_encoded,  # (N, 7)
                vxy_rms,  # (N, 2)
                times,  # (N, 1)
                nusc_filter[:, None],  # (N, 1)
            ],
            axis=1,
        )  # Shape: (N, F)

        return features
    
    def _augment_boxes(self, bev_aug, points, inverse=True):
        """from PointBeV."""

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
            points_out = Rquery @ (points_in[:3, :] + tquery)
        else:
            Rquery[:3, :3] = bev_aug[:3, :3]
            tquery = np.array([1, 1, -1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]

            # Rquery @ X + tquery
            points_out = (Rquery @ points_in[:3, :]) + tquery

        return points_out


class LidarDataLoader:
    """
    Data loader for LiDAR point clouds from the nuScenes dataset.
    Processes multi-sweep data, transforms to ego coordinates, and applies BEV augmentation.
    """

    def __init__(
        self,
        nusc,
        num_sweeps: int = 10,
        lidar_names: List[str] = ["LIDAR_TOP"],
        min_distance: float = 1.0,
    ) -> None:
        """
        Args:
            nusc: The NuScenes instance.
            num_sweeps: Number of sweeps to aggregate.
            lidar_names: List of LiDAR sensors to use (usually just LIDAR_TOP).
            min_distance: Minimum distance to filter ego-vehicle reflections.
        """
        self.nusc = nusc
        self.num_sweeps = num_sweeps
        self.lidar_names = lidar_names
        self.min_distance = min_distance

    def __call__(self, data_dict: Dict[str, any]) -> Dict[str, any]:
        """
        Loads, transforms, and encodes LiDAR data for a specific sample.
        """
        # Assert necessary keys are present.
        assert "bev_augm" in data_dict, "Key 'bev_augm' not found in data_dict"
        assert "token" in data_dict, "Key 'token' not found in data_dict"
        
        bev_augm = data_dict["bev_augm"] # Shape (4, 4)

        # Extract sample information.
        sample_token = data_dict["token"]
        sample_rec = self.nusc.get("sample", sample_token)
        
        # We define the main LiDAR as the reference frame (usually LIDAR_TOP).
        ref_channel = self.lidar_names[0] 
        ref_token = sample_rec["data"][ref_channel]
        ref_sd_record = self.nusc.get("sample_data", ref_token)
        ref_cs_record = self.nusc.get("calibrated_sensor", ref_sd_record["calibrated_sensor_token"])

        lidar_points_list = []

        for lidar_name in self.lidar_names:
            
            # Points are loaded and transformed to the reference frame (LIDAR_TOP) 
            # by the SDK logic internally if ref_chan is set.
            pc, times = LidarPointCloud.from_file_multisweep(
                nusc=self.nusc,
                sample_rec=sample_rec,
                chan=lidar_name,
                ref_chan=ref_channel,
                nsweeps=self.num_sweeps,
                min_distance=self.min_distance,
            )

            # pc.points shape: (4, N) -> x, y, z, intensity
            
            # Create transformation matrix: Sensor (Lidar) -> Ego
            sensor_to_ego = transform_matrix(
                ref_cs_record["translation"],
                Quaternion(ref_cs_record["rotation"]),
            )

            # Homogeneous coordinates for geometric transformation
            # Shape: (4, N) -> x, y, z, 1
            homog_points = np.concatenate([pc.points[:3, :], np.ones((1, pc.points.shape[1]))], axis=0)
            
            # Transform to Ego frame
            homog_points = sensor_to_ego @ homog_points 
            
            # Apply Data Augmentation (Inverse BEV Augmentation)
            homog_points = self._augment_boxes(bev_augm, homog_points) 
            
            xyz_ego = homog_points[:3, :].T  # Shape: (N, 3)

            # Perform encoding of LiDAR variables.
            features = self.encode_lidar_features(pc.points.T, xyz_ego, times.T)
            
            lidar_points_list.append(features)

        # Stack all points (if multiple LiDARs were used)
        if len(lidar_points_list) > 0:
            lidar_points = np.vstack(lidar_points_list)

            # Mask points outside the BEV grid bounds.
            mask = (
                (lidar_points[:, 0] >= -50.0) & (lidar_points[:, 0] < 50.0) &
                (lidar_points[:, 1] >= -50.0) & (lidar_points[:, 1] < 50.0) &
                (lidar_points[:, 2] >= -10.0) & (lidar_points[:, 2] < 10.0)
            )
            lidar_points = lidar_points[mask]
        else:
            # Fallback for empty clouds
            lidar_points = np.zeros((0, 5), dtype=np.float32)

        data_dict["lidar_points"] = torch.from_numpy(lidar_points).float()
        return data_dict

    def encode_lidar_features(
        self,
        points: np.ndarray,
        xyz_ego: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """
        Encodes LiDAR features.
        
        Original features from LidarPointCloud:
        - x, y, z [0, 1, 2]: Coordinates in sensor frame.
        - intensity [3]: Reflection intensity (0-255).
        
        Output features:
        - x, y, z [0, 1, 2]: Coordinates in Ego frame (augmented).
        - intensity [3]: Scaled or raw intensity.
        - time [4]: Time delta.

        Args:
            points (N, 4): Raw LiDAR data [x, y, z, intensity].
            xyz_ego (N, 3): Transformed coordinates.
            times (N, 1): Time lag.

        Returns:
            features (N, 5): Encoded features.
        """
        
        intensity = points[:, 3:4] # Shape: (N, 1)
        
        # Optional: Normalize intensity to [0, 1] if it's in [0, 255]
        # intensity = intensity / 255.0 

        # Concatenate features
        features = np.concatenate(
            [
                xyz_ego,    # (N, 3)
                intensity,  # (N, 1)
                times,      # (N, 1)
            ],
            axis=1
        ) # Shape: (N, 5)

        return features

    def _augment_boxes(self, bev_aug, points, inverse=True):
        """
        Applies BEV augmentation transformation to points.
        
        Args:
            bev_aug (4, 4): Augmentation matrix.
            points (4, N): Homogeneous points.
            inverse (bool): Whether to apply inverse transformation.
        """
        points_in = np.copy(points)
        Rquery = np.zeros((3, 3))

        if isinstance(bev_aug, torch.Tensor):
            bev_aug = bev_aug.cpu().numpy()

        if inverse:
            # Inverse query aug:
            # Rotates and translates points to match the inverse of the BEV crop/rotation
            Rquery[:3, :3] = bev_aug[:3, :3].T
            tquery = np.array([-1, -1, 1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]

            # Rquery @ (X + tquery)
            points_out = Rquery @ (points_in[:3, :] + tquery)
        else:
            Rquery[:3, :3] = bev_aug[:3, :3]
            tquery = np.array([1, 1, -1]) * bev_aug[:3, 3]
            tquery = tquery[:, None]

            # Rquery @ X + tquery
            points_out = (Rquery @ points_in[:3, :]) + tquery

        return points_out