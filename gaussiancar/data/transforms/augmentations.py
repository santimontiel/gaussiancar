from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision
from scipy.spatial.transform import Rotation as R

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


class RandomTransformBev:
    """
    Handles random data augmentation for Bird's Eye View (BEV) transformation matrices.
    """

    def __init__(
        self, 
        bev_aug_conf: Optional[List[float]] = None, 
        training: bool = True
    ) -> None:
        """
        Args:
            bev_aug_conf: Configuration list containing [tx, ty, tz, rx, ry, rz] coefficients.
            training: Whether to apply augmentation (True) or return identity (False).
        """
        self.training = training
        self.bev_aug_conf = bev_aug_conf

    def get_random_ref_matrix(self) -> np.ndarray:
        """
        Generates a random reference transformation matrix using SciPy.

        Returns:
            np.ndarray: A 4x4 homogeneous transformation matrix (float32).
        """
        # Unpack configuration: first 3 are translation, last 3 are rotation
        coeffs = self.bev_aug_conf
        trans_coeff = np.array(coeffs[:3], dtype=np.float32)
        rot_coeff = np.array(coeffs[3:], dtype=np.float32)

        # Initialize 4x4 Identity matrix
        mat = np.eye(4, dtype=np.float32)

        # 1. Translation
        # Logic: Generate random values in range [-1, 1) and scale by coefficients
        random_trans_noise = np.random.random(3).astype(np.float32) * 2 - 1
        mat[:3, 3] = random_trans_noise * trans_coeff

        # 2. Rotation
        # Logic: Generate random Euler angles (zyx) in range [-1, 1), scale, and convert to matrix
        random_rot_noise = np.random.random(3).astype(np.float32) * 2 - 1
        random_zyx = random_rot_noise * rot_coeff
        
        mat[:3, :3] = R.from_euler("zyx", random_zyx, degrees=True).as_matrix()

        return mat

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the transformation.

        Args:
            data_dict: A dictionary containing data to be augmented.

        Returns:
            Dict[str, Any]: The updated data dictionary with the transformation matrix.
        """
        bev_augm = self.get_random_ref_matrix() if self.training else np.eye(4, dtype=np.float32)
        data_dict['bev_augm'] = bev_augm
        return data_dict
    


@dataclass
class AugmentationParams:
    """Holds configuration for a single image augmentation."""
    scale: float
    resize_dims: Tuple[int, int]  # (width, height)
    crop: Tuple[int, int, int, int]  # (left, top, right, bottom)
    flip: bool
    rotate: float
    crop_zoom: Tuple[int, int, int, int]
    final_dims: Tuple[int, int]
    
    @property
    def ida_mat_args(self) -> Tuple:
        """Helper to unpack args for affinity matrix calculation."""
        return (
            self.scale, self.crop[1], self.crop_zoom, 
            self.flip, self.rotate, self.final_dims
        )

class RandomTransformImage:
    def __init__(
        self,
        img_params: dict,
        training: bool = True,
        transform: Optional[Any] = None,
        max_range: float = 80.0,
        orig_img_size: Tuple[int, int] = (1600, 900),
    ):
        self.img_params = img_params
        self.training = training
        self.transform = (
            transform if transform is not None
            else torchvision.transforms.ToTensor()
        )
        self.max_range = max_range
        self.orig_img_size = orig_img_size

    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        data_dict['ida_mat']: List[torch.Tensor] = []
        num_images = len(data_dict['images'])

        for _ in range(num_images):
            aug_params = self.sample_params()
            ida_mat = self.get_affinity_matrix(aug_params, self.orig_img_size)
            data_dict['ida_mat'].append(torch.tensor(ida_mat))

        data_dict['ida_mat'] = torch.stack(data_dict['ida_mat'], dim=0)
        return data_dict


    def sample_params(self) -> AugmentationParams:
            """Generates augmentation parameters based on current mode (train/eval)."""
            H, W = self.img_params["H"], self.img_params["W"]
            final_dims = tuple(self.img_params["final_dim"][::-1]) # (W, H)

            if self.training:
                scale = np.random.uniform(*self.img_params["scale"])
                newW, newH = int(W * scale), int(H * scale)
                resize_dims = (newW, newH)

                crop_h = int((1 - np.random.uniform(*self.img_params["crop_up_pct"])) * newH)
                crop = (0, crop_h, newW, newH)

                zoom = np.random.uniform(*self.img_params["zoom_lim"])
                crop_zoomh = ((newH - crop_h) * (1 - zoom)) // 2
                crop_zoomw = (newW * (1 - zoom)) // 2
                
                crop_zoom = (
                    -crop_zoomw,
                    -crop_zoomh,
                    crop_zoomw + newW,
                    crop_zoomh + newH - crop_h,
                )

                flip = self.img_params["rand_flip"] and np.random.choice([0, 1])
                rotate = np.random.uniform(*self.img_params["rot_lim"])
            else:
                scale = np.mean(self.img_params["scale"])
                newW, newH = int(W * scale), int(H * scale)
                resize_dims = (newW, newH)

                crop_h = int((1 - np.mean(self.img_params["crop_up_pct"])) * newH)
                crop = (0, crop_h, newW, newH)
                
                # zoom = 1.0 implicitly
                crop_zoom = (0, 0, newW, newH - crop_h)
                flip = False
                rotate = 0

            return AugmentationParams(
                scale=scale,
                resize_dims=resize_dims,
                crop=crop,
                flip=bool(flip),
                rotate=rotate,
                crop_zoom=tuple(map(int, crop_zoom)),
                final_dims=final_dims
        )

    def get_affinity_matrix(
        self,
        params: AugmentationParams,
        input_size: Tuple[int, int],
    ) -> np.ndarray:
            """Calculates the affine transformation matrix."""
            # Unpack specific params needed for calculation
            scale, crop_sky, crop_zoom, flip, rotate, final_dims = params.ida_mat_args
            
            # W_H default from original code logic (1600, 900)
            res = [input_size[0] * scale, input_size[1] * scale]

            affine_mat = np.eye(3)
            affine_mat[:2, :2] *= scale

            w, h = final_dims
            affine_mat[0, :2] *= w / (crop_zoom[2] - crop_zoom[0])
            affine_mat[1, :2] *= h / (crop_zoom[3] - crop_zoom[1])
            affine_mat[0, 2] += (w - res[0] * w / (crop_zoom[2] - crop_zoom[0])) / 2
            affine_mat[1, 2] += (h - (res[1] + crop_sky) * h / (crop_zoom[3] - crop_zoom[1])) / 2

            if flip:
                flip_mat = np.eye(3)
                flip_mat[0, 0] = -1
                flip_mat[0, 2] += w
                affine_mat = flip_mat @ affine_mat

            theta = -rotate * np.pi / 180
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            x, y = w / 2, h / 2
            
            rot_center_mat = np.array([
                [cos_theta, -sin_theta, -x * cos_theta + y * sin_theta + x],
                [sin_theta, cos_theta, -x * sin_theta - y * cos_theta + y],
                [0, 0, 1],
            ])
            
            return (rot_center_mat @ affine_mat).astype(np.float32)

