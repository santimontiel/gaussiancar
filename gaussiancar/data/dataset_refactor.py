from typing import Any, Callable, Dict, List, Optional

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def relative_transform(
    T_a: np.ndarray, T_b: np.ndarray
) -> np.ndarray:
    """
    Compute relative transform T_a→b using numpy.

    Args:
        T_a: (4,4) pose matrix of frame a in world coords.
        T_b: (4,4) pose matrix of frame b in world coords.

    Returns:
        (4,4) relative transform matrix.
    """
    if T_a.shape != (4, 4) or T_b.shape != (4, 4):
        raise ValueError("Inputs must be 4x4 transformation matrices.")

    # Inverse of T_a
    T_a_inv = np.linalg.inv(T_a)

    # Relative transform
    T_rel = T_a_inv @ T_b

    # # Extract rotation and translation
    # R_rel = T_rel[:3, :3]
    # t_rel = T_rel[:3, 3]

    return T_rel


class NuScenesDataset(Dataset):
    def __init__(self,
        scene_name: str,
        labels_dir: str,
        transforms: Optional[List[Callable]] = None,
    ) -> None:
        scene_path = Path(labels_dir) / f"{scene_name}.json"
        self.samples = json.loads(scene_path.read_text())
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)
    
    def _extract_data(self, sample_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts and converts necessary data fields from a raw sample dictionary.
        
        Args:
            sample_data: The raw dictionary for one frame loaded from the JSON.

        Returns:
            A processed dictionary with tensor/numpy types.
        """
        # Assuming the 'Sample' class from the original code was mainly for 
        # dictionary-like unpacking and access. We extract fields directly.
        data_dict: Dict[str, Any] = {
            'view': torch.tensor(sample_data['view']),
            'token': sample_data['token'],
            'map_name': sample_data['map_name'],
            'pose': np.float32(sample_data['pose']),
            'pose_inverse': np.float32(sample_data['pose_inverse']),
            'images': sample_data['images'],
            'intrinsics': sample_data['intrinsics'],
            'extrinsics': sample_data['extrinsics'],
            'scene': sample_data['scene'],
            'gt_box': sample_data['gt_box'],
        }
        return data_dict

    def __getitem__(self, idx):
        result: Dict[str, Any] = {}

        # 1. Extract base data from raw JSON sample
        raw_sample = self.samples[idx]
        data_dict = self._extract_data(raw_sample)

        # 2. Apply transformations
        for transform in self.transforms:
            data_dict = transform(data_dict)
            
        # 3. Suffix keys and populate the result dictionary
        for key, value in data_dict.items():
            result[key] = value

        return result