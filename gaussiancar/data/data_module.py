from typing import Any

import lightning as L
import torch

from . import get_dataset_module_by_name


def collate_fn_with_list(batch):
    keys_to_list = ["radar_points"]
    
    # First, collect all keys from the batch
    all_keys = set()
    for item in batch:
        all_keys.update(item.keys())
    
    result = {}
    
    # Process each key
    for key in all_keys:
        if key in keys_to_list:
            # Keep as list
            result[key] = [item.get(key) for item in batch]
        else:
            # Batch the tensors
            values = [item.get(key) for item in batch if key in item]
            if len(values) > 0 and isinstance(values[0], torch.Tensor):
                try:
                    result[key] = torch.stack(values)
                except:
                    result[key] = values
            else:
                result[key] = values
    
    return result


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset: str,
        debug_mode: bool = False,
        data_config: dict = None,
        loader_config: dict = None,
        model_config: dict = None,
        img_params: dict = None,
    ) -> None:
        super().__init__()
        self.get_data = get_dataset_module_by_name(dataset).get_data
        self.data_config = data_config
        self.loader_config = loader_config
        self.model_config = model_config
        if img_params is not None:
            self.data_config['img_params'] = img_params

        if debug_mode:
            self.loader_config['batch_size'] = 1
            self.loader_config['num_workers'] = 0
            self.loader_config['prefetch_factor'] = None

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_data = self.get_data(split='train', **self.data_config)
            self.train_dataset = torch.utils.data.ConcatDataset(self.train_data)
        if stage in ["fit", "validate"]:
            self.val_data = self.get_data(split='val', **self.data_config)
            self.val_dataset = torch.utils.data.ConcatDataset(self.val_data)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            shuffle=True,
            collate_fn=collate_fn_with_list,
            **self.loader_config
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            shuffle=False,
            collate_fn=collate_fn_with_list,
            **self.loader_config
        )
