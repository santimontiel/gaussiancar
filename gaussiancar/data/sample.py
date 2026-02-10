import pathlib


class Sample(dict):
    def __init__(
        self,
        token,
        scene,
        map_name,
        intrinsics,
        extrinsics,
        images,
        view,
        gt_box,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Used to create path in save/load
        self.token = token
        self.scene = scene
        self.view = view
        self.map_name = map_name
        self.images = images
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.gt_box = gt_box

    def __getattr__(self, key):
        return super().__getitem__(key)

    def __setattr__(self, key, val):
        self[key] = val

        return super().__setattr__(key, val)


class SaveDataTransform:
    """
    All data to be saved to .json must be passed in as native Python lists
    """
    def __init__(self, labels_dir):
        self.labels_dir = pathlib.Path(labels_dir)

    def get_cameras(self, batch: Sample):
        return {
            'images': batch.images,
            'intrinsics': batch.intrinsics,
            'extrinsics': batch.extrinsics
        }

    def get_box(self, batch: Sample):
        scene_dir = self.labels_dir / batch.scene
        gt_box_path = f'gt_box_{batch.token}.npz'
        gt_box = batch.gt_box
        np.savez_compressed(scene_dir / gt_box_path, gt_box=gt_box)
        return {'gt_box': gt_box_path}

    def __call__(self, batch):
        """
        Save sensor/label data and return any additional info to be saved to json
        """
        result = {}
        result.update(self.get_cameras(batch))
        result.update(self.get_box(batch))
        result.update({k: v for k, v in batch.items() if k not in result})

        return result
