# python3.7
"""Contains the class of EG3D image dataset.
"""

import numpy as np
import json
import io

from utils.formatting_utils import raw_label_to_one_hot
from .base_dataset import BaseDataset
import torch
import math

import torchvision
import torch

__all__ = ['BerfSceneDataset']


class BerfSceneDataset(BaseDataset):
    """Defines the image dataset class.

    NOTE: In order to keep consistent with original implementation of the
    dataset, here we retain original `label` and use `pose` as the additional
    pose label in case of confusion.

    NOTE: Each image can be grouped with a simple label, which contanis its
    corresponding pose information. The returned item format is

    {
        'index': int,
        'raw_image': np.ndarray,
        'image': np.ndarray,
        'raw_label': int,  # optional
        'label': np.ndarray  # optional
    }

    Available transformation kwargs:

    - image_size: Final image size produced by the dataset. (required)
    - image_channels (default: 3)
    - min_val (default: -1.0)
    - max_val (default: 1.0)
    - use_square (default: True)
    - central_crop (default: True)
    """

    def __init__(self,
                 root_dir,
                 file_format='zip',
                 semantic_nc=1,
                 annotation_path=None,
                 annotation_meta=None,
                 annotation_format='json',
                 max_samples=-1,
                 mirror=False,
                 transform_kwargs=None,
                 use_label=True,
                 num_classes=None,
                 use_pose=True,
                 dataset_name='clevr',
                 bevlen=256,
                 pose_meta='dataset.json',
             ):

        """Initializes the dataset.

        Args:
            use_label: Whether to enable conditioning label? Even if manually
                set this to `True`, it will be changed to `False` if labels are
                unavailable. If set to `False` manually, dataset will ignore all
                given labels. (default: True)
            num_classes: Number of classes. If not provided, the dataset will
                parse all labels to get the maximum value. This field can also
                be provided as a number larger than the actual number of
                classes. For example, sometimes, we may want to leave an
                additional class for an auxiliary task. (default: None)
        """
        super().__init__(root_dir=root_dir,
                         file_format=file_format,
                         annotation_path=annotation_path,
                         annotation_meta=annotation_meta,
                         annotation_format=annotation_format,
                         max_samples=max_samples,
                         mirror=mirror,
                         transform_kwargs=transform_kwargs)

        self.dataset_classes = 0  # Number of classes contained in the dataset.
        self.num_classes = 0  # Actual number of classes provided by the loader.
        self.dataset_name=dataset_name
        self.semantic_nc=semantic_nc

        # Check if the dataset contains categorical information.
        self.use_label = False
        item_sample = self.items[0]
        if isinstance(item_sample, (list, tuple)) and len(item_sample) > 1:
            labels = [int(item[1]) for item in self.items]
            self.dataset_classes = max(labels) + 1
            self.use_label = use_label

        if self.use_label:
            if num_classes is None:
                self.num_classes = self.dataset_classes
            else:
                self.num_classes = int(num_classes)
            assert self.num_classes > 0
        else:
            self.num_classes = 0

        self.use_pose = use_pose
        if use_pose:
            fp = self.reader.open_anno_file(root_dir, pose_meta)
            self.poses, self.infos = self._load_raw_poses(fp)

        self.set_h(bevlen)
        self.set_w(bevlen)

    def _load_raw_poses(self, fp):
        f = json.load(fp)
        poses = f['labels']
        poses = dict(poses)
        poses = [
            poses[fname.replace('\\', '/')]
            for fname in self.items
        ]
        poses = np.array(poses)
        poses = poses.astype({1: np.int64, 2: np.float32}[poses.ndim])

        infos = f['infos']
        infos = dict(infos)
        infos = [
            infos[fname.replace('\\', '/')]
            for fname in self.items
        ]
        return poses, infos

    def _load_raw_infos(self, fp):
        infos = json.load(fp)['infos']
        return infos
        
    @staticmethod
    def get_R(theta):
        return np.array([[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0,0], [0,0,1,0], [0,0,0,1]]).astype(np.float32)

    def get_pose(self, idx):
        pose = self.poses[idx].copy()
        crop_size = self.transform_kwargs.get('crop_size')
        if self.dataset_name == 'carla':
            pose[16] *= 256 / crop_size
            pose[20] *= 256 / crop_size
        if self.dataset_name == 'clevr':
            cam2world = np.array(
				[
				 [-1, 0, 0, 0],
                                 [0, 0.5, -math.sqrt(3)/2, 12 * math.sqrt(3)/2],
                                 [0, -math.sqrt(3)/2, -0.5, 6],
                                 [0, 0, 0, 1]
				], dtype=float
				)
            world2cam = np.array(
				[
				 [-1, 0, 0, 0],
                                 [0, 0.5, -math.sqrt(3)/2, 0],
                                 [0, -math.sqrt(3)/2, -0.5, 12],
                                 [0, 0, 0, 1]
				], dtype=float
				)
            pose[:16] = cam2world.flatten()
            pose[16] *= 3/4
            pose[18] *= 1/4
            pose[20] *= 3/4
            pose[21] *= 1/4
        return pose.copy()

    def trans(self, x, y, z):
        assert self.w == self.h, 'Currently only support rectangular BEV'
        x = 0.5 * self.w - 128 + 256 - (x/9 + .5) * 256
        y = 0.5 * self.h - 128 + (y/9 + .5) * 256
        z = z / 9 * 256
        return x, y, z

    @property
    def COLOR_NAME_LIST(self):
        return ['cyan', 'green', 'purple', 'red', 'yellow', 'gray', 'brown', 'blue']

    @property
    def SHAPE_NAME_LIST(self):
        return ['cube', 'sphere', 'cylinder']

    @property
    def MATERIAL_NAME_LIST(self):
        return ['rubber', 'metal']

    def set_h(self, h):
        self.h_value = h

    @property
    def h(self):
        return self.h_value

    def set_w(self, w):
        self.w_value = w

    @property
    def w(self):
        return self.w_value

    def get_info(self, idx):
        return self.infos[idx]

    def get_bev(self, idx, nc=None, dx=None, dy=None, rotate_angle=0):
        if self.dataset_name == 'clevr':
            h, w = self.h, self.w
            objs = self.infos[idx]['objects']
            if nc is None:
                nc = 1 + len(self.COLOR_NAME_LIST) + len(self.SHAPE_NAME_LIST) + len(self.MATERIAL_NAME_LIST)

            canvas = np.zeros([h, w, nc])
            xx = np.ones([h, w]).cumsum(0)
            yy = np.ones([h, w]).cumsum(1)

            p = 0

            for obj in objs:
                x, y, z = obj['3d_coords']
                # here change x and y's position due to coordinate difference
                y, x, z = self.trans(x, y, z)
                if dx is not None:
                    x += dx[p]
                    y += dy[p]
                    p += 1

                shape = obj['shape']
                color = obj['color']
                material = obj['material']

                feat = [0] * nc
                feat[0] = 1
                feat[self.COLOR_NAME_LIST.index(color) + 1] = 1
                feat[self.SHAPE_NAME_LIST.index(shape) + 1 + len(self.COLOR_NAME_LIST)] = 1
                feat[self.MATERIAL_NAME_LIST.index(material) + 1 + len(self.COLOR_NAME_LIST) + len(self.SHAPE_NAME_LIST)] = 1
                feat = np.array(feat)
                rot = obj['rotation']
                rot_sin = np.sin(rot / 180 * np.pi)
                rot_cos = np.cos(rot / 180 * np.pi)

                if shape == 'cube':
                    mask = (np.abs(+rot_cos * (xx-x) + rot_sin * (yy-y)) <= z) * \
                           (np.abs(-rot_sin * (xx-x) + rot_cos * (yy-y)) <= z)
                else:
                    mask = ((xx-x)**2 + (y-yy)**2) ** 0.5 <= z
                canvas[mask] = feat
            canvas = np.transpose(canvas, [2, 0, 1]).astype(np.float32)
            rotate_angle = 0
            canvas = torchvision.transforms.functional.rotate(torch.tensor(canvas), rotate_angle).numpy()
        elif self.dataset_name == 'carla':
            ret = self.get_raw_data(idx)
            canvas = ret[-1]
        else:
            raise NotImplementedError
        
        return canvas[:self.semantic_nc, ...]

    def get_raw_data(self, idx):
        # Handle data mirroring.
        do_mirror = self.mirror and idx >= (self.num_samples // 2)
        if do_mirror:
            idx = idx - self.num_samples // 2

        if self.use_label:
            image_path, raw_label = self.items[idx][:2]
            raw_label = int(raw_label)
            label = raw_label_to_one_hot(raw_label, self.num_classes)
        else:
            image_path = self.items[idx]

        if self.use_pose:
            pose = self.get_pose(idx)

        # get bev information
        if self.dataset_name == 'carla':
            bev = np.load(io.BytesIO(self.fetch_file(image_path.replace('.png', '_bev.npy')))).transpose(2, 0, 1).astype(np.float32)
        else:
            bev = self.get_bev(idx)

        # Load image to buffer.
        buffer = np.frombuffer(self.fetch_file(image_path), dtype=np.uint8)

        idx = np.array(idx)
        do_mirror = np.array(do_mirror)
        if self.use_label:
            raw_label = np.array(raw_label)
            return [idx, do_mirror, buffer, pose, raw_label, label, bev]
        return [idx, do_mirror, buffer, pose, bev]

    @property
    def num_raw_outputs(self):
        if self.use_label:
            return 6  # [idx, do_mirror, buffer, raw_label, label, pose]
        return 4  # [idx, do_mirror, buffer, pose]

    def parse_transform_config(self):
        image_size = self.transform_kwargs.get('image_size')
        crop_size = self.transform_kwargs.get('crop_size')
        image_channels = self.transform_kwargs.setdefault('image_channels', 3)
        min_val = self.transform_kwargs.setdefault('min_val', -1.0)
        max_val = self.transform_kwargs.setdefault('max_val', 1.0)
        use_square = self.transform_kwargs.setdefault('use_square', True)
        center_crop = self.transform_kwargs.setdefault('center_crop', True)
        self.transform_config = dict(
            decode=dict(transform_type='Decode', image_channels=image_channels,
                        return_square=use_square, center_crop=center_crop),
            resize=dict(transform_type='Resize', image_size=image_size),
            normalize=dict(transform_type='Normalize',
                           min_val=min_val, max_val=max_val),
            centercrop=dict(transform_type='CenterCrop', crop_size=crop_size)
        )

    def transform(self, raw_data, use_dali=False):
        if self.use_label:
            idx, do_mirror, buffer, pose, raw_label, label, bev = raw_data
        else:
            idx, do_mirror, buffer, pose, bev = raw_data

        raw_image = self.transforms['decode'](buffer, use_dali=use_dali)
        raw_image = self.transforms['centercrop'](raw_image, use_dali=use_dali)
        raw_image = self.transforms['resize'](raw_image, use_dali=use_dali)
        raw_image = raw_image[..., :3]
        raw_image = self.mirror_aug(raw_image, do_mirror, use_dali=use_dali)
        image = self.transforms['normalize'](raw_image, use_dali=use_dali)

        if self.use_label:
            return [idx, raw_image, image, raw_label, label, pose, bev]
        return [idx, raw_image, image, pose, bev]

    @property
    def output_keys(self):
        if self.use_label:
            return ['index', 'raw_image', 'image', 'raw_label', 'label', 'pose', 'bev']
        return ['index', 'raw_image', 'image', 'pose', 'bev']

    def info(self):
        dataset_info = super().info()
        dataset_info['Dataset classes'] = self.dataset_classes
        dataset_info['Use label'] = self.use_label
        if self.use_label:
            dataset_info['Num classes for training'] = self.num_classes
        return dataset_info

if __name__ == '__main__':
    dataset = BEV3DDataset(root_dir='clevr.zip', annotation_format=None, bevlen=512,
            transform_kwargs={'image_size': 256, 'image_channels': 3, 'min_val': -1.0, 'max_val': 1.0, 'use_square': False, 'center_crop': False, 'crop_size': 160})
    from tqdm import trange
    from PIL import Image
    for i in trange(10):
        a = dataset.get_bev(i)
        b = dataset[i]['image']
        Image.fromarray((b.transpose(1,2,0)*128+128).astype(np.uint8)).save(f'tmp/{i}_img.png')
        Image.fromarray((a[0]*254).astype(np.uint8)).save(f'tmp/{i}_64_bev.png')
