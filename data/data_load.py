import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


# Dataset label naming conventions
DATASET_CONFIGS = {
    'ITS': {
        'label_suffix': '.png',
        'crop_size': 256,
    },
    'Haze4K': {
        'label_suffix': '.png',
        'crop_size': 256,
    },
    'DenseHaze': {
        'label_suffix': '_GT.png',
        'crop_size': (600, 800),
    },
    'NH-HAZE': {
        'label_suffix': '_GT.png',
        'crop_size': (600, 600),
    },
    'O-HAZE': {
        'label_suffix': '_outdoor_GT.jpg',
        'crop_size': 512,
    },
    'FoggyCityscapes': {
        'label_suffix': None,  # Special handling
        'crop_size': (600, 800),
    },
}


def get_label_filename(image_name, dataset_type):
    """Get the corresponding label filename based on dataset type."""
    config = DATASET_CONFIGS.get(dataset_type, DATASET_CONFIGS['ITS'])

    if dataset_type == 'ITS' or dataset_type == 'Haze4K':
        # ITS/Haze4K: 275_2_0.94076.png -> 275.png
        # Extract the first number before the first underscore
        base_name = image_name.split('_')[0]
        return base_name + '.png'
    elif dataset_type == 'DenseHaze' or dataset_type == 'NH-HAZE':
        # DenseHaze/NH-HAZE: 01_hazy.png -> 01_GT.png
        base_name = image_name.split('_')[0]
        return base_name + config['label_suffix']
    elif dataset_type == 'O-HAZE':
        # O-HAZE: 01_outdoor_hazy.png -> 01_outdoor_GT.jpg
        base_name = image_name.replace('_hazy', '')
        return base_name.replace('.png', '_GT.jpg')
    else:
        return image_name


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True, dataset_type='ITS'):
    image_dir = os.path.join(path, 'train')

    config = DATASET_CONFIGS.get(dataset_type, DATASET_CONFIGS['ITS'])
    crop_size = config['crop_size']

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                PairRandomCrop(crop_size),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(
        DeblurDataset(image_dir, transform=transform, dataset_type=dataset_type),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0, dataset_type='ITS'):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(
        DeblurDataset(image_dir, is_test=True, dataset_type=dataset_type),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0, dataset_type='ITS'):
    dataloader = DataLoader(
        DeblurDataset(os.path.join(path, 'test'), dataset_type=dataset_type),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False, dataset_type='ITS'):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'hazy'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test
        self.dataset_type = dataset_type

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'hazy', self.image_list[idx]))

        # Get corresponding label filename
        label_name = get_label_filename(self.image_list[idx], self.dataset_type)
        label = Image.open(os.path.join(self.image_dir, 'clear', label_name))

        if self.transform:
            image, label = self.transform(image, label)
        else:
            image = F.to_tensor(image)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, label, name
        return image, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1].lower() not in ['png', 'jpg', 'jpeg']:
                print("Warning: unsupported image format:", x)
