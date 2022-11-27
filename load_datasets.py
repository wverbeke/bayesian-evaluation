"""Load datasets for neural network model training and evaluation."""
from typing import Callable
import os
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from mapillary_data_loader.load_mapillary import MapillaryDataset
from mapillary_data_loader.preproc_mapillary import TRAINING_PATCH_SIZE
from mapillary_data_loader.make_class_list import mapillary_class_list

DATA_DIRECTORY = "datasets"
_DATASET_KWARGS = {"root": DATA_DIRECTORY, "download": True}
_SHARED_TRANSFORMS = transforms.Compose([transforms.ToTensor()])
_DATALOADER_KWARGS = {"num_workers": os.cpu_count(), "prefetch_factor": 4}


def _build_data_loader_type_1(dataset: Callable, train: bool, transforms: Callable, batch_size: int):
    data = dataset(train=train, transform=transforms, **_DATASET_KWARGS)
    return DataLoader(data, batch_size=batch_size, shuffle=train, drop_last=train, **_DATALOADER_KWARGS)

def _build_data_loader_type_2(dataset: Callable, train: bool, transforms: Callable, batch_size: int):
    split = "train" if train else "test"
    data = dataset(split=split, transform=transforms, **_DATASET_KWARGS)
    return DataLoader(data, batch_size=batch_size, shuffle=train, drop_last=train, **_DATALOADER_KWARGS)

class Dataset:

    @staticmethod
    @abstractmethod
    def torch_dataset():
        """Dataset function from torchvision."""

    @staticmethod
    @abstractmethod
    def train_transforms():
        """Transformations for the training data set."""
        raise NotImplementedError()

    @staticmethod
    def eval_transforms():
        """Transformations for the evaluation data set."""
        return _SHARED_TRANSFORMS

    @staticmethod
    def train_batch_size():
        """Batch size for training."""
        return 32

    @classmethod
    def _build_data_loader(cls, train: bool, batch_size: int):
        try:
            return _build_data_loader_type_1(dataset=cls.torch_dataset(), train=train, transforms=cls.train_transforms(), batch_size=batch_size)
        except TypeError:
            return _build_data_loader_type_2(dataset=cls.torch_dataset(), train=train, transforms=cls.train_transforms(), batch_size=batch_size)

    @classmethod
    def train_loader(cls, batch_size=None):
        return cls._build_data_loader(train=True, batch_size=cls.train_batch_size() if batch_size is None else batch_size)

    @classmethod
    def eval_loader(cls):
        return cls._build_data_loader(train=False, batch_size=1024)

    @classmethod
    def classes(cls):
        try:
            return cls.torch_dataset().classes
        except AttributeError:
            return cls.torch_dataset()(**_DATASET_KWARGS).classes



class FashionMNISTLoader(Dataset):
    
    def torch_dataset():
        return datasets.FashionMNIST

    def train_transforms():
        return transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            _SHARED_TRANSFORMS
        ])


class MNISTLoader(Dataset):

    def torch_dataset():
        return datasets.MNIST

    def train_transforms():
        return transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            _SHARED_TRANSFORMS
        ])



class CIFAR10Loader(Dataset):

    def torch_dataset():
        return datasets.CIFAR10

    def train_transforms():
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            _SHARED_TRANSFORMS
        ])



class CIFAR100Loader(Dataset):
    
    def torch_dataset():
        return datasets.CIFAR100

    def train_transforms():
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(),
            _SHARED_TRANSFORMS
        ])



class GTSRBLoader(Dataset):

    def torch_dataset():
        return datasets.GTSRB

    def train_transforms():
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.Resize((56, 56)),
            transforms.RandomCrop(56, padding=4),
            _SHARED_TRANSFORMS
        ])

    def eval_transforms():
        return transforms.Compose([
            transforms.Resize((56, 56)),
            _SHARED_TRANSFORMS
        ])

    # We want to train with larger batch sizes because of the presence of rare classes.
    def train_batch_size():
        return 256

    # Pytorch interface for getting GTSRB class list seems not to exist.
    @classmethod
    def classes(cls):
        return ["speed_limit_20", "speed_limit_30", "speed_limit_50", "speed_limit_60", "speed_limit_70", "speed_limit_80", "end_of_speed_limit_80", "speed_limit_100", "speed_limit_120", "no_passing", "no_passing_over_3p5_tons", "priority_next_intersection", "priority_road", "yield", "stop", "no_vehicles", "no_vehicles_over_3p5_tons", "no_entry", "caution", "dangerous_curve_left", "dangerous_curve_right", "double_curve", "bumpy_road", "slippery_road", "road_narrows_right", "road_works", "traffic_signals", "pedestrians", "children_crossing", "bicycles_crossing", "ice_snow", "wild_animals", "end_of_all", "turn_right_ahead", "turn_left_ahead", "proceed_ahead", "proceed_ahead_or_right", "proceed_ahead_or_left", "proceed_right", "proceed_left", "roundabout", "end_of_no_passing", "end_of_no_passing_over_3p5_tons"]



class MapillaryLoader(Dataset):

    def torch_dataset():
        return MapillaryDataset

    def train_transforms():
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomCrop(TRAINING_PATCH_SIZE),
            _SHARED_TRANSFORMS
        ])

    def eval_transforms():
        return transforms.Compose([
            transforms.CenterCrop(TRAINING_PATCH_SIZE),
            _SHARED_TRANSFORMS
        ])

    # A large batch size is used to avoid forgetting of very rare classes.
    def train_batch_size():
        return 1024

    @classmethod
    def classes(cls):
        return mapillary_class_list()


class StanfordCarsLoader(Dataset):

    def torch_dataset():
        return datasets.StanfordCars

    def train_transforms():
        return transforms.Compose([
            transforms.Resize((244,244)),
            transforms.RandomRotation(15),
            transforms.RandomCrop((224,244)),
            transforms.RandomHorizontalFlip(),
            _SHARED_TRANSFORMS
        ])

    def eval_transforms():
        return transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.CenterCrop((224, 224)),
            _SHARED_TRANSFORMS
        ])
            

    # Larger batch size than number of classes.
    def train_batch_size():
        return 256


class Flowers102Loader(Dataset):

    def torch_dataset():
        return datasets.Flowers102

    def train_transforms():
        return transforms.Compose([
            # Keep aspect ratio by resizing short axis to 256
            transforms.Resize(256),
            transforms.RandomRotation(15),
            transforms.RandomCrop((224,244)),
            transforms.RandomHorizontalFlip(),
            _SHARED_TRANSFORMS
        ])

    def eval_transforms():
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            _SHARED_TRANSFORMS
        ])

    def train_batch_size():
        return 128

    #TODO Find class list, this is not available in the pytorch class for this data set.
    @classmethod
    def classes(cls):
        return list(range(102))
