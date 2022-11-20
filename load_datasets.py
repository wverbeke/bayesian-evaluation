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


def _build_data_loader(dataset: Callable, train: bool, transforms: Callable, batch_size: int):
    data = dataset(train=train, transform=transforms, **_DATASET_KWARGS)
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
    def train_loader(cls):
        return _build_data_loader(dataset=cls.torch_dataset(), train=True, transforms=cls.train_transforms(), batch_size=cls.train_batch_size())

    @classmethod
    def eval_loader(cls):
        return _build_data_loader(dataset=cls.torch_dataset(), train=False, transforms=cls.eval_transforms(), batch_size=1024)

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

    # For GTSRB the data loading methods must be overridden because the interface is different.
    @classmethod
    def train_loader(cls):
        train_data = cls.torch_dataset()(split="train", transform=cls.train_transforms(), **_DATASET_KWARGS)
        return DataLoader(train_data, batch_size=256, shuffle=True, drop_last=True, **_DATALOADER_KWARGS)

    @classmethod
    def eval_loader(cls):
        eval_data = cls.torch_dataset()(split="test", transform=cls.eval_transforms(), **_DATASET_KWARGS)
        return DataLoader(eval_data, batch_size=1024, shuffle=False, drop_last=False, **_DATALOADER_KWARGS)



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
