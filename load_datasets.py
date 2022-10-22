"""Load datasets for neural network model training and evaluation."""
from typing import Callable
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_DATA_KWARGS = {"root": "datasets", "download": True}
_EVAL_TRANSFORMS = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : x/255.)])
_SHARED_TRAIN_TRANSFORMS = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Lambda(lambda x : x/255.)])


def _build_data_loader(dataset: Callable, train: bool, transforms: Callable, batch_size: int):
    data = dataset(train=True, transform=transforms, **_DATA_KWARGS)
    return DataLoader(data, batch_size=batch_size, num_workers=os.cpu_count())


def load_fashionMNIST_data():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        _SHARED_TRAIN_TRANSFORMS
    ])
    train_loader = _build_data_loader(datasets.FashionMNIST, train=True, transforms=train_transforms, batch_size=32)
    eval_loader = _build_data_loader(datasets.FashionMNIST, train=False, transforms=_EVAL_TRANSFORMS, batch_size=1024)
    return train_loader, eval_loader


def load_CIFAR10_data():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        _SHARED_TRAIN_TRANSFORMS
    ])
    train_loader = _build_data_loader(datasets.CIFAR10, train=True, transforms=train_transforms, batch_size=32)
    eval_loader = _build_data_loader(datasets.CIFAR10, train=False, transforms=_EVAL_TRANSFORMS, batch_size=1024)
    return train_loader, eval_loader


def load_CIFAR100_data():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        _SHARED_TRAIN_TRANSFORMS
    ])
    train_loader = _build_data_loader(datasets.CIFAR100, train=True, transforms=train_transforms, batch_size=32)
    eval_loader = _build_data_loader(datasets.CIFAR100, train=False, transforms=_EVAL_TRANSFORMS, batch_size=1024)
    return train_loader, eval_loader
