"""Load datasets for neural network model training and evaluation."""
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_DATA_KWARGS = {"root": "datasets", "download": True, "transform": transforms.ToTensor()}


def _build_data_loader(dataset: Callable, train: bool, batch_size: int):
    data = dataset(train=True, **_DATA_KWARGS)
    dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader
    

def load_fashionMNIST_data():
    train_loader = _build_data_loader(datasets.FashionMNIST, train=True, batch_size=32)
    test_loader = _build_data_loader(datasets.FashionMNIST, train=False, batch_size=1024)
    return train_loader, test_loader


def load_CIFAR10_data():
    train_loader = _build_data_loader(datasets.CIFAR10, train=True, batch_size=32)
    test_loader = _build_data_loader(datasets.CIFAR10, train=False, batch_size=1024)
    return train_loader, test_loader


def load_CIFAR100_data():
    train_loader = _build_data_loader(datasets.CIFAR100, train=True, batch_size=32)
    test_loader = _build_data_loader(datasets.CIFAR100, train=False, batch_size=1024)
    return train_loader, test_loader
