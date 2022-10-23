"""Load datasets for neural network model training and evaluation."""
from typing import Callable
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_DATA_KWARGS = {"root": "datasets", "download": True}
_SHARED_TRANSFORMS = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : x/255.)])


def _build_data_loader(dataset: Callable, train: bool, transforms: Callable, batch_size: int):
    data = dataset(train=True, transform=transforms, **_DATA_KWARGS)
    return DataLoader(data, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=train, drop_last=True)


def load_fashionMNIST_data():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        _SHARED_TRANSFORMS
    ])
    train_loader = _build_data_loader(datasets.FashionMNIST, train=True, transforms=train_transforms, batch_size=32)
    eval_loader = _build_data_loader(datasets.FashionMNIST, train=False, transforms=_SHARED_TRANSFORMS, batch_size=1024)
    return train_loader, eval_loader


def load_CIFAR10_data():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        _SHARED_TRANSFORMS
    ])
    train_loader = _build_data_loader(datasets.CIFAR10, train=True, transforms=train_transforms, batch_size=32)
    eval_loader = _build_data_loader(datasets.CIFAR10, train=False, transforms=_SHARED_TRANSFORMS, batch_size=1024)
    return train_loader, eval_loader


def load_CIFAR100_data():
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        _SHARED_TRANSFORMS
    ])
    train_loader = _build_data_loader(datasets.CIFAR100, train=True, transforms=train_transforms, batch_size=32)
    eval_loader = _build_data_loader(datasets.CIFAR100, train=False, transforms=_SHARED_TRANSFORMS, batch_size=1024)
    return train_loader, eval_loader


def load_GTSRB_data():
    train_transforms = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.Resize((56, 56)),
        transforms.RandomCrop(56, padding=4),
        _SHARED_TRANSFORMS
    ])
    # Because of inconsistent pytorch interfaces we can not reuse _build_data_loader here.
    train_data = datasets.GTSRB(split="train", transform=train_transforms, **_DATA_KWARGS)
    train_loader = DataLoader(train_data, batch_size=256, num_workers=os.cpu_count(), shuffle=True, drop_last=True)
    eval_data = datasets.GTSRB(split="test", transform=_SHARED_TRANSFORMS, **_DATA_KWARGS)
    eval_loader = DataLoader(eval_data, batch_size=1024, num_workers=os.cpu_count(), shuffle=False, drop_last=False)
    return train_loader, eval_loader
