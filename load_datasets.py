"""Load datasets for neural network model training and evaluation."""
from typing import Callable
import os
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_DATA_KWARGS = {"root": "datasets", "download": True}
_SHARED_TRANSFORMS = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : x/255.)])


def _build_data_loader(dataset: Callable, train: bool, transforms: Callable, batch_size: int):
    data = dataset(train=True, transform=transforms, **_DATA_KWARGS)
    return DataLoader(data, batch_size=batch_size, num_workers=os.cpu_count(), shuffle=train, drop_last=True)


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
        return cls.torch_dataset().classes



class FashionMNISTLoader(Dataset):
    
    def torch_dataset():
        return datasets.FashionMNIST

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

    # Torchvision data sets have inconsistent interfaces.
    # The CIFAR10 and CIFAR100 data sets do not have the 'classes' attribute.
    @classmethod
    def classes(cls):
        return ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]



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

    @classmethod 
    def classes(cls):
        return ["beaver", "dolphin", "otter", "seal", "whale",
                "aquarium fish", "flatfish", "ray", "shark", "trout",
                "orchids", "poppies", "roses", "sunflowers", "tulips",
                "bottles", "bowls", "cans", "cups", "plates",
                "apples", "mushrooms", "oranges", "pears", "sweet peppers",
                "clock", "computer keyboard", "lamp", "telephone", "television",
                "bed", "chair", "couch", "table", "wardrobe",
                "bee", "beetle", "butterfly", "caterpillar", "cockroach",
                "bear", "leopard", "lion", "tiger", "wolf",
                "bridge", "castle", "house", "road", "skyscraper",
                "cloud", "forest", "mountain", "plain", "sea",
                "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
                "fox", "porcupine", "possum", "raccoon", "skunk",
                "crab", "lobster", "snail", "spider", "worm",
                "baby", "boy", "girl", "man", "woman",
                "crocodile", "dinosaur", "lizard", "snake", "turtle",
                "hamster", "mouse", "rabbit", "shrew", "squirrel",
                "maple", "oak", "palm", "pine", "willow"
                "bicycle", "bus", "motorcycle", "pickup truck", "train",
                "lawn-mower", "rocket", "streetcar", "tank", "tractor"]



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
    def train_loader(cls):
        train_data = datasets.GTSRB(split="train", transform=cls.train_transforms(), **_DATA_KWARGS)
        return DataLoader(train_data, batch_size=256, num_workers=os.cpu_count(), shuffle=True, drop_last=True)

    def eval_loader(cls):
        eval_data = datasets.GTSRB(split="test", transform=cls.eval_transforms(), **_DATA_KWARGS)
        return DataLoader(eval_data, batch_size=1024, num_workers=os.cpu_count(), shuffle=False, drop_last=False)
