import os
from abc import abstractmethod

import torch

from build_neural_networks import Resnet18, SimpleCNN, Classifier
from load_datasets import load_fashionMNIST_data, load_CIFAR10_data, load_CIFAR100_data, load_GTSRB_data

MODEL_DIRECTORY = "trained_models"
DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

class DataTask:

    @staticmethod
    @abstractmethod
    def num_classes():
        """Get the number of classes in the data set."""

    @staticmethod
    @abstractmethod
    def saved_model_name():
        """Path to which the model will be loaded."""

    @classmethod
    def model_path(cls):
        return os.path.join(MODEL_DIRECTORY, cls.saved_model_name())

    @staticmethod
    @abstractmethod
    def _build_model():
        """Build the neural network solving the task."""

    @staticmethod
    @abstractmethod
    def load_data():
        """Return the training and evaluation data loaders."""

    @classmethod
    def build_model(cls):
        model = Classifier(body=cls._build_model(), num_classes=cls.num_classes())

        # Place the model on GPU if available.
        model = model.to(DEVICE)
        return model

    @classmethod
    def load_model(cls):
        if not os.path.isfile(cls.model_path()):
            raise ValueError(f"Can not load model {cls.model_path()} because it does not exist.")
        model = cls.build_model()
        model.load_state_dict(torch.load(cls.model_path()))



class FashionMNISTTask(DataTask):

    def num_classes():
        return 10

    def saved_model_name():
        return "fashionMNIST_model"

    def _build_model():
        return SimpleCNN(in_channels=1)

    def load_data():
        return load_fashionMNIST_data()


class CIFAR10Task(DataTask):

    def num_classes():
        return 10

    def saved_model_name():
        return "CIFAR10_model"

    def _build_model():
        return Resnet18(in_channels=3)

    def load_data():
        return load_CIFAR10_data()



class CIFAR100Task(DataTask):

    def num_classes():
        return 100

    def saved_model_name():
        return "CIFAR100_model"

    def _build_model():
        return Resnet18(in_channels=3)

    def load_data():
        return load_CIFAR100_data()



class GTSRBTask(DataTask):

    def num_classes():
        return 43

    def saved_model_name():
        return "GTSRB_model"

    def _build_model():
        return Resnet18(in_channels=3)

    def load_data():
        return load_GTSRB_data()
