import os
from abc import abstractmethod

import torch

from build_neural_networks import Resnet18, SimpleCNN, Classifier
from load_datasets import load_fashionMNIST_data, load_CIFAR10_data, load_CIFAR100_data, load_GTSRB_data

MODEL_DIRECTORY = "trained_models"
EVAL_DIRECTORY = "confusion_matrices"
DEVICE_CPU = "cpu"
DEVICE_GPU = "cuda"
DEVICE = (DEVICE_GPU if torch.cuda.is_available() else DEVICE_CPU)

class DataTask:

    @staticmethod
    @abstractmethod
    def num_classes():
        """Get the number of classes in the data set."""

    @staticmethod
    @abstractmethod
    def name():
        """Name of the task.

        This will determine the name of all output files from training and evaluation.
        """

    @classmethod
    def model_path(cls):
        model_name = f"{cls.name()}_model"
        return os.path.join(MODEL_DIRECTORY, model_name)

    @classmethod
    def confusion_matrix_path(cls):
        cm_name = f"{cls.name()}_confusion_matrix.npy"
        return os.path.join(EVAL_DIRECTORY, cm_name)

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
        return model


task_register = []
def register_task(cls):
    task_register.append(cls)
    return cls


@register_task
class FashionMNISTTask(DataTask):

    def num_classes():
        return 10

    def name():
        return "fashionMNIST"

    def _build_model():
        return SimpleCNN(in_channels=1)

    def load_data():
        return load_fashionMNIST_data()



@register_task
class CIFAR10Task(DataTask):

    def num_classes():
        return 10

    def name():
        return "CIFAR10"

    def _build_model():
        return Resnet18(in_channels=3)

    def load_data():
        return load_CIFAR10_data()



@register_task
class CIFAR100Task(DataTask):

    def num_classes():
        return 100

    def name():
        return "CIFAR100"

    def _build_model():
        return Resnet18(in_channels=3)

    def load_data():
        return load_CIFAR100_data()



@register_task
class GTSRBTask(DataTask):

    def num_classes():
        return 43

    def name():
        return "GTSRB"

    def _build_model():
        return Resnet18(in_channels=3)

    def load_data():
        return load_GTSRB_data()
