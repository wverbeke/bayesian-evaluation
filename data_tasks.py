import os
from abc import abstractmethod

import torch

from build_neural_networks import Resnet18, SimpleCNN, Classifier
#from load_datasets import load_fashionMNIST_data, load_CIFAR10_data, load_CIFAR100_data, load_GTSRB_data
from load_datasets import FashionMNISTLoader, CIFAR10Loader, CIFAR100Loader, GTSRBLoader

MODEL_DIRECTORY = "trained_models"
EVAL_DIRECTORY = "confusion_matrices"
DEVICE_CPU = "cpu"
DEVICE_GPU = "cuda"
DEVICE = (DEVICE_GPU if torch.cuda.is_available() else DEVICE_CPU)

class DataTask:

    @staticmethod
    @abstractmethod
    def data_loader():
        """Get the data loader class for this task."""
        raise NotImplementedError()

    @classmethod
    def classes(cls):
        return cls.data_loader().classes()

    @classmethod
    def num_classes(cls):
        """Get the number of classes in the data set."""
        return len(cls.classes())

    @staticmethod
    @abstractmethod
    def name():
        """Name of the task.

        This will determine the name of all output files from training and evaluation.
        """
        raise NotImplementedError()

    @classmethod
    def model_path(cls):
        model_name = f"{cls.name()}_model"
        return os.path.join(MODEL_DIRECTORY, model_name)

    @classmethod
    def confusion_matrix_path(cls):
        cm_name = f"{cls.name()}_confusion_matrix.npy"
        return os.path.join(EVAL_DIRECTORY, cm_name)

    @staticmethod
    def _build_model():
        """Build the neural network solving the task."""
        return Resnet18(in_channels=3)

    @classmethod
    def load_data(cls):
        """Return the training and evaluation data loaders."""
        train_loader = cls.data_loader().train_loader()
        eval_loader = cls.data_loader().eval_loader()
        return train_loader, eval_loader

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


# Register all the individual data sets and their training/evaluatin tasks.
# This makes it easy to train all models in a loop later on and collect the results.
TASK_REGISTER = []
def register_task(cls):
    TASK_REGISTER.append(cls)
    return cls


@register_task
class FashionMNISTTask(DataTask):

    def data_loader():
        return FashionMNISTLoader

    def name():
        return "fashionMNIST"

    # The model should be overriden for FashionMNIST because of the number of input channels.
    # We also use a simpler model for this small data set than a ResNet.
    def _build_model():
        return SimpleCNN(in_channels=1)


@register_task
class CIFAR10Task(DataTask):

    def data_loader():
        return CIFAR10Loader

    def name():
        return "CIFAR10"


@register_task
class CIFAR100Task(DataTask):

    def data_loader():
        return CIFAR100Loader

    def name():
        return "CIFAR100"


@register_task
class GTSRBTask(DataTask):

    def data_loader():
        return GTSRBLoader

    def name():
        return "GTSRB"
