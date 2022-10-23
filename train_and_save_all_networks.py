from build_neural_network_models import Resnet18, SimpleCNN, Classifier
from train_neural_networks import train_model


def _build_fashionMNIST_model():
    body = SimpleCNN(in_channels=1)
    return Classifier(body=body, num_classes=10)


def _build_CIFAR10_model():
    body = Resnet18(in_channels=3)
    return Classifier(body=body, num_classes=10)


def _build_CIFAR100_model():
    body=Resnet18(in_channels=3)
    return Classifier(body=body, num_classes=100)


def _build_GTSRB_model():
    body=Resnet18(in_channels=3)
    return Classifier(body=body, num_classes=43)


