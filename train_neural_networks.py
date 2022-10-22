from typing import Callable
from tqdm import tqdm

import torch
from torch import nn

from load_datasets import load_fashionMNIST_data
from build_neural_networks import SimpleCNN, Classifier

DEVICE = ("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer:

    def __init__(self, loss_fn: Callable, optimizer: Callable, model: nn.Module):
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._model = model
        self._device = DEVICE
        self._epoch_counter = 0

    def _to_device(self, x_batch, y_batch):
        """Move a batch to the GPU if available."""
        x_batch = x_batch.to(self._device)
        y_batch = y_batch.to(self._device)
        return x_batch, y_batch

    def _forward_pass(self, x_batch, y_batch):
        """Forward pass and loss calculation."""
        x_batch, y_batch = self._to_device(x_batch, y_batch)
        pred = self._model(x_batch)
        loss = self._loss_fn(pred, y_batch)
        return loss

    def train_step(self, x_batch, y_batch):
        """Apply a single training batch."""
        loss = self._forward_pass(x_batch, y_batch)

        # Backpropagation
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def train_epoch(self, dataloader):
        """A single training epoch."""
        for x_batch, y_batch in tqdm(dataloader):
            self.train_step(x_batch, y_batch)

    def eval_step(self, x_batch, y_batch):
        """Evaluation of a single batch."""
        return self._forward_pass(x_batch, y_batch)

    def eval_epoch(self, dataloader):
        """A single evaluation epoch."""
        total_eval_loss = 0
        num_batches = len(dataloader)
        with torch.no_grad():
            for x_batch, y_batch in tqdm(dataloader):
                loss = self.eval_step(x_batch, y_batch)
                total_eval_loss += loss.item()
        avg_eval_loss = total_eval_loss/num_batches
        return avg_eval_loss

    def train_and_eval_epoch(self, train_loader, eval_loader):
        """Do a training and evaluation epoch and print information."""
        print("-"*100)
        print(f"Epoch {self._epoch_counter}")
        print("Train:")
        self.train_epoch(train_loader)
        print("Eval:")
        eval_loss = self.eval_epoch(eval_loader)
        print(f"Eval loss = {eval_loss}")



if __name__ == '__main__':
    # Test ModelTrainer on fashionMNIST.
    train_loader, eval_loader = load_fashionMNIST_data()
    simple_cnn = Classifier(SimpleCNN(in_channels=1), num_classes=10).to(DEVICE)
    trainer = ModelTrainer(nn.CrossEntropyLoss(), torch.optim.Adam(simple_cnn.parameters()), simple_cnn)

    for i in range(5):
        trainer.train_and_eval_epoch(train_loader, eval_loader)

