from typing import Callable
from tqdm import tqdm
import math
import os

import torch
from torch import nn

from load_datasets import load_fashionMNIST_data
from build_neural_networks import SimpleCNN, Classifier
from data_tasks import MODEL_DIRECTORY, DEVICE, FashionMNISTTask, DataTask

class ModelTrainer:

    def __init__(self, loss_fn: Callable, optimizer: Callable, model: nn.Module):
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._model = model
        self._device = DEVICE
        self._epoch_counter = 1

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
        self._model.train()
        for x_batch, y_batch in tqdm(dataloader):
            self.train_step(x_batch, y_batch)

    def eval_step(self, x_batch, y_batch):
        """Evaluation of a single batch."""
        return self._forward_pass(x_batch, y_batch)

    def eval_epoch(self, dataloader):
        """A single evaluation epoch."""
        self._model.eval()
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
        print(f"Eval loss = {eval_loss:.3f}")
        self._epoch_counter += 1
        return eval_loss


class EarlyStopper:

    def __init__(self, tolerance: int):
        self._tolerance = tolerance
        self._fail_count = 0
        self._min_eval_loss = math.inf

    def __call__(self, new_eval_loss):
        if new_eval_loss < self._min_eval_loss:
            self._min_eval_loss = new_eval_loss
            self._fail_count = 0
            return True
        if self._fail_count < self._tolerance:
            self._fail_count += 1
            return True
        return False

    
#def train_model(data_loader_fn: Callable, model: nn.Module, model_name: str):
def train_model(data_task: DataTask):

    # Make the data laoders for the model.
    train_loader, eval_loader = data_task.load_data()
    model = data_task.build_model()

    # Train the model until the eval loss stops improving for more than 5 epochs.
    trainer = ModelTrainer(nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters()), model)
    callback = EarlyStopper(tolerance=5)
    while True:
        eval_loss = trainer.train_and_eval_epoch(train_loader, eval_loader)
        if not callback(eval_loss):
            break
        break

    # Save the model.
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)
    torch.save(model.state_dict(), data_task.model_path())



if __name__ == '__main__':
    # Test whether the fashionMNIST model can be trained.
    # It should converge relatively quickly.
    train_model(FashionMNISTTask)
