import argparse
from typing import Callable
from tqdm import tqdm
import math
import os
from enum import Enum

import torch
from torch import nn

from build_neural_networks import SimpleCNN, Classifier
from data_tasks import MODEL_DIRECTORY, DEVICE, DataTask, TASK_REGISTER

class ModelTrainer:
    """Class collecting all the functionality to train and evaluate a neural network model."""
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

        return loss

    def train_epoch(self, dataloader):
        """A single training epoch."""
        total_train_loss = 0
        num_batches = 0
        self._model.train()
        for x_batch, y_batch in tqdm(dataloader):
            loss = self.train_step(x_batch, y_batch)
            total_train_loss += loss
            num_batches += 1
        return (total_train_loss/num_batches)

    def eval_step(self, x_batch, y_batch):
        """Evaluation of a single batch."""
        return self._forward_pass(x_batch, y_batch)

    def eval_epoch(self, dataloader):
        """A single evaluation epoch."""
        self._model.eval()

        # Note that the last batch might be smaller than the rest, so it is better to count the
        # number of individual samples.
        num_samples = 0
        total_eval_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in tqdm(dataloader):
                loss = self.eval_step(x_batch, y_batch)
                num_samples += len(y_batch)
                total_eval_loss += loss.item()*len(y_batch)

        avg_eval_loss = total_eval_loss/num_samples
        return avg_eval_loss

    def train_and_eval_epoch(self, train_loader, eval_loader):
        """Do a training and evaluation epoch and print information."""
        print("-"*100)
        print(f"Epoch {self._epoch_counter}")
        print("Train:")
        train_loss = self.train_epoch(train_loader)
        print(f"Train loss = {train_loss:.3f}")
        print("Eval:")
        eval_loss = self.eval_epoch(eval_loader)
        print(f"Eval loss = {eval_loss:.3f}")
        self._epoch_counter += 1
        return eval_loss



class CallbackResult(Enum):
    """Results used in the convergence check."""
    NEW_BEST = 0
    WORSE = 1
    STOP = 2



class EarlyStopper:
    """Convergence criterion for model training.

    If the eval loss has not improved for a given number of training epochs, the training is
    considered to have congerged.
    """
    def __init__(self, tolerance: int):
        self._tolerance = tolerance
        self._fail_count = 0
        self._min_eval_loss = math.inf

    def __call__(self, new_eval_loss):
        if new_eval_loss < self._min_eval_loss:
            self._min_eval_loss = new_eval_loss
            self._fail_count = 0
            return CallbackResult.NEW_BEST
        if self._fail_count < self._tolerance:
            self._fail_count += 1
            return CallbackResult.WORSE
        return CallbackResult.STOP

    
def train_model(data_task: DataTask):
    """Train a neural network model until the convergence criterion is achieved."""
    # Make the data laoders for the model.
    try:
        train_loader, eval_loader = data_task.load_data()

    # The end goal of the repository is to evaluate Bayesian models. Some tasks involve no neural
    # network, and these will raise a NotImplemntedError when calling load_data().
    except NotImplementedError:
        return
    model = data_task.build_model()

    # Directory where the trained model will be stored.
    os.makedirs(MODEL_DIRECTORY, exist_ok=True)

    # Train the model until the eval loss stops improving for more than 5 epochs.
    trainer = ModelTrainer(nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=1e-3), model)
    callback = EarlyStopper(tolerance=7)
    while True:
        eval_loss = trainer.train_and_eval_epoch(train_loader, eval_loader)
        callback_result = callback(eval_loss)
        if callback_result == CallbackResult.NEW_BEST:

            # Save the model.
            torch.save(model.state_dict(), data_task.model_path())
        elif callback_result == CallbackResult.STOP:
            break


def parse_args():
    """Command line arguments to make the script more flexible."""
    parser = argparse.ArgumentParser(description="Arguments for training neural networks to solve some basic classification tasks.")
    parser.add_argument("--retrain", action="store_true", help="Whether to retrain the neural networks for data tasks that already have a saved model or not.")
    possible_tasks=[t.name() for t in TASK_REGISTER]
    parser.add_argument("--task", choices=possible_tasks, help="Only train the neural network for a specific data task. By default all neural networks are trained unless they already have a trained model.")
    return parser.parse_args()


if __name__ == '__main__':
    # Read the command line arguments
    args = parse_args()

    # Train model for a single task.
    if args.task:
        task_index = [t.name() for t in TASK_REGISTER].index(args.task)
        task = TASK_REGISTER[task_index]
        print(f"Training task {task.name()}")
        if not task.model_exists() or args.retrain:
            train_model(TASK_REGISTER[task_index])

    # Train models for all tasks.
    for task in TASK_REGISTER:
        if task.model_exists() and (not args.retrain): continue
        print(f"Training task {task.name()}")
        train_model(task)
