import os

import torch
from torchmetrics.classification import MulticlassConfusionMatrix
import numpy as np


from data_tasks import DataTask, task_register, DEVICE, DEVICE_CPU, EVAL_DIRECTORY

def compute_confusion_matrix(data_task: DataTask):

    # Load the model.
    model = data_task.load_model()
    model = model.eval()

    # Data loader.
    _, eval_data_loader = data_task.load_data()

    # Run inference over the evaluation data.
    predictions = []
    labels = []
    for x, y in eval_data_loader:
        x = x.to(DEVICE)
        model_out = model(x)
        model_out.to(DEVICE_CPU)
        predictions.append(model_out)
        labels.append(y)
    
    # Compute the confusion matrix.
    prediction_tensor = torch.cat(predictions, dim=0).cpu()
    label_tensor = torch.cat(labels, dim=0).cpu()
    cm_computer = MulticlassConfusionMatrix(num_classes=data_task.num_classes()).to(DEVICE_CPU)
    cm = cm_computer(prediction_tensor, label_tensor)
    return cm



if __name__ == "__main__":
    os.makedirs(EVAL_DIRECTORY, exist_ok=True)
    for task in task_register:
        print(f"Computing confusion matrix for {task.name()} task.")
        cm = compute_confusion_matrix(task)
        np.save(task.confusion_matrix_path(), cm.numpy())
