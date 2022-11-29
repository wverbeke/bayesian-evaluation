import os

import torch
from torchmetrics.classification import MulticlassConfusionMatrix
import numpy as np


from data_tasks import DataTask, TASK_REGISTER, DEVICE, DEVICE_CPU, CM_DIRECTORY

def compute_confusion_matrix(data_task: DataTask):
    """Compute the confusion matrix on the evaluation data for a given task and its model.

    The confusion matrix will be returned in the usual format where true labels are columns and
    predictions are rows.
    """

    # Load the model.
    model = data_task.load_model()
    model = model.eval()

    # Data loader.
    _, eval_data_loader = data_task.load_data()

    # Run inference over the evaluation data.
    predictions = []
    labels = []
    with torch.no_grad():
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

    # Pytorch uses the opposite convention from what is usual, i.e. the predictions are columns and 
    # the true labels rows. We want it in the usual format where the true labels are columns.
    cm = np.transpose(cm)
    return cm



if __name__ == "__main__":
    os.makedirs(CM_DIRECTORY, exist_ok=True)
    for task in TASK_REGISTER:
        print(f"Computing confusion matrix for {task.name()} task.")
        cm = compute_confusion_matrix(task)
        np.save(task.confusion_matrix_path(), cm.numpy())
