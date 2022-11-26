import os
import sys
main_directory = os.path.dirname(os.path.dirname(os.path.abspath( __file__)))
sys.path.insert(0, main_directory)

from collections import Counter
import json
import torch
from data_tasks import TASK_REGISTER

SAMPLE_COUNTS_DIRECTORY="sample_counts"
TRAINING_COUNTS_FILE_NAME = "training_class_counts"

def _training_counts_path(task):
    return os.path.join(SAMPLE_COUNTS_DIRECTORY, f"{task.name()}_{TRAINING_COUNTS_FILE_NAME}.json")

def count_training_samples(task):

    # Use a batch size of 1 to avoid the drop_remainder dropping samples.
    train_loader = task.data_loader().train_loader(batch_size=1)

    # Only the labels matter.
    # Count the occurences of each label.
    sample_counts = Counter()
    for _, y_batch in train_loader:
        sample_counts.update([int(y_batch)])

    # Dump the class counts to a JSON file
    json_path = _training_counts_path(task)
    with open(json_path, "w") as f:
        json.dump(sample_counts, f)
    
if __name__ == "__main__":
    # To ensure pytorch does not trigger an error about too many open files.
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Ensure the directory where the counts are dumped exists.
    os.makedirs(SAMPLE_COUNTS_DIRECTORY, exist_ok=True)

    # Compute the training class counts for each task
    for t in TASK_REGISTER:
        count_training_samples(t)
