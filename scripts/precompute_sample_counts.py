import os
import sys
main_directory = os.path.dirname(os.path.dirname(os.path.abspath( __file__)))
sys.path.insert(0, main_directory)

from collections import Counter
import json
import torch
from data_tasks import TASK_REGISTER, SAMPLE_COUNTS_DIRECTORY

def count_train_samples(task):

    # Use a batch size of 1 to avoid the drop_remainder dropping samples.
    try:
        train_loader = task.data_loader().train_loader(batch_size=1)

    # For synthetic data there is no loader.
    except NotImplementedError:
        return

    # Only the labels matter.
    # Count the occurences of each label.
    sample_counts = Counter()
    for _, y_batch in train_loader:
        sample_counts.update([int(y_batch)])

    # Dump the class counts to a JSON file
    json_path = task.train_counts_path()
    with open(json_path, "w") as f:
        json.dump(sample_counts, f)
    
if __name__ == "__main__":
    # To ensure pytorch does not trigger an error about too many open files.
    torch.multiprocessing.set_sharing_strategy("file_system")

    # Ensure the directory where the counts are dumped exists.
    os.makedirs(SAMPLE_COUNTS_DIRECTORY, exist_ok=True)

    # Compute the training class counts for each task
    for t in TASK_REGISTER:
        count_train_samples(t)
