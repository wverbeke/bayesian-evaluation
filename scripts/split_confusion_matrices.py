import os
import sys
main_directory = os.path.dirname(os.path.dirname(os.path.abspath( __file__)))
sys.path.insert(0, main_directory)

import numpy as np
import argparse

from data_tasks import TASK_REGISTER, FIT_CM_DIRECTORY, TEST_CM_DIRECTORY
from confusion_matrix import check_valid_confusion_matrix


def random_split_confusion_matrix(confusion_matrix, test_fraction: float=0.5):
    """Split a confusion matrix into two, one for fitting Bayesian models and one for testing.
    
    The split is done by sampling from a binomial for each position in the original confusion
    matrix to sample a submatrix. The remainder is the other matrix.
    """
    check_valid_confusion_matrix(confusion_matrix)
    if test_fraction <= 0 or test_fraction >= 1:
        raise ValueError(f"test_fraction must be between 0 and 1, but is {test_fraction}.")
    # No attempt is made to vectorize this operation since looping over the elements of a single
    # matrix is fast.
    test_cm = np.zeros_like(confusion_matrix)
    fit_cm = np.zeros_like(confusion_matrix)
    for i in range(test_cm.shape[0]):
        for j in range(test_cm.shape[1]):
            original_count = confusion_matrix[i, j]
            test_count = np.random.binomial(n=original_count, p=test_fraction)
            test_cm[i, j] = test_count

            fit_count = original_count - test_count
            fit_cm[i, j] = fit_count
    return fit_cm, test_cm

def split_and_save_confusion_matrix(data_task, test_fraction=0.5):
    """For a given data task split the stored confusion matrix and save the splits."""
    fit_cm, test_cm = random_split_confusion_matrix(confusion_matrix=data_task.confusion_matrix(), test_fraction=test_fraction)
    np.save(data_task.fit_confusion_matrix_path(), fit_cm)
    np.save(data_task.test_confusion_matrix_path(), test_cm)

def parse_args():
    """Command line arguments to make the script more flexible."""
    parser = argparse.ArgumentParser(description="Arguments for splitting confusion matrices into fit and test matrices.")
    parser.add_argument("--test-fraction", type=float, default=0.5, help="Fraction of the confusion matrix used for testing. The rest is used for fitting Bayesian models.")
    parser.add_argument("--seed", type=int, default=69, help="Random seed used for splitting the confusion matrices.")
    return parser.parse_args()


if __name__ == "__main__":
    # Read the command line arguments
    args = parse_args()

    # Ensure consistent results by setting the random seed.
    np.random.seed(args.seed)

    os.makedirs(FIT_CM_DIRECTORY, exist_ok=True)
    os.makedirs(TEST_CM_DIRECTORY, exist_ok=True)
    for t in TASK_REGISTER:
        split_and_save_confusion_matrix(data_task=t, test_fraction=args.test_fraction)
