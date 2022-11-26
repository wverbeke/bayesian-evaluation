""" Implementation of the statistical model used to constrain the performance of a classifier.

Neural network models are assumed to be trained and are evaluated in terms of their confusion matrix.

The statistical model consists of three stages:
1. A hyperprior generates parameters that determines the prior of the performance for a particular class.
2. The prior sampled from the hyperprior is used together with the likelihood to form a posterior for the performance of a class.
3. A multinomial likelihood models the observed counts in the confusion matrix for a given class.
"""
import argparse
from typing import List, Union, Optional
import os
from abc import abstractmethod

import numpy as np
import pymc as pm
import xarray as xr

from confusion_matrix import BinaryCM, convert_to_binary, divide_safe
from data_tasks import TASK_REGISTER, DataTask, find_task, get_task_names
from plot_metrics import plot_posterior_comparison
from bayesian_models import BayesianModel, BAYESIAN_MODEL_REGISTER, get_bayesian_model_names, find_bayesian_model, PLOT_DIRECTORY

def compute_recalls(cm_array: np.ndarray) -> np.ndarray:
    """Compute the recalls for an array of sampled confusion matrices."""
    tp_array = cm_array[:, 0]
    fn_array = cm_array[:, 2]
    return divide_safe(tp_array, tp_array + fn_array)


def compute_precisions(cm_array: np.ndarray) -> np.ndarray:
    """Compute the precisions for an array of sampled confusion matrices."""
    tp_array = cm_array[:, 0]
    fp_array = cm_array[:, 1]
    return divide_safe(tp_array, tp_array + fp_array)


def plot_posterior_metrics(bayesian_models):
    if len(bayesian_models) < 2:
        raise ValueError("There should be at least two Bayesian models to compared.")

    # Verify that all Bayesian models have the same classes and can be compared.
    classes = bayesian_models[0].data_task.classes()
    if not all((b.data_task.classes() == classes) for b in bayesian_models[1:]):
        raise ValueError("All Bayesian models being compared should have the same list of classes.")

    # Verify that all posterior samples exist.
    if not all(b.posterior_samples_exist() for b in bayesian_models):
        raise ValueError("Can only plot the posterior distributions for the metrics when the posterior samples are available.")


    # All models are assumed to operate on the same task.
    model_names = [b.name() for b in bayesian_models]
    task_name = bayesian_models[0].data_task.name()
    os.makedirs(os.path.join(PLOT_DIRECTORY, task_name), exist_ok=True)


    for class_index, class_name in enumerate(classes):
        recall_arrays = []
        precision_arrays = []
        for b in bayesian_models:
            cm_array = b.load_posterior_samples(class_index=class_index)
            recall_array = compute_recalls(cm_array)
            recall_arrays.append(recall_array)
            precision_array = compute_precisions(cm_array)
            precision_arrays.append(precision_array)

        plot_posterior_comparison(
            model_posteriors=recall_arrays,
            model_names=model_names,
            plot_path=os.path.join(PLOT_DIRECTORY, task_name, f"{task_name}_recall_class_{class_index}"),
            metric_name="Recall",
            task_name=task_name,
            class_name=classes[class_index],

            # Given that all models work on the same task, they have the same underlying observed confusion matrix
            # and class counts.
            num_class_samples=b.num_samples_per_class(class_index),
            observed=b.observed_binary_cm(class_index).recall()
        )
        plot_posterior_comparison(
            model_posteriors=precision_arrays,
            model_names=model_names,
            plot_path=os.path.join(PLOT_DIRECTORY, task_name, f"{task_name}_precision_class_{class_index}"),
            metric_name="Precision",
            task_name=task_name,
            class_name=class_name,
            num_class_samples=b.num_samples_per_class(class_index),
            observed=b.observed_binary_cm(class_index).precision()
        )

def parse_args():
    """Command line arguments for running the evaluation of the Bayesian models."""
    parser = argparse.ArgumentParser(description="Arguments for the evaluation of Bayesian models used to descripe the performance metrics of neural networks.")
    parser.add_argument("--num-samples-per-core", type=int, default=2000, help="Number of elements in the Markov chain generated on each cpu core to evaluate the Bayesian model.")
    parser.add_argument("--reevaluate", action="store_true", help="Whether to rerun the evaluation of models that have an existing set of sampled Markov chains or posterios samples. By default existing evaluation results will be reused.")
    parser.add_argument("--tasks", choices=get_task_names(), nargs="+", help="Only evaluate the bayesian model for a specific data task. By default all tasks are evaluated.")
    parser.add_argument("--bayesian-models", choices=get_bayesian_model_names(), nargs="+", help="Only evaluate the particular Bayesian models specified here. By default all Bayesian models are evaluated.")

    return parser.parse_args()


if __name__ == "__main__":
    # Parse the command line arguments.
    args = parse_args()

    # Loop over all requested data tasks and Bayesian models and evaluate the results.
    if args.tasks:
        tasks_to_evaluate = [find_task(t) for t in args.tasks]
    else:
        tasks_to_evaluate = TASK_REGISTER
    if args.bayesian_models:
        print(args.bayesian_models)
        models_to_evaluate = args.bayesian_models
    else:
        models_to_evaluate = BAYESIAN_MODEL_REGISTER

    for task in tasks_to_evaluate:
        print("#"*50)
        print(f"Evaluating bayesian models for {task.name()}")
        b_models = []
        for model_class in BAYESIAN_MODEL_REGISTER:
            print(f"Sampling for {model_class.name()}")
            bm = model_class(task)
            b_models.append(bm)
            if not bm.trace_exists() or args.reevaluate:
                trace = bm.trace(num_samples_per_core=args.num_samples_per_core)
                bm.sample_posterior_predictive(trace)
            elif not bm.posterior_samples_exist():
                bm.sample_posterior_predictive(bm.load_trace())
        plot_posterior_metrics(b_models)
