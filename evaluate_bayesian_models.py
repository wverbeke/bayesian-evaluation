""" Implementation of the statistical model used to constrain the performance of a classifier.

Neural network models are assumed to be trained and are evaluated in terms of their confusion matrix.

The statistical model consists of three stages:
1. A hyperprior generates parameters that determines the prior of the performance for a particular class.
2. The prior sampled from the hyperprior is used together with the likelihood to form a posterior for the performance of a class.
3. A multinomial likelihood models the observed counts in the confusion matrix for a given class.
"""
import argparse
import time
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


def plot_posterior_metrics(bayesian_models, separate_test_matrix):
    if len(bayesian_models) < 2:
        raise ValueError("There should be at least two Bayesian models to compared.")

    # Verify that all Bayesian models have the same classes and can be compared.
    classes = bayesian_models[0].data_task.classes()
    if not all((b.data_task.classes() == classes) for b in bayesian_models[1:]):
        raise ValueError("All Bayesian models being compared should have the same list of classes.")

    #verify that traces exist
    if not all((b.trace_exists() for b in bayesian_models)):
        raise ValueError("Posterior traces must be sampled for each plotted model.")

    # All models are assumed to operate on the same task.
    model_names = [b.name() for b in bayesian_models]
    task_name = bayesian_models[0].data_task.name()
    os.makedirs(os.path.join(PLOT_DIRECTORY, task_name), exist_ok=True)

    for class_index, class_name in enumerate(classes):

        # Posterior distributions to plot.
        recall_arrays = []
        precision_arrays = []
        for b in bayesian_models:
            if separate_test_matrix:
                pp = b.load_posterior_predictive()
                recall_arrays.append(b.posterior_predictive_recalls(pp_samples=pp, class_index=class_index))
                precision_arrays.append(b.posterior_predictive_precisions(pp_samples=pp, class_index=class_index))

            else:
                trace = b.load_trace()
                recall_arrays.append(b.posterior_recalls(trace=trace, class_index=class_index))
                precision_arrays.append(b.posterior_precisions(trace=trace, class_index=class_index))
        
        # If needed convert the test confusion matric to a one-vs-all confusion matrix for this
        # class.
        test_cm = None
        if separate_test_matrix:
            test_cm = convert_to_binary(confusion_matrix=b.data_task.test_confusion_matrix(), class_index=class_index)

        plot_posterior_comparison(
            model_posteriors=recall_arrays,
            model_names=model_names,
            plot_path=os.path.join(PLOT_DIRECTORY, task_name, f"{task_name}_recall_class_{class_index}"),
            metric_name="Recall",
            task_name=task_name,
            class_name=classes[class_index],

            # Given that all models work on the same task, they have the same underlying observed confusion matrix
            # and class counts.
            num_train_samples=b.num_train_samples_per_class(class_index),
            num_eval_samples=b.num_eval_samples_per_class(class_index),
            observed_fit=b.observed_binary_cm(class_index).recall(),
            observed_test=test_cm.recall() if test_cm else None
        )
        plot_posterior_comparison(
            model_posteriors=precision_arrays,
            model_names=model_names,
            plot_path=os.path.join(PLOT_DIRECTORY, task_name, f"{task_name}_precision_class_{class_index}"),
            metric_name="Precision",
            task_name=task_name,
            class_name=class_name,
            num_train_samples=b.num_train_samples_per_class(class_index),
            num_eval_samples=b.num_eval_samples_per_class(class_index),
            observed_fit=b.observed_binary_cm(class_index).precision(),
            observed_test=test_cm.precision() if test_cm else None
        )

def parse_args():
    """Command line arguments for running the evaluation of the Bayesian models."""
    parser = argparse.ArgumentParser(description="Arguments for the evaluation of Bayesian models used to descripe the performance metrics of neural networks.")
    parser.add_argument("--num-samples-per-core", type=int, default=2000, help="Number of elements in the Markov chain generated on each cpu core to evaluate the Bayesian model.")
    parser.add_argument("--num-cores", type=int, default=None, help="Number of CPU cores to use for sampling.")
    parser.add_argument("--reevaluate", action="store_true", help="Whether to rerun the evaluation of models that have an existing set of sampled Markov chains or posterios samples. By default existing evaluation results will be reused.")
    parser.add_argument("--tasks", choices=get_task_names(), nargs="+", help="Only evaluate the bayesian model for a specific data task. By default all tasks are evaluated.")
    parser.add_argument("--bayesian-models", choices=get_bayesian_model_names(), nargs="+", help="Only evaluate the particular Bayesian models specified here. By default all Bayesian models are evaluated.")
    parser.add_argument("--separate-test-matrix", action="store_true", help="Whether to use a separate hold-out confusion matrix to test the Bayesian models against or not.")

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
        models_to_evaluate = [find_bayesian_model(b) for b in args.bayesian_models]
    else:
        models_to_evaluate = BAYESIAN_MODEL_REGISTER

    for task in tasks_to_evaluate:
        print("#"*50)
        print(f"Evaluating bayesian models for {task.name()}.")
        b_models = []
        for model_class in models_to_evaluate:
            print(f"Analyzing {model_class.name()}")
            bm = model_class(task, separate_test_matrix=args.separate_test_matrix)
            b_models.append(bm)
            if not bm.trace_exists() or args.reevaluate:
                tic = time.time()
                trace = bm.trace(num_samples_per_core=args.num_samples_per_core, num_cores=args.num_cores)
                print(f"Tracing {model_class.name()} took {time.time() - tic:.2f} s.")
            else:
                trace = None
                print("Trace already exists.")

            # When comparing the fits on one confusion matrix to a separate test matrix we also need the posterior-predictive samples
            if args.separate_test_matrix and not bm.posterior_predictive_exists():
                trace = bm.load_trace() if trace is None else trace
                bm.sample_posterior_predictive(trace=trace)
            elif args.separate_test_matrix:
                print("Posterior predictive samples already exist.")

        plot_posterior_metrics(b_models, separate_test_matrix=args.separate_test_matrix)
