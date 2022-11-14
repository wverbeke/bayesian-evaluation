""" Implementation of the statistical model used to constrain the performance of a classifier.

Neural network models are assumed to be trained and are evaluated in terms of their confusion matrix.

The statistical model consists of three stages:
1. A hyperprior generates parameters that determines the prior of the performance for a particular class.
2. The prior sampled from the hyperprior is used together with the likelihood to form a posterior for the performance of a class.
3. A multinomial likelihood models the observed counts in the confusion matrix for a given class.
"""
from typing import List
import os

import numpy as np
import pymc as pm
import xarray as xr

from confusion_matrix import BinaryCM, convert_to_binary
from data_tasks import CIFAR10Task, GTSRBTask

def _prior_name(class_index):
    return f"prior_class_{class_index}"


def _likelihood_name(class_index):
    return f"likelihood_class_{class_index}"


def build_simple_model(observed_cms: List[BinaryCM]):
    """Build a simple model in which each class has an independent prior on its confusion matrix."""
    
    # Compute the number of classes and the total count.
    num_classes = len(observed_cms)
    total_count = np.sum(observed_cms[0].numpy())

    with pm.Model() as model:
        for class_index in range(num_classes):
            # A confusion matrix has 4 entries so we need 4 priors.
            #prior = pm.Uniform(f"prior_class_{i}", lower=0.0, upper=1.0, shape=4)
            prior = pm.Dirichlet(_prior_name(class_index), a=np.ones(4))
            likelihood = pm.Multinomial(_likelihood_name(class_index), n=total_count, p=prior, observed=observed_cms[i].numpy())
    return model


def build_dirichlet_hyperprior_model(observed_cms: List[BinaryCM]):
    """Build the hierarchical model with the hyperprior.

    A dirichlet distribution yields values that sum to one, which is a property the (hyperprior) for
    a model with a multinomial likelihood must have since the parameters of a multinomial sum to 1.
    """

    # Compute the number of classes and the total count.
    num_classes = len(observed_cms)
    total_count = np.sum(observed_cms[0].numpy())

    with pm.Model() as model:
        # The prior over each of the 4 parameters describing the multinomial from which a confusion
        # matrix is sampled is a dirichlet distribution whose parameters are samples from the hyperprior.
        hyperprior = pm.Uniform("hyperprior", lower=0.0, upper="100.", shape=4)
    
        # For each class, a prior is sampled from the hyperprior and a likelihood function is
        # defined based on the observed confusion matrix.
        class_priors = []
        class_likelihoods = []
        for class_index in range(num_classes):
            prior = pm.Dirichlet(_prior_name(class_index), a=hyperprior)
            likelihood = pm.Multinomial(_likelihood_name(class_index), n=total_count, p=prior, observed=observed_cms[i].numpy())
            class_priors.append(prior)
            class_likelihoods.append(likelihood)

    return model


def compute_recall(cm_array):
    """Compute the recalls for an array of sampled confusion matrices."""
    tp_array = cm_array[:, 0]
    fn_array = cm_array[:, 2]
    return tp_array/(tp_array + fn_array)


def compute_precision(cm_array):
    """Compute the precisions for an array of sampled confusion matrices."""
    tp_array = cm_array[:, 0]
    fp_array = cm_array[:, 1]
    return tp_array/(tp_array + fp_array)


def draw_posterior_metrics(trace, model):
    with model:
        predictive_samples = pm.sample_posterior_predictive(trace)
    sample_array = predictive_samples.posterior_predictive.likelihood_class_40
    recalls = []
    for chain_index in range(len(sample_array)):
        recalls.append(compute_recall(sample_array[chain_index]))
    recalls = np.concatenate(recalls)
    return recalls



from plot_metrics import plot_posterior_comparison

if __name__ == "__main__":
    cm_path = GTSRBTask.confusion_matrix_path()
    cm = np.load(cm_path)
    
    binary_cms = []
    for i in range(len(cm)):
        binary_cms.append(convert_to_binary(cm, i))

    print("Building hyperprior model.")
    hyperprior_model = build_hyperprior_model(binary_cms)
    with hyperprior_model:
        trace = pm.sample(draws=100, cores=os.cpu_count())
    trace.posterior.to_netcdf("hyperprior_trace.cdf") 

    print("Building simple model.")
    simple_model = build_simple_model(binary_cms)
    with simple_model:
        trace = pm.sample(draws=100, cores=os.cpu_count())
    trace.posterior.to_netcdf("simple_trace.cdf") 

    # load trace file 
    print("Opening hyperprior trace.")
    hyperprior_trace = xr.open_dataset("hyperprior_trace.cdf")

    print("Opening simple trace.")
    simple_trace = xr.open_dataset("simple_trace.cdf")
    #print(disk_trace.hyperprior)

    print("Drawing hyperprior metrics.")
    hyperprior_recalls = draw_posterior_metrics(hyperprior_trace, hyperprior_model)

    print("Drawing simple metrics.")
    simple_recalls = draw_posterior_metrics(simple_trace, simple_model)

    print("Plotting.")
    plot_posterior_comparison(hyperprior_recalls, simple_recalls)
