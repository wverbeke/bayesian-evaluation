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



def build_simple_model(observed_cms: List[BinaryCM]):
    """Build a simple model in which each class has an independent prior on its confusion matrix."""
    
    # Compute the number of classes and the total count.
    num_classes = len(observed_cms)
    total_count = np.sum(observed_cms[0].numpy())

    with pm.Model() as model:
        for i in range(num_classes):
            # A confusion matrix has 4 entries so we need 4 priors.
            #prior = pm.Uniform(f"prior_class_{i}", lower=0.0, upper=1.0, shape=4)
            prior = pm.Dirichlet(f"prior_class_{i}", a=np.ones(4))
            likelihood = pm.Multinomial(f"likelihood_class_{i}", n=total_count, p=prior, observed=observed_cms[i].numpy())
    return model


def build_hyperprior_model(observed_cms: List[BinaryCM]):
    """Build the hierarchical model with the hyperprior."""

    # Compute the number of classes and the total count.
    num_classes = len(observed_cms)
    total_count = np.sum(observed_cms[0].numpy())

    with pm.Model() as model:
        # THIS DOES NOT SEEM TO WORK
        # The hyperprior has 8 parameters.
        # The prior over each of the 4 parameters describing the multinomial from which a confusion
        # matrix is sampled is determined from two parameters sampled from the hyperprior, one from
        # h_alpha and one from h_gamma.
        #h_alpha = pm.Uniform("h_alpha", lower=0.0, upper=10, shape=4)
        #h_gamma = pm.Uniform("h_gamma", lower=0.0, upper=10, shape=4)
        # -----
        hyperprior = pm.Uniform("hyperprior", lower=0.0, upper="100.", shape=4)
    
        # For each class, a prior is sampled from the hyperprior and a likelihood function is
        # defined based on the observed confusion matrix.
        class_priors = []
        class_likelihoods = []
        for i in range(num_classes):
            #_prior = pm.Beta(f"_prior_class_{i}", alpha=h_alpha, beta=h_gamma, shape=4)
            #prior = pm.Deterministic(f"prior_class_{i}", _prior/_prior.sum())
            prior = pm.Dirichlet(f"prior_class_{i}", a=hyperprior)
            likelihood = pm.Multinomial(f"likelihood_class_{i}", n=total_count, p=prior, observed=observed_cms[i].numpy())
            class_priors.append(prior)
            class_likelihoods.append(likelihood)

    return model


def compute_recall(cm_array):
    tp_array = cm_array[:, 0]
    fn_array = cm_array[:, 2]
    return tp_array/(tp_array + fn_array)


def compute_precision(cm_array):
    tp_array = cm_array[:, 0]
    fp_array = cm_array[:, 1]
    return tp_array/(tp_array + fp_array)


def draw_posterior_metrics(trace, model):
    print("crash 1")
    with model:
        print("crash 1.1")
        predictive_samples = pm.sample_posterior_predictive(trace)
    print("crash 2")
    #print(predictive_samples)
    #print(predictive_samples.posterior_predictive.likelihood_class_0)
    #print(predictive_samples.observed_data)
    sample_array = predictive_samples.posterior_predictive.likelihood_class_40
    recalls = []
    for chain_index in range(len(sample_array)):
        recalls.append(compute_recall(sample_array[chain_index]))
    recalls = np.concatenate(recalls)
    return recalls



from plot_metrics import plot_posterior_comparison

if __name__ == "__main__":
    #cm_path = CIFAR10Task.confusion_matrix_path()
    cm_path = GTSRBTask.confusion_matrix_path()
    cm = np.load(cm_path)
    
    binary_cms = []
    for i in range(len(cm)):
        binary_cms.append(convert_to_binary(cm, i))

    #print(binary_cms[0])
    #print(binary_cms[0].recall())
    #print(binary_cms[0].precision())
    #print(binary_cms[0].f1score())

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


    #print(trace)
    #print(trace.posterior)
    #print(trace.posterior.hyperprior)
    #print(trace.sample_stats)

        #posterior_draws = pm.fast_sample_posterior_predictive(trace)
        #for p in posterior_draws:
        #    print(posterior_draws[p])
        #keys = [f"prior_class_{i}" for i in range(num_classes)]
        #posterior_draws = {k:[] for k in keys}

