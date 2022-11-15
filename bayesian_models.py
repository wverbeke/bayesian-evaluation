""" Implementation of the statistical model used to constrain the performance of a classifier.

Neural network models are assumed to be trained and are evaluated in terms of their confusion matrix.

The statistical model consists of three stages:
1. A hyperprior generates parameters that determines the prior of the performance for a particular class.
2. The prior sampled from the hyperprior is used together with the likelihood to form a posterior for the performance of a class.
3. A multinomial likelihood models the observed counts in the confusion matrix for a given class.
"""
from typing import List
import os
from abc import abstractmethod

import numpy as np
import pymc as pm
import xarray as xr

from confusion_matrix import BinaryCM, convert_to_binary
from data_tasks import CIFAR10Task, GTSRBTask, TASK_REGISTER
from plot_metrics import plot_posterior_comparison

TRACE_DIRECTORY="mc_traces"
POSTERIOR_DIRECTORY="posterior_samples"
PLOT_DIRECTORY="plots"

def _prior_name(class_index):
    return f"prior_class_{class_index}"


def _likelihood_name(class_index):
    return f"likelihood_class_{class_index}"


def compute_recalls(cm_array):
    """Compute the recalls for an array of sampled confusion matrices."""
    tp_array = cm_array[:, 0]
    fn_array = cm_array[:, 2]
    return tp_array/(tp_array + fn_array)


def compute_precisions(cm_array):
    """Compute the precisions for an array of sampled confusion matrices."""
    tp_array = cm_array[:, 0]
    fp_array = cm_array[:, 1]
    return tp_array/(tp_array + fp_array)


BAYESIAN_MODEL_REGISTER = []
def register_bayesian_model(cls):
    BAYESIAN_MODEL_REGISTER.append(cls)
    return cls


class BayesianModel:

    def __init__(self, data_task):
        self._data_task = data_task

        total_cm = np.load(data_task.confusion_matrix_path())
        binary_cms = []
        for class_index, _ in enumerate(total_cm):
            binary_cm = convert_to_binary(confusion_matrix=total_cm, class_index=class_index)
            binary_cms.append(binary_cm)
        self._model = self.build_model(observed_cms=binary_cms)

    @property
    def data_task(self):
        return self._data_task

    def trace_file_path(self):
        return os.path.join(TRACE_DIRECTORY, f"trace_{self.name()}_{self._data_task.name()}.cdf")

    def posterior_file_path(self, class_index):
        return os.path.join(POSTERIOR_DIRECTORY, f"posterior_sample_{self.name()}_{self._data_task.name()}_class_{class_index}.npy")

    def plot_file_path(self, metric_name, class_index):
        return os.path.join(PLOT_DIRECTORY, f"plot_{metric_name}_{self.name()}_{self._data_task.name()}_class_{class_index}")

    @classmethod
    @abstractmethod
    def name(cls):
        """Name for this class of Bayesian models."""

    @classmethod
    @abstractmethod
    def build_model(cls, observed_cms):
        """Build the Bayesian model."""


    def trace(self, num_samples_per_core=100):

        with self._model:
            # Build a Markov chain of samples on each cpu core.
            trace = pm.sample(draws=num_samples_per_core, cores=os.cpu_count())

            # Write the Markov chain to disk.
            os.makedirs(TRACE_DIRECTORY, exist_ok=True)
            trace.posterior.to_netcdf(self.trace_file_path())
        
        return trace


    def load_trace(self):
        return xr.open_dataset(self.trace_file_path())


    def sample_posterior_predictive(self, trace):
        os.makedirs(POSTERIOR_DIRECTORY, exist_ok=True)
        with self._model:
            posterior_samples = pm.sample_posterior_predictive(trace)
        sampled_cm_arrays = []
        for class_index in range(self.data_task.num_classes()):
            likelihood_name = _likelihood_name(class_index)
            sampled_cm_chains = getattr(posterior_samples.posterior_predictive, likelihood_name)
            sampled_cm_array = np.concatenate(sampled_cm_chains, axis=0)
            sampled_cm_arrays.append(sampled_cm_array)
            np.save(self.posterior_file_path(class_index), sampled_cm_array)
        return sampled_cm_arrays


    def load_posterior_samples(self, class_index=None):
        if class_index is None:
            sampled_cm_arrays = []
            for class_index in range(self.data_task.num_classes()):
                sampled_cm_array = np.load(self.posterior_file_path(class_index))
                sampled_cm_arrays.append(sampled_cm_array)
            return sampled_cm_arrays
        else:
            return np.load(self.posterior_file_path(class_index))


    def trace_exists(self):
        return os.path.isfile(self.trace_file_path())


    def posterior_samples_exist(self):
        return all(os.path.isfile(self.posterior_file_path(class_index)) for class_index in range(self.data_task.num_classes()))

        
@register_bayesian_model
class SimpleModel(BayesianModel):

    @classmethod
    def name(cls):
        return "simple_model"


    @classmethod
    def build_model(cls, observed_cms: List[BinaryCM]):
        """Build a simple model in which each class has an independent prior on its confusion matrix."""
        
        # Compute the number of classes and the total count.
        num_classes = len(observed_cms)
        total_count = np.sum(observed_cms[0].numpy())

        with pm.Model() as model:
            for class_index in range(num_classes):
                # A confusion matrix has 4 entries so we need 4 priors.
                prior = pm.Dirichlet(_prior_name(class_index), a=np.ones(4))
                likelihood = pm.Multinomial(_likelihood_name(class_index), n=total_count, p=prior, observed=observed_cms[class_index].numpy())
        return model


@register_bayesian_model
class DirichletHyperpriorModel(BayesianModel):

    @classmethod
    def name(cls):
        return "dirichlet_hyperprior_model"


    @classmethod
    def build_model(cls, observed_cms: List[BinaryCM]):
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
            hyperprior = pm.Uniform("hyperprior", lower=0.0, upper=100.0, shape=4)
        
            # For each class, a prior is sampled from the hyperprior and a likelihood function is
            # defined based on the observed confusion matrix.
            class_priors = []
            class_likelihoods = []
            for class_index in range(num_classes):
                prior = pm.Dirichlet(_prior_name(class_index), a=hyperprior)
                likelihood = pm.Multinomial(_likelihood_name(class_index), n=total_count, p=prior, observed=observed_cms[class_index].numpy())
                class_priors.append(prior)
                class_likelihoods.append(likelihood)
    
        return model


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

    os.makedirs(PLOT_DIRECTORY, exist_ok=True)

    model_names = [b.name() for b in bayesian_models]
    task_name = bayesian_models[0].data_task.name()

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
            plot_path=os.path.join(PLOT_DIRECTORY, f"{task_name}_recall_class_{class_index}"),
            metric_name="Recall",
            task_name=task_name,
            class_name=classes[class_index]
        )
        plot_posterior_comparison(
            model_posteriors=precision_arrays,
            model_names=model_names,
            plot_path=os.path.join(PLOT_DIRECTORY, f"{task_name}_precision_class_{class_index}"),
            metric_name="Precision",
            task_name=task_name,
            class_name=class_name
        )


if __name__ == "__main__":
    for task in TASK_REGISTER:
        b_models = []
        for model_class in BAYESIAN_MODEL_REGISTER:
            bm = model_class(task)
            b_models.append(bm)
            if not bm.trace_exists():
                trace = bm.trace(num_samples_per_core=2000)
                bm.sample_posterior_predictive(trace)
            elif not bm.posterior_samples_exist():
                bm.sample_posterior_predictive(bm.load_trace())
        plot_posterior_metrics(models)
