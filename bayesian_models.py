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

TRACE_DIRECTORY="mc_traces"
POSTERIOR_DIRECTORY="posterior_samples"
PLOT_DIRECTORY="plots"

# Cutoff for logarithm calculations.
LOG_CUTOFF=1e-4


def _prior_name(class_index: int) -> str:
    """Internal name of the priors in a Bayesian model."""
    return f"prior_class_{class_index}"


def _likelihood_name(class_index: int) -> str:
    """Internal name of the likelihoods in a Bayesian model."""
    return f"likelihood_class_{class_index}"


# Make a register of all Bayesian models so they can be easily looped over later on.
BAYESIAN_MODEL_REGISTER = []
def register_bayesian_model(cls):
    """Register the Bayesian model."""
    BAYESIAN_MODEL_REGISTER.append(cls)
    return cls

def get_bayesian_model_names() -> List[str]:
    return [b.name() for b in BAYESIAN_MODEL_REGISTER]

def find_bayesian_model(name: str):
    """Find a Bayesian model by its name."""
    try:
        names = get_bayesian_model_names()
        index = names.index(name)
        return BAYESIAN_MODEL_REGISTER[index]
    except ValueError:
        raise ValueError(f"No bayesian model with name {name}")


def concatenate_chains(markov_chains: np.ndarray) -> np.ndarray:
    """Concatenate the markov chains generated on each CPU core."""
    return np.concatenate(markov_chains, axis=0)


class BayesianModel:
    """Base class for Bayesian models.

    This class provides all the boilerplate functionality that makes it easy to keep track of
    everything in a Bayesian model created with PyMC. The only thing a user needs to implement
    is a function that given a set of observed confusion matrices for binary classification
    problems returns a PyMC model.

    This class handles the loading and saving of Markov chains and posterior samples for a given
    Bayesian model and a data task (such as CIFAR10 of CIFAR100).
    """
    def __init__(self, data_task, separate_test_matrix = False):
        self._data_task = data_task
    
        # Load the observed confusion matrix for the given task.
        if separate_test_matrix:
            total_cm = data_task.fit_confusion_matrix()
        else:
            total_cm = data_task.confusion_matrix()

        # Convert the total confusion matrix to a set of binary one-vs-all confusion matrices for each class.
        # Store the binary confusion matrices for plotting the observed values later on.
        self._binary_cms = []
        for class_index, _ in enumerate(total_cm):
            binary_cm = convert_to_binary(confusion_matrix=total_cm, class_index=class_index)
            self._binary_cms.append(binary_cm)


        # Build the Bayesian model with the observed confusion matrices.
        # The function the builds the model must be specified by the user for each concrete model.
        self._model = self.build_model(observed_cms=self._binary_cms)

        # Compute the number of entries in the evaluation set used for each class in the data sets
        # used to make the confusion matrices.
        self._num_eval_samples_per_class = []

        # Use the fact that the CM is square.
        for class_index, _ in enumerate(total_cm):
            self._num_eval_samples_per_class.append(np.sum(total_cm[:, class_index]))

    @classmethod
    @abstractmethod
    def name(cls):
        """Name for this class of Bayesian models."""

    @property
    def data_task(self) -> DataTask:
        """Retrieve the underlying data task."""
        return self._data_task

    def trace_file_path(self) -> str:
        """Path to the sampled Markov chains."""
        return os.path.join(TRACE_DIRECTORY, f"trace_{self.name()}_{self._data_task.name()}.cdf")

    def plot_file_path(self, metric_name: str, class_index: int) -> str:
        """Path to plots."""
        return os.path.join(PLOT_DIRECTORY, f"plot_{metric_name}_{self.name()}_{self._data_task.name()}_class_{class_index}")

    @abstractmethod
    def build_model(self, observed_cms: List[BinaryCM]):
        """Build the Bayesian model."""

    # TODO: rename to sample
    def trace(self, num_samples_per_core: int, num_cores: Optional[int] = None):
        """Sample a Markov chain from the Bayesian model on each CPU core."""
        if num_cores is None:
            num_cores = os.cpu_count()
        with self._model:
            # Build a Markov chain of samples on each cpu core.
            trace = pm.sample(draws=num_samples_per_core, cores=num_cores)

            # Write the Markov chain to disk.
            os.makedirs(TRACE_DIRECTORY, exist_ok=True)
            trace.posterior.to_netcdf(self.trace_file_path())
        
        return trace

    def load_trace(self):
        """Load the sampled Markov chain."""
        return xr.open_dataset(self.trace_file_path())

    def trace_exists(self) -> bool:
        """Verify that Markov chains for the model were already written to disk."""
        return os.path.isfile(self.trace_file_path())

    def num_eval_samples_per_class(self, class_index: int) -> int:
        """Number of evaluation samples per class."""
        return self._num_eval_samples_per_class[class_index]

    def num_train_samples_per_class(self, class_index: int) -> int:
        """Number of training samples per class."""
        return self.data_task.num_train_samples_per_class

    def observed_binary_cm(self, class_index: int) -> BinaryCM:
        """Retrieve the observed binary confusion matrix for a particular class."""
        return self._binary_cms[class_index]

    @abstractmethod
    def posterior_recalls(self, trace, class_index):
        """Extract posterior recall values from a simulated trace.

        This has to be implemented for each model because it depends on the model details.
        """

    @abstractmethod
    def posterior_precisions(self, trace, class_index):
        """Extract posterior precision values from a simulated trace.

        This has to be implemented for each model because it depends on the model details.
        """


class MultinomialLikelihoodModel(BayesianModel):
    """Bayesian models in which the final likelihood is a multinomial distribution.

    For such models the computation of the posterior recall and precision values can be computed
    in a generic fashion.
    """
    def _posterior_multinomial_params(self, trace, class_index):
        """Extract samples from the posterior parameters describing the multinomial."""
        # For each Markov chain (one per CPU core) a separate array with sampled confusion matrices is returned.
        return concatenate_chains(trace["prior"][:, :, class_index])

    def posterior_recalls(self, trace, class_index):
        """Compute the recalls from the multinomial parameters."""
        cm_array = self._posterior_multinomial_params(trace=trace, class_index=class_index)
        tp_array = cm_array[:, 0]
        fn_array = cm_array[:, 2]
        return divide_safe(tp_array, tp_array + fn_array)

    def posterior_precisions(self, trace, class_index):
        """Compute the precisions from the multinomial parameters."""
        cm_array = self._posterior_multinomial_params(trace=trace, class_index=class_index)
        tp_array = cm_array[:, 0]
        fp_array = cm_array[:, 1]
        return divide_safe(tp_array, tp_array + fp_array)

        
@register_bayesian_model
class SimpleModel(MultinomialLikelihoodModel):
    """Simple model in which each class has an independent prior.

    Each class has an independent prior, so the observations for one class put no constraints on
    the prior of another class. This is the model introduced in
    https://link.springer.com/article/10.1007/s10472-017-9564-8

    Each confusion matrix is taken to be a sample drawn from a multinomial, and a Dirichlet prior
    is assumed over the parameters of the multinomial. Vector samples drawn from a Dirichlet sum to
    1, making it a good prior distribution for parameters of a multinomial which must also sum
    to 1.
    """
    @classmethod
    def name(cls):
        return "simple_model"

    def build_model(self, observed_cms: List[BinaryCM]):
        """Build a simple model in which each class has an independent prior on its confusion matrix."""
        # Compute the number of classes and the total count.
        num_classes = len(observed_cms)
        total_count = np.sum(observed_cms[0].numpy())
        observed_array = np.array([cm.numpy() for cm in observed_cms])

        with pm.Model() as model:
            # A confusion matrix has 4 entries so we need 4 priors.
            prior = pm.Dirichlet("prior", a=np.ones((num_classes, 4)))
            likelihood = pm.Multinomial("likelihood", n=total_count, p=prior, observed=observed_array)
        return model


@register_bayesian_model
class FractionCountModel(BayesianModel):

    @classmethod
    def name(cls):
        return "fraction_count_model"

    @classmethod
    def build_model(cls, observed_cms: List[BinaryCM]):

        num_classes = len(observed_cms)
        total_count = np.sum(observed_cms[0].numpy())

        counts_per_class = np.array([(cm.tp + cm.fn) for cm in observed_cms])


        with pm.Model() as model:
            count_prior = pm.Dirichlet("count_prior", a=np.ones(num_classes))
            count_likelihood = pm.Multinomial("count_likelihood", n=total_count, p=count_prior, observed=counts_per_class)

            true_class_size_hyperprior = pm.Exponential("true_class_size_hyperprior", 1/100)
            true_class_bias_hyperprior = pm.Beta("true_class_bias_hyperprior", 1, 1)
            true_class_alpha_hyperprior = pm.Deterministic("true_class_alpha_hyperprior", true_class_size_hyperprior * true_class_bias_hyperprior)
            true_class_beta_hyperprior = pm.Deterministic("true_class_beta_hyperprior", true_class_size_hyperprior * (1-true_class_bias_hyperprior))

            false_class_size_hyperprior = pm.Exponential("false_class_size_hyperprior", 1/100)
            false_class_bias_hyperprior = pm.Beta("false_class_bias_hyperprior", 1, 1)
            false_class_alpha_hyperprior = pm.Deterministic("false_class_alpha_hyperprior", false_class_size_hyperprior * false_class_bias_hyperprior)
            false_class_beta_hyperprior = pm.Deterministic("false_class_beta_hyperprior", false_class_size_hyperprior * (1-false_class_bias_hyperprior))

            n_obs_true = count_likelihood
            n_obs_false = pm.math.sum(count_likelihood) - n_obs_true
            observed_tps = np.array([cl.tp for cl in observed_cms])
            observed_tns = np.array([cl.tn for cl in observed_cms])

            true_class_prior = pm.Beta("true_class_prior", alpha=true_class_alpha_hyperprior, beta=true_class_beta_hyperprior, shape=num_classes)
            false_class_prior = pm.Beta("false_class_prior", alpha=false_class_alpha_hyperprior, beta=false_class_beta_hyperprior, shape=num_classes)

            true_class_fraction = pm.Binomial("true_class_fraction", p=true_class_prior, n=n_obs_true, observed=observed_tps)
            false_class_fraction = pm.Binomial("false_class_fraction", p=false_class_prior, n=n_obs_false, observed=observed_tns)

            return model


    def posterior_recalls(self, trace, class_index):
        """Compute the recalls from the multinomial parameters."""
        return concatenate_chains(trace["true_class_prior"][:, :, class_index])

    def posterior_precisions(self, trace, class_index):
        """Compute the precisions from the multinomial parameters."""
        count_priors = concatenate_chains(trace["count_prior"])

        true_fraction_prior = concatenate_chains(trace["true_class_prior"][:, :, class_index])
        true_counts = count_priors[:, class_index]
        true_positives = true_counts*true_fraction_prior

        false_fraction_prior = concatenate_chains(trace["false_class_prior"][:, :, class_index])
        false_counts = np.sum(count_priors, axis=1) - true_counts

        false_positives = false_counts*(1.0 - false_fraction_prior)
        return true_positives/(true_positives + false_positives)

