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


class BayesianModel:
    """Base class for Bayesian models.

    This class provides all the boilerplate functionality that makes it easy to keep track of
    everything in a Bayesian model created with PyMC. The only thing a user needs to implement
    is a function that given a set of observed confusion matrices for binary classification
    problems returns a PyMC model.

    This class handles the loading and saving of Markov chains and posterior samples for a given
    Bayesian model and a data task (such as CIFAR10 of CIFAR100).
    """
    def __init__(self, data_task):
        self._data_task = data_task
    
        # Load the observed confusion matrix for the given task.
        total_cm = data_task.get_confusion_matrix()

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
        self._num_samples_per_class = []

        # Use the fact that the CM is square.
        for class_index, _ in enumerate(total_cm):
            self._num_samples_per_class.append(np.sum(total_cm[:, class_index]))

    @property
    def data_task(self) -> DataTask:
        """Retrieve the underlying data task."""
        return self._data_task

    def trace_file_path(self) -> str:
        """Path to the sampled Markov chains."""
        return os.path.join(TRACE_DIRECTORY, f"trace_{self.name()}_{self._data_task.name()}.cdf")

    def posterior_file_path(self, class_index) -> str:
        """Path to the samples drawn from the posteriors."""
        return os.path.join(POSTERIOR_DIRECTORY, f"posterior_sample_{self.name()}_{self._data_task.name()}_class_{class_index}.npy")

    def plot_file_path(self, metric_name, class_index) -> str:
        """Path to plots."""
        return os.path.join(PLOT_DIRECTORY, f"plot_{metric_name}_{self.name()}_{self._data_task.name()}_class_{class_index}")

    @classmethod
    @abstractmethod
    def name(cls):
        """Name for this class of Bayesian models."""

    @classmethod
    @abstractmethod
    def build_model(cls, observed_cms: List[BinaryCM]):
        """Build the Bayesian model."""

    def trace(self, num_samples_per_core: int =100):
        """Sample a Markov chain from the Bayesian model on each CPU core."""
        with self._model:
            # Build a Markov chain of samples on each cpu core.
            trace = pm.sample(draws=num_samples_per_core, cores=os.cpu_count())

            # Write the Markov chain to disk.
            os.makedirs(TRACE_DIRECTORY, exist_ok=True)
            trace.posterior.to_netcdf(self.trace_file_path())
        
        return trace

    def load_trace(self):
        """Load the sampled Markov chain."""
        return xr.open_dataset(self.trace_file_path())

    def sample_posterior_predictive(self, trace) -> List[np.ndarray]:
        """Draw posterior samples given a Markov chain from the Bayesian model.

        For each element in the chain a single posterior sample will be drawn.
        """
        os.makedirs(POSTERIOR_DIRECTORY, exist_ok=True)
        with self._model:
            posterior_samples = pm.sample_posterior_predictive(trace)
        sampled_cm_arrays = []

        # Get out the values corresponding to sampled confusion matrices from the posterior.
        for class_index in range(self.data_task.num_classes()):
            likelihood_name = _likelihood_name(class_index)
            sampled_cm_chains = getattr(posterior_samples.posterior_predictive, likelihood_name)

            # For each Markov chain (one per CPU core) a separate array with sampled confusion matrices is returned.
            sampled_cm_array = np.concatenate(sampled_cm_chains, axis=0)
            sampled_cm_arrays.append(sampled_cm_array)
            np.save(self.posterior_file_path(class_index), sampled_cm_array)
        return sampled_cm_arrays


    def load_posterior_samples(self, class_index: Optional[int] = None) -> Union[List[np.ndarray], np.ndarray]:
        """Load samples drawn from the posterior and written to disk."""
        if class_index is None:
            sampled_cm_arrays = []
            for class_index in range(self.data_task.num_classes()):
                sampled_cm_array = np.load(self.posterior_file_path(class_index))
                sampled_cm_arrays.append(sampled_cm_array)
            return sampled_cm_arrays
        else:
            return np.load(self.posterior_file_path(class_index))

    def trace_exists(self) -> bool:
        """Verify that Markov chains for the model were already written to disk."""
        return os.path.isfile(self.trace_file_path())

    def posterior_samples_exist(self) -> bool:
        """Verify that samples drawn from the posterior were already written to disk."""
        return all(os.path.isfile(self.posterior_file_path(class_index)) for class_index in range(self.data_task.num_classes()))

    def num_samples_per_class(self, class_index: int) -> int:
        """Number of evaluation samples per class."""
        return self._num_samples_per_class[class_index]

    def observed_binary_cm(self, class_index: int) -> BinaryCM:
        """Retrieve the observed binary confusion matrix for a particular class."""
        return self._binary_cms[class_index]

        
@register_bayesian_model
class SimpleModel(BayesianModel):
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
    """Simplest model with a hyperprior.

    Similarly to in SimpleModel, the observed confusion matrices are assumed to come from a multinomial,
    which have Dirichlet priors. But instead of having independent Dirichlet priors for each model, the
    parameters governing the Dirchlet priors are drawn from a hyperprior. In this way the knowledge of
    the performance of each class constrains the posterior of the other classes. This encapsulates the
    fact that for a given neural network we can already make an educated guess on the performance of the
    Nth class if we have observed the performance for N - 1 classes.
    """
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
            hyperprior = pm.Uniform("hyperprior", lower=0.0, upper=10000.0, shape=4)
        
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


# TODO: Refactor code to have easy access to the number of training samples.
@register_bayesian_model
class LogRegressionModel(BayesianModel):

    @classmethod
    def name(cls):
        return "log_regression_model"

    @classmethod
    def build_model(cls, observed_cms: List[BinaryCM]):

        num_classes = len(observed_cms)
        total_count = np.sum(observed_cms[0].numpy())

        with pm.Model() as model:
            bias_hyperprior = pm.Uniform("bias_hyperprior", lower=0.0, upper=10000.0, shape=4)
            reg_hyperprior = pm.Uniform("reg_hyperprior", lower=0.0, upper=10000.0, shape=4)

            class_priors = []
            class_likelihoods = []
            for class_index in range(num_classes):
                
                # Use the evaluation set counts as a prior for the training counts.
                observed_cm = observed_cms[class_index]
                example_count = (observed_cm.tp + observed_cm.fn)

                prior = pm.Dirichlet(_prior_name(class_index), a=(bias_hyperprior + example_count*pm.math.log(LOG_CUTOFF+ reg_hyperprior)))
                likelihood = pm.Multinomial(_likelihood_name(class_index), n=total_count, p=prior, observed=observed_cm.numpy())
                class_priors.append(prior)
                class_likelihoods.append(likelihood)

            return model

@register_bayesian_model
class FractionModel(BayesianModel):

    @classmethod
    def name(cls):
        return "fraction_model"

    @classmethod
    def build_model(cls, observed_cms: List[BinaryCM]):

        num_classes = len(observed_cms)
        total_count = np.sum(observed_cms[0].numpy())
        
        eval_counts_per_class = np.array([(cm.tp + cm.fn) for cm in observed_cms])
        train_counts_per_class = np.array([self.data_task.num_train_samples(class_index) for class_index in range(num_classes)])
        count_likelihood = pm.Multinomial("count_likelihood", n=total_count, observed=eval_counts_per_class)

        true_class_bias_hyperprior = pm.Uniform("true_class_bias_hyperprior", lower=0.0, upper=10000.0, shape=2)
        true_class_reg_hyperprior = pm.Uniform("true_class_reg_hyperprior", lower=0.0, upper=10000.0, shape=2)

        false_class_bias_hyperprior = pm.Uniform("false_class_bias_hyperprior", lower=0.0, upper=10000.0, shape=2)
        false_class_reg_hyperprior = pm.Uniform("false_class_reg_hyperprior", lower=0.0, upper=10000.0, shape=2)


        for class_index in range(num_classes):

            num_train_true = train_counts_per_class[class_index]
            num_train_false = np.sum(np.delete(train_counts_per_class, class_index))

            n_obs_true = counts_likelihood[class_index]
            n_obs_false = pymc.math.sum(pymc.math.where(c != class_index, counts_likelihood, 0))

            true_fraction_prior = pm.Beta("true_class_" + _prior_name(class_index), a=(true_class_bias_hyperprior + true_class_reg_hyperprior*pm.math.log(LOG_CUTOFF + num_train_truei)))
            true_likelihood = pm.Binomial("true_class_" + _likelihood_name(class_index), p=true_fraction_prior(), n=n_obs_true, observed=[observed_cms[class_index].tp + observed_cms[class_index].fn])

            false_fraction_prior = pm.Beta("false_class_" + _prior_name(class_index), a=(false_class_bias_hyperprior + false_class_reg*pm.math.log(LOG_CUTOFF + num_train_false)))
            false_likelihood = pm.Binomial("false_class_" + _likelihood_name(class_index), p=false_fraction_prior(), n=n_obs_false, observed=[observed_cms[class_index].fp + observed_cms[class_index].tn])
