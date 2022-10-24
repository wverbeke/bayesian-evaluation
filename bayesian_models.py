""" Implementation of the statistical model used to constrain the performance of a classifier.

Neural network models are assumed to be trained and are evaluated in terms of their confusion matrix.

The statistical model consists of three stages:
1. A hyperprior generates parameters that determines the prior of the performance for a particular class.
2. The prior sampled from the hyperprior is used together with the likelihood to form a posterior for the performance of a class.
3. A multinomial likelihood models the observed counts in the confusion matrix for a given class.
"""
import numpy as np
import pymc3 as pm


def build_simple_model(num_classes: int, observed_cm: ):
    """Build a simple model in which each class has an independent prior on its confusion matrix."""
    with pm.Model() as model:
        for i in range(num_classes):
            # A confusion matrix has 4 entries so we need 4 priors.
            prior = pm.Uniform(f"prior_class_{i}", lower=0.0, upper=1.0, shape=4)
            likelihood = pm.Multinomial(f"likelihood_class_{i}", n=total_count, p=prior, observed=observed_cms[i])
    return model






def build_model(num_classes, total_count, observed_cm):

    with pm.Model() as model:
        # The model has 8 parameters: A hyperprior with 2 parameters for each model.
# The final thing we want is to constrain t1, t2, t3, and t4 from a multinomial as precisely as possible.
        h_alpha = pm.Uniform("h_1", lower=0.0, upper=100, shape=4)
        h_gamma = pm.Uniform("h_2", lower=0.0, upper=100, shape=4)
    
        class_priors = []
        class_likelihoods = []
        for i in range(num_classes):
            prior = pm.Beta(f"prior_class_{i}", alpha=h_alpha, beta=h_gamma, shape=4)
            like = pm.Multinomial(f"likelihood_class_{i}", n=total_count, p=prior, observed=observed_cm[i])
            class_priors.append(prior)
            class_likelihoods.append(like)

        trace = pm.sample(100, tune=100, cores=4)
        posterior_draws = pm.fast_sample_posterior_predictive(trace)
        for p in posterior_draws:
            print(posterior_draws[p])
        #keys = [f"prior_class_{i}" for i in range(num_classes)]
        #posterior_draws = {k:[] for k in keys}

        #for draw in trace:
        #    for k in keys:
        #        posterior_draws[k].append(draw[k])
        #print(posterior_draws)

    #print(model.basic_RVs)
    #for i in range(10):
    #    print("#"*60)
    #print(model.free_RVs)
    #for i in range(10):
    #    print("#"*60)
    #print(model.observed_RVs)


        #trace = pm.sample(1000)
        #print(trace)

        #class_probabolities = pm.Multinomial(n=total_count, p=priors, shape=num_classs)
        #priors = []
        #for i in range(num_classes):
        #    class_priors = []
        #    for j in 4:
        #        class_priors.append(pm.Beta(f"prior_{i}", alpha=h_1, beta=h_2))

def sample_random_params():
    t1 = np.random.uniform(0.0, 0.5)
    t2 = np.random.uniform(0.0, 0.3)
    t3 = np.random.uniform(0.0, 0.2)
    t4 = 1.0 - t1 - t2 - t3
    return np.array([t1, t2, t3, t4])
    

if __name__ == "__main__":
    rp = random_performances()

    num_classes = 10
    total_count = 1000
    random_cm = [np.random.multinomial(total_count, sample_random_params()) for _ in range(num_classes)]
    model = build_model(num_classes=num_classes, total_count=total_count, observed_cm=random_cm)
    
    
