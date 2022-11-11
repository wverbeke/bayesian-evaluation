import matplotlib.pyplot as plt
import numpy as np

def plot_posterior_comparison(samples_lhs, samples_rhs, num_bins: int = 40):
#def plot_posterior_comparison(samples_lhs, num_bins: int = 40):

    # Get the binning
    _, bins = np.histogram(np.hstack([samples_lhs, samples_rhs]), bins=num_bins)
    plt.hist(samples_lhs, bins=bins, color="blue", histtype="step", label="hyperprior")
    plt.hist(samples_rhs, bins=bins, color="red", histtype="step", label="simple model")
    plt.xlabel("Recall")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.show()
    plt.savefig("test_plot.pdf")
    #_, bins, _ = plt.hist(samples, histtype=)
