import matplotlib.pyplot as plt
import numpy as np

def plot_posterior_comparison(model_posteriors, model_names, plot_path, metric_name, task_name, class_name, num_class_samples = None, num_bins: int = 40):

    if len(model_posteriors) != len(model_names):
        raise ValueError("Each posterior should have a corresponding name.")

    # Get a good binning for the sum of all posteriors.
    _, bins = np.histogram(np.hstack([*model_posteriors]), bins=num_bins)

    # Plot the posteriors for each model.
    for posterior, name in zip(model_posteriors, model_names):
        plt.hist(posterior, bins=bins, histtype="step", label=name)
    plt.xlabel(metric_name)
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title(f"{task_name}: {class_name}")

    if not "." in plot_path:
        plt.savefig(plot_path + ".pdf")
        plt.savefig(plot_path + ".png")
    plt.clf()
