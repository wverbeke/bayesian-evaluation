import matplotlib.pyplot as plt
import numpy as np


def plot_posterior_comparison(model_posteriors, model_names, plot_path, metric_name, task_name, class_name, num_train_samples, num_eval_samples, observed_fit, observed_test=None, num_bins: int = 40):

    if len(model_posteriors) < 1:
        raise ValueError("There must be at least one posterior to plot.")

    if len(model_posteriors) != len(model_names):
        raise ValueError("Each posterior should have a corresponding name.")

    # Get a good binning for the sum of all posteriors.
    _, bins = np.histogram(np.hstack([*model_posteriors]), bins=num_bins)

    # Plot the posteriors for each model.
    for posterior, name in zip(model_posteriors, model_names):
        plt.hist(posterior, bins=bins, histtype="step", label=name.replace("_", " "))
    plt.xlabel(metric_name)

    # A single observed confusion matrix is given in cases where we are interested in the posterior distribution.
    if observed_test is None:
        plt.ylabel("Number of posterior samples")
    # In the two matrix case we are interested in the posterior predictive distribution.
    else:
        plt.ylabel("Number of posterior predictive samples")
    plt.legend()
    plt.title(f"{task_name}: {class_name}, {num_eval_samples} eval samples, {num_train_samples} train samples.")

    # Make the y-range decent to avoid overlap between the legend and the graph.
    ax = plt.gca()
    y_upper = ax.get_ylim()[-1]*1.2
    plt.ylim([0, y_upper])

    # The same number of samples should be generated for each model.
    chain_length = len(model_posteriors[0])

    # Write the chain length on the plot.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xloc = (xmax - xmin)*0.05 + xmin
    yloc = (ymax - ymin)*0.9 + ymin
    text = f"{chain_length} samples drawn."
    ax.text(xloc, yloc, text, fontsize=10)

    # Draw an arrow at the observed metric value.
    observed_color = "blue" if observed_test else "black"
    plt.annotate("Observed\nfit", xy=(observed_fit, (ymax - ymin)*0.05), xytext=(observed_fit, (ymax - ymin)*0.4), arrowprops={"facecolor":observed_color, "width":0.02}, ha="center")
    if observed_test:
        plt.annotate("Observed\ntest", xy=(observed_test, (ymax - ymin)*0.05), xytext=(observed_test, (ymax - ymin)*0.4), arrowprops={"facecolor":"red", "width":0.02}, ha="center")

    if not "." in plot_path:
        plt.savefig(plot_path + ".png")
    plt.clf()
