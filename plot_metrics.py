import matplotlib.pyplot as plt
import numpy as np



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
