import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

from verona.data.download import get_dataset
from verona.data.results import load_results_plackett_luce
from verona.evaluation.stattests.plackettluce import PlackettLuceResults, PlackettLuceRanking


def plot_posteriors_plackett(plackett_results : PlackettLuceResults, save_path=None):
    """
    Plot the posteriors of the Plackett-Luce model (quantiles 95%, 05% and 50%). If two approaches do not overlap,
    they have a significative different ranking.

    Parameters
        save_path: String that indicates the path where the plot will be saved. If None, the plot will not be saved.

    Returns
        fig.figure : matplotlib figure of the aforementioned plot

    Examples:
        >>> result_matrix = pd.DataFrame([[0.75, 0.6, 0.8], [0.8, 0.7, 0.9], [0.9, 0.8, 0.7]])
        >>> plackett_ranking = PlackettLuceRanking(result_matrix, ["a1", "a2", "a3"])
        >>> results = plackett_ranking.run(n_chains=10, num_samples=300000, mode="max")
        >>> plot = plot_posteriors_plackett(results, save_path=None)
        >>> print(plot)
    """

    if plackett_results is None or plackett_results.posterior is None:
        raise ValueError("You must run the model first")

    posterior = plackett_results.posterior
    y95 = posterior.quantile(q=0.95, axis=0)
    y05 = posterior.quantile(q=0.05, axis=0)
    y50 = posterior.quantile(q=0.5, axis=0)
    df_boxplot = pd.concat([y05, y50, y95], axis=1)
    df_boxplot.columns = ["y05", "y50", "y95"]
    df_boxplot["Approaches"] = posterior.columns

    y50 = df_boxplot["y50"].tolist()
    y05 = (df_boxplot["y50"] - df_boxplot["y05"]).tolist()
    y95 = (df_boxplot["y95"] - df_boxplot["y50"]).tolist()
    sizes = [y05, y95]

    fig = df_boxplot.plot.scatter(x="Approaches", y="y50", rot=90, ylabel="Probability")
    plt.tight_layout()
    fig.errorbar(df_boxplot["Approaches"], y50, yerr=sizes, solid_capstyle="projecting", capsize=5, fmt="none")
    fig.grid(linestyle="--")
    fig.xaxis.set_label_text("")
    fig.xaxis.set_major_formatter(ticker.FixedFormatter(df_boxplot["Approaches"]))
    if save_path is not None:
        fig.figure.savefig(save_path)

    return fig.figure

if __name__ == "__main__":
    get_dataset("bpi2012a", store_path=None, extension="xes")


"""
if __name__ == "__main__":
    import os
    print(os.getcwd())
    results, _ = load_results_plackett_luce()
    print(results)

    # Plot
    ax = results.plot(kind='bar', figsize=(15, 10))

    # Add some labels and title
    # Add some labels and title
    plt.xlabel('Logs')
    plt.ylabel('Performance')
    plt.title('Performance by Logs across Different Approaches')

    plt.legend(title='Approaches', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig('performance_by_logs.png', dpi=300, format='png', bbox_inches='tight')

    # Show the plot
    plt.show()
"""


"""
if __name__ == "__main__":
    # Example of usage
    result_matrix = pd.DataFrame([[0.75, 0.6, 0.8], [0.8, 0.7, 0.9], [0.9, 0.8, 0.7]])
    plackett_ranking = PlackettLuceRanking(result_matrix, ["a1", "a2", "a3"])
    results = plackett_ranking.run(n_chains=10, num_samples=300000, mode="max")
    plot = plot_posteriors_plackett(results, save_path=None)
    print(plot)
"""