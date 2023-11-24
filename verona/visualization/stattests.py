import pandas as pd
import plotly.graph_objects as go

from verona.evaluation.stattests.plackettluce import PlackettLuceResults, PlackettLuceRanking


def plot_posteriors_plackett(plackett_results: PlackettLuceResults, save_path=None, use_latex=False):
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

    if use_latex:
        probability_str = "$\\text{Probability}$"
        approaches = [r"$\text{ " + str(col) + "}$" for col in posterior.columns]
        df_boxplot["Approaches"] = approaches
    else:
        probability_str = "Probability"
        df_boxplot["Approaches"] = posterior.columns

    y50 = df_boxplot["y50"]
    yerr_lower = df_boxplot["y50"] - df_boxplot["y05"]
    yerr_upper = df_boxplot["y95"] - df_boxplot["y50"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_boxplot["Approaches"], y=y50,
        error_y=dict(
            type='data', symmetric=False,
            array=yerr_upper, arrayminus=yerr_lower
        ),
        mode='markers'
    ))

    fig.update_layout(
        xaxis_title="",
        yaxis_title=probability_str,
        xaxis=dict(tickmode='array', tickvals=list(range(len(df_boxplot["Approaches"]))),
                   ticktext=df_boxplot["Approaches"]),
        margin={'l': 10, 'r': 5, 't': 5, 'b': 5}
    )

    if save_path is not None:
        fig.write_image(save_path)

    return fig
