import pandas as pd

from verona.evaluation.stattests.plackettluce import PlackettLuceRanking
from verona.visualization import stattests


def test_plot_posteriors_plackett():
    result_matrix = pd.DataFrame([[0.75, 0.6, 0.8], [0.8, 0.7, 0.9], [0.9, 0.8, 0.7]])

    plackett_ranking = PlackettLuceRanking(result_matrix, ["a1", "a2", "a3"])
    results = plackett_ranking.run(n_chains=10, num_samples=300000, mode="max")

    plot = stattests.plot_posteriors_plackett(results, save_path=None)
    plot.show()
