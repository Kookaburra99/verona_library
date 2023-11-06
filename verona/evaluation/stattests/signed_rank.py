from typing import List

import numpy as np
from scipy.stats import dirichlet


class BayesianSignedRankTestResult:
    """
    Represents the results of a Bayesian Signed Rank Test.

    Attributes:
        method (str): The name of the statistical method used.
        posterior_probabilities (dict): Probabilities for the left, rope, and right regions.
        approximated (bool): Whether the posterior distribution is approximated.
        parameters (dict): Parameters used in the Bayesian Signed Rank Test.
        posterior (pd.DataFrame): Sampled probabilities for left, rope, and right areas.

    """

    def __init__(self, posterior_probs, approximated, posterior):
        """
        Initializes a new instance of the BayesianSignedRankTestResult class.

        Args:
            method (str): The name of the method used.
            posterior_probs (dict): Probabilities for the left, rope, and right regions.
            approximated (bool): Whether the posterior distribution is approximated.
            parameters (dict): Parameters used in the Bayesian Signed Rank Test.
            posterior (pd.DataFrame): Sampled probabilities for the regions.
        """
        self.posterior_probabilities = posterior_probs
        self.approximated = approximated
        self.posterior = posterior


class BayesianSignedRankTest:
    """
    Bayesian equivalent to Wilcoxon's signed-rank test.

    This function implements the Bayesian version of the signed-rank test as
    presented in Benavoli et al., 2017. This Bayesian test aims to evaluate the
    difference between two related samples (or one sample against a zero null hypothesis)
    and provides probabilities for three regions: left, rope, and right.

    Attributes:
        x (array-like): First sample.
        y (array-like, optional): Second sample. If not provided, x is assumed to be the difference.
        approaches (array-like): Names of the two methods or approaches to be compared.

    Methods:
        run: Executes the Bayesian test.

    References:
    - A. Benavoli, G. Corani, J. Demsar, M. Zaffalon (2017) Time for a Change: a Tutorial for Comparing Multiple
    Classifiers Through Bayesian Analysis. Journal of Machine Learning Research, 18, 1-36.
    - scmamp: Statistical Comparison of Multiple Algorithms in Multiple Problems.
    """

    def __init__(self, x, y, approaches : List[str]):
        """
        Initializes the BayesianSignedRankTest class.

        Args:
            x (array-like): Results for the first approach.
            y (array-like): Results for the second approach.
            approaches (array-like): Names of the two methods or approaches to be compared.
        """

        self.x = x
        self.y = y
        self.approaches = approaches

    def run(self, s=0.5, z0=0, rope=(-1, 1), nsim=100000, seed=None) -> BayesianSignedRankTestResult:

        """
        Args:
            s (float, optional): Scale parameter of the prior Dirichlet Process. Defaults to 0.5.
            z0 (float, optional): Position of the pseudo-observation associated to the prior Dirichlet Process.
            Defaults to 0.
            rope (tuple, optional): Interval for the difference considered as "irrelevant". Defaults to (-1, 1).
            nsim (int, optional): Number of samples used to estimate the posterior distribution. Defaults to 100000.
            seed (int, optional): Optional parameter used to fix the random seed. Defaults to None.

        Returns:
            dict: A dictionary containing:
                - method: a string with the name of the method used
                - posterior_probabilities: a dictionary with the left, rope and right probabilities
                - approximate: a boolean, True if the posterior distribution is approximated (sampled) and
                False if it is exact
                - parameters: a dictionary of parameters used by the method
                - posterior: a list of dictionaries containing the sampled probabilities


        References:
        - A. Benavoli, G. Corani, J. Demsar, M. Zaffalon (2017) Time for a Change: a Tutorial for Comparing Multiple
        Classifiers Through Bayesian Analysis. Journal of Machine Learning Research, 18, 1-36.
        - scmamp: Statistical Comparison of Multiple Algorithms in Multiple Problems.
        """

        if rope[1] < rope[0]:
            print("Warning: Rope parameter should contain ordered limits. They will be swapped.")
            rope = sorted(rope)

        # Convert data to differences
        sample = np.array(self.x) - np.array(self.y)

        # Create the parameter vector for the sampling of the weights
        weights_dir_params = np.append(s, np.ones(len(sample)))
        # Add the pseudo-observation due to the prior to the sample vector
        sample = np.append(z0, sample)

        if seed is not None:
            np.random.seed(seed)

        # Sample from the Dirichlet distribution
        weights = dirichlet.rvs(weights_dir_params, size=nsim)

        # Calculate the terms for all pairs i, j
        sample_matrix = sample[:, None] + sample
        left_matrix = sample_matrix < 2 * rope[0]
        right_matrix = sample_matrix > 2 * rope[1]
        rope_matrix = (sample_matrix >= 2 * rope[0]) & (sample_matrix <= 2 * rope[1])

        posterior_distribution = []

        left_str = "left (" + self.approaches[0] + " < " + self.approaches[1] + ")"
        right_str = "right (" + self.approaches[0] + " > " + self.approaches[1] + ")"
        rope_str = "rope (" + self.approaches[0] + " = " + self.approaches[1] + ")"

        for i in range(nsim):
            weight_matrix = np.outer(weights[i], weights[i])
            left_prob = np.sum(left_matrix * weight_matrix)
            rope_prob = np.sum(rope_matrix * weight_matrix)
            right_prob = np.sum(right_matrix * weight_matrix)

            posterior_distribution.append({left_str: left_prob, rope_str: rope_prob, right_str: right_prob})

        # Calculate posterior probabilities
        posterior_probs = np.array([[d[left_str], d[rope_str], d[right_str]] for d in posterior_distribution])
        max_indices = np.argmax(posterior_probs, axis=1)

        left_prob = np.mean(max_indices == 0)
        rope_prob = np.mean(max_indices == 1)
        right_prob = np.mean(max_indices == 2)

        return BayesianSignedRankTestResult(
            {left_str: left_prob, rope_str: rope_prob, right_str: right_prob},
            True,
            posterior_distribution
        )
