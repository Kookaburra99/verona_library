from typing import List

import numpy as np
import pandas as pd
from scipy.stats import t

class BayesianTTestResult:
    """
    Represents the results of a Bayesian Correlated t-test.

    Attributes:
        posterior_probabilities (dict): A dictionary containing the probabilities
            for the left, rope, and right regions of the posterior distribution.
        approximated (bool): Indicates if the posterior distribution is approximated 
            (True if approximated, e.g., by MCMC sampling, and False if exact).
        parameters (dict): The parameters used for running the Bayesian t-test,
            specifically 'rho' and 'rope'.
        posterior (dict): A dictionary containing the density, cumulative, and 
            quantile functions for the posterior distribution.
        additional (dict): Additional details about the posterior distribution,
            such as degrees of freedom, mean, and standard deviation.

    Methods:
        summarize(): Provides a summary of the test results, e.g., mean, standard deviation,
            and posterior probabilities.
    """

    def __init__(self, posterior_probs, approximated, parameters, posterior, additional):
        """
        Initializes a new instance of the BayesianTTestResult class.

        Args:
            posterior_probs (dict): Probabilities for the left, rope, and right regions.
            approximated (bool): Whether the posterior distribution is approximated.
            parameters (dict): Parameters used in the Bayesian t-test.
            posterior (dict): Functions related to the posterior distribution.
            additional (dict): Additional details about the posterior distribution.
        """
        self.posterior_probabilities = posterior_probs
        self.approximated = approximated
        self.parameters = parameters
        self.posterior = posterior
        self.additional = additional

class CorrelatedBayesianTTest:
    """
    Bayesian equivalent to the correlated t-test.

    This class offers a Bayesian alternative to the traditional frequentist correlated t-test,
    often used for comparing the means of two paired samples to determine if they come from 
    populations with equal means. It extends the paired Student's t-test to a Bayesian framework,
    offering a richer set of inferences that can be drawn from the data. 
    
    In particular, this implementation follows the Bayesian correlated t-test as described by Benavoli et al., 2017,
    which provides not just point estimates, but also credible intervals and posterior probabilities
    that can more informatively capture the uncertainty around the true parameter values.

    Attributes:
        x (array-like): First sample.
        y (array-like): Second sample. If not provided, x is assumed to be the difference. approaches (array-like): Methods or approaches to be compared.

    Methods:
        run: Executes the Bayesian t-test.

    Example:
        >>> sample1 = [random.gauss(1, 1) for _ in range(25)]
        >>> sample2 = [random.gauss(1.2, 1) for _ in range(25)]
        >>> test = CorrelatedBayesianTTest(sample1, sample2, ["Method1", "Method2"])
        >>> test.run(rho=0.1, rope=[-1, 1])

    References:
    - A. Benavoli, G. Corani, J. Demsar, M. Zaffalon (2017) Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis. Journal of Machine Learning Research, 18, 1-36.
    - scmamp: Statistical Comparison of Multiple Algorithms in Multiple Problems.

    """

    def __init__(self, x, y, approaches : List[str]):
        """
        Initializes the CorrelatedBayesianTTest class.

        Args:
            x (array-like): Results for the first approach.
            y (array-like): Results for the second approach.
            approaches (array-like): Names of the two methods or approaches to be compared.
        """

        self.x = x
        self.y = y
        self.approaches = approaches

    def run(self, rho=0.2, rope=[-1, 1]) -> BayesianTTestResult:
        """
        Executes the Bayesian t-test.

        Args:
            rho (float, optional): Correlation factor between the paired samples. Defaults to 0.2.

            - A rho of 0 implies that the paired samples are entirely independent, essentially converting the test into a standard Bayesian t-test.
            - A rho of 1 implies that the paired samples are perfectly correlated, making the test trivial.
            - Values between 0 and 1 adjust the test to account for the degree of correlation between the paired samples. For instance, in the context of machine learning, rho could be set to the proportion of the test set size to the total dataset size to account for data reuse across different folds in k-fold cross-validation.

            rope (list, optional): Interval for the difference considered as "irrelevant" or "equivalent". Defaults to [-1, 1].
            
        Returns:
            BayesianTTestResult: An instance of the BayesianTTestResult class that contains the following:

            - `posterior_probabilities`: Probabilities for the left, rope, and right regions.
            - `approximated`: Whether the posterior distribution is approximated.
            - `parameters`: Parameters used in the Bayesian t-test.
            - `posterior`: Functions related to the posterior distribution.
            - `additional`: Additional details about the posterior distribution.


        Details:
            The default value for rho is 0.2, which accounts for a 20% split in the testing set.
        """


        # Check the rope parameter
        if rope[1] < rope[0]:
            print("Warning: The rope parameter is not ordered. They will be swapped to proceed.")
            rope = sorted(rope)

        # Check the correlation factor
        if rho >= 1:
            raise ValueError("The correlation factor must be strictly smaller than 1!")

        # Convert data to differences
        sample = self.x - self.y

        # Compute mean and standard deviation
        sample_mean = np.mean(sample)
        sample_sd = np.std(sample, ddof=1)  # ddof=1 to use sample standard deviation
        n = len(sample)

        tdist_df = n - 1
        tdist_mean = sample_mean
        tdist_sd = sample_sd * np.sqrt(1 / n + rho / (1 - rho))

        # Functions for posterior density, cumulative, and quantile
        dpos = lambda mu: t.pdf((mu - tdist_mean) / tdist_sd, tdist_df)
        ppos = lambda mu: t.cdf((mu - tdist_mean) / tdist_sd, tdist_df)
        qpos = lambda q: t.ppf(q, tdist_df) * tdist_sd + tdist_mean

        # Compute posterior probabilities
        left_prob = ppos(rope[0])
        rope_prob = ppos(rope[1]) - left_prob
        right_prob = 1 - ppos(rope[1])

        left_str = "left (" + self.approaches[0] + " < " + self.approaches[1] + ")"
        right_str = "right (" + self.approaches[0] + " > " + self.approaches[1] + ")"
        rope_str = "rope (" + self.approaches[0] + " = " + self.approaches[1] + ")"

        posterior_probs = {
            left_str: left_prob,
            rope_str: rope_prob,
            right_str: right_prob
        }

        result = BayesianTTestResult(
            posterior_probs,
            False,  # True if you use sampling methods like MCMC
            {"rho": rho, "rope": rope},
            {"density_function": dpos, "cumulative_function": ppos, "quantile_function": qpos},
            {
                "posterior_df": tdist_df,
                "posterior_mean": tdist_mean,
                "posterior_sd": tdist_sd
            }
        )

        return result