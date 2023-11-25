import tempfile
import sys
from typing import List

import numpy as np
import pandas as pd
import scipy.stats as stats
from cmdstanpy import CmdStanModel, install_cmdstan, cmdstan_path

from verona.evaluation.stattests.stan_codes import STAN_CODE


class BayesianHierarchicalResults:
    def __init__(self, approximated, global_wins, posterior_distribution, per_dataset, global_sign, raw_results):
        """
        A class to store the results of the Bayesian hierarchical test.

        Args:
            global_wins (pd.DataFrame): A DataFrame containing the global winning probabilities for each condition
            (left, right, rope).
            posterior_distribution (pd.DataFrame): A DataFrame with the posterior distribution probabilities
            (left, rope, right).
            per_dataset (pd.DataFrame): A DataFrame with per-dataset statistics.
            global_sign (pd.DataFrame): A DataFrame containing the global sign probabilities (positive, negative).
            raw_results (pd.DataFrame): A DataFrame containing the raw results from the sampling.
        """

        self.global_wins = global_wins
        self.approximated = approximated
        self.posterior_distribution = posterior_distribution
        self.per_dataset = per_dataset
        self.global_sign = global_sign
        self.raw_results = raw_results


class HierarchicalBayesianTest:

    @staticmethod
    def pt_scaled(q, df, mean=0, sd=1):
        """
        This function emulates the function pt.scaled from the "metRology" R package. That function computes the
        cumulative distribution function of a T-Student for a given number of degrees of freedom (df). This computation
        is scaled and shifted by mean, and sd, respectively.

        Args:
            q : quantile
            df : degrees of freedom
            mean : scale (mean)
            sd : shift (standard deviation)

        Returns:
            Cumulative distribution function shifted and scaled.
        """
        return stats.t.cdf((q - mean) / sd, df)

    def __init__(self, x_result: pd.DataFrame, y_result: pd.DataFrame, approaches: List[str], datasets: List[str]):
        """
        Args:
            x_result: A 2D dataframe containing the performance metrics obtained by the first algorithm. Each row in
            this matrix represents a unique dataset, and each column corresponds to the result of an individual fold in
            a k-fold cross-validation for that dataset.
            y_result: A 2D dataframe containing the performance metrics obtained by the second algorithm. The structure
            is the same as `x_result`, where rows correspond to datasets and columns to individual k-fold
            cross-validation results.
            approaches: A list containing the names of the two algorithms being compared. The first element in the list
            should correspond to the algorithm associated with `x_result`, and the second element should correspond to
            the algorithm associated with `y_result`.
            datasets: A list containing the names of the datasets used in the model. The order should match the row
            order in `x_result` and `y_result`.
        """

        assert len(approaches) == 2, "The number of names of the approaches is not 2"
        assert len(datasets) > 0, "The number of datasets using for comparing the approaches must be greater than 0"

        self.x_result = pd.DataFrame(x_result)
        self.y_result = pd.DataFrame(y_result)
        self.approaches = approaches
        self.datasets = datasets

        assert self.x_result.shape[0] == self.y_result.shape[0], "The number of rows of both matrices do not match"
        assert self.x_result.shape[1] == self.y_result.shape[1], "The number of columns of both matrices do not match"
        assert self.x_result.shape[0] == len(self.datasets)

        try:
            cmdstan_path()
        except:
            install_cmdstan()

    def run(self, rope=[-1,1], rho=0.2, n_chains=4, num_samples=300000, std_upper=1000, alpha_lower=0.5, alpha_upper=5,
            beta_lower=0.05, beta_upper=0.15, d0_lower=None, d0_upper=None, suppress_output=False) -> BayesianHierarchicalResults:
        """
        Executes a Bayesian hierarchical model tailored for comparing the performance of two machine learning algorithms
        across multiple datasets based on cross-validation results.

        This model employs a two-level hierarchical structure to account for both dataset-specific variations and global
        trends in the performance differences between the two algorithms. Specifically, it treats the performance
        metrics from each dataset as arising from a dataset-specific distribution, which in turn is governed by global
        hyperparameters.

        The model can work directly on a variety of metrics obtained through cross-validation, thereby providing a
        comprehensive statistical insight into the comparative evaluation. This is a significant advantage over
        frequentist methods which may not fully capture the uncertainty in the metrics.

        This implementation is ported from the Bayesian hierarchical model implemented in [2], which provides posterior
        probabilities for a richer interpretation of the comparison.


        Args:
            rope (List, default=[-1, 1]): Region of Practical Equivalence, defines the interval within which performance
            differences are considered "irrelevant" or "insignificant".
            rho (float, default=0.2): A hyperparameter representing the correlation factor across datasets. A higher
            value indicates stronger correlation between datasets in terms of algorithm performance.
            n_chains (int, default=4): The number of Markov Chains to be used in the simulation. Half of the simulations
            are used for warm-up.
            num_samples (int, default=300000): The total number of samples (per chain) used for estimating the posterior
            distribution. By default, half of these samples are used for the burn-in phase.
            std_upper (int, default=1000): A scaling factor that sets the upper bounds for the hyperparameters sigma_i
            and sigma_0, which represent dataset-specific and global variability, respectively.
            alpha_lower (float, default=0.5): Lower bound for the uniform prior of the alpha hyperparameter, which
            models the global variance.
            alpha_upper (float, default=5): Upper bound for the uniform prior of the alpha hyperparameter.
            beta_lower (float, default=0.05): Lower bound for the uniform prior of the beta hyperparameter, which models
            dataset-specific variances.
            beta_upper (float, default=0.15): Upper bound for the uniform prior of the beta hyperparameter.
            d0_lower (float, optional): Lower bound for the prior distribution of mu_0, the grand mean of performance
            differences. If not provided, the smallest observed difference is used as the lower bound.
            d0_upper (float, optional): Upper bound for the prior distribution of mu_0. If not provided, the largest
            observed difference is used as the upper bound.
            suppress_output (bool, optional): Whether or not suppress the tqdm outputs.

        Notes:
            The results includes the typical information relative to the three areas of the posterior density (left,
            right and rope probabilities), both global and per dataset (in the additional information). Also, the
            simulation results are included.

            As for the prior parameters, they are set to the default values indicated in [1,2], except for the bound for
            the prior distribution of mu_0, which are set to the maximum and minimum values observed in the sample. You
            should not modify them unless you know what you are doing.

            [1] A. Benavoli, G. Corani, J. Demsar, M. Zaffalon (2017) Time for a Change: a Tutorial for Comparing
            Multiple Classifiers Through Bayesian Analysis. \emph{Journal of Machine Learning Research}, 18, 1-36.
            [2] Borja Calvo and Guzmán Santafé (2016) scmamp: Statistical Comparison of Multiple Algorithms in Multiple
            Problems. *The R Journal*, 8(1), 248-256. [DOI: 10.32614/RJ-2016-017](https://doi.org/10.32614/RJ-2016-017)

        Returns:
            BayesianHierarchicalResults
                An object containing the following attributes:
                - `approximated` : Boolean value that indicates whether the posterior distribution is approximated
                (True in this case).
                - `global_wins`: DataFrame containing the global winning probabilities for each condition (left, right,
                rope).
                - `posterior_distribution`: DataFrame with the posterior distribution probabilities (left, rope, right).
                - `per_dataset`: DataFrame with per-dataset statistics.
                - `global_sign`: DataFrame containing the global sign probabilities (positive, negative).
                - `raw_results`: DataFrame containing the raw results from the sampling.

        Examples:
            >>> x_data = pd.DataFrame([[75.3, 78.3, 60.4], [68.5, 77.5, 76.9], [77.9, 74.5, 80.9], [90, 90, 90]])
            >>> y_data = pd.DataFrame([[74.3, 75.3, 61.4], [65.5, 70.5, 80.9], [79.9, 76.2, 81.9], [90, 90, 90]])
            >>> results = HierarchicalBayesianTest(x_data, y_data, approaches=["approach 1", "approach 2"],
            datasets=["d1", "d2", "d3", "d4"]).run([-1, 1])
            >>> print("Global wins: ", results.global_wins)
            Global wins:     left (approach 1 < approach 2)  rope (approach 1 = approach 2)  right (approach 1 >
                approach 2)
                                    0.809272                             0.0                         0.190728
            >>> print("Per dataset: ", results.per_dataset.iloc[:, 1:4])
            Per dataset:      left (approach 1 < approach 2)  rope (approach 1 = approach 2)  right (approach 1 >
                approach 2)
            d1                        0.207137                        0.503125                         0.289738
            d2                        0.279955                        0.419848                         0.300197
            d3                        0.619778                        0.337055                         0.043167
            d4                        0.039968                        0.937468                         0.022563
        """
        results = self._run_stan(x_matrix=self.x_result, y_matrix=self.y_result, rope=rope, rho=rho,
                              n_chains=n_chains, stan_samples=num_samples, std_upper=std_upper,
                              alpha_lower=alpha_lower, alpha_upper=alpha_upper, beta_lower=beta_lower,
                              beta_upper=beta_upper, d0_lower=d0_lower, d0_upper=d0_upper, suppress_output=suppress_output)
        return results


    def _run_stan(self, x_matrix: pd.DataFrame, y_matrix: pd.DataFrame, rope: List, rho=0.2, n_chains=4,
                  stan_samples=1000, std_upper=1000, alpha_lower=0.5, alpha_upper=5, beta_lower=0.05, beta_upper=0.15,
                  d0_lower=None, d0_upper=None, seed=42, suppress_output=False) -> BayesianHierarchicalResults:
        # TODO: refactor this method since it follows the structure of the original code, but it is not very pythonic

        assert len(rope) == 2, "The number of elements of rope must be 2"
        assert rope[0] < rope[1], "rope[0] must be less than rope[1]"

        stan_code = STAN_CODE.HIERARCHICAL_TEST

        num_samples = x_matrix.shape[1]
        num_datasets = x_matrix.shape[0]

        experiment_results = x_matrix - y_matrix

        dataset_sds = experiment_results.std(axis=1)
        mean_dataset_sd = dataset_sds.mean()
        scale_factor = mean_dataset_sd  # TODO: ???

        # Scale the cross-validation results and the rope
        sample_matrix = experiment_results / mean_dataset_sd
        rope[0] = rope[0] / mean_dataset_sd
        rope[1] = rope[1] / mean_dataset_sd

        # scmamp: In case there is any dataset with 0 variance, add a small value to avoid problems
        # taking care to not alter the mean value
        # me: Taking care of not altering the mean value means that if the number of samples is odd, one of the
        # elements of the sample will NOT be altered (the one in the middle). Therefore, adding and subtracting
        # the same noise in different positions of the array (as long as the noise is within the rope, to not alter
        # the final statistical test results) keeps the mean value of the sample unchanged.
        for id, sd in enumerate(dataset_sds):
            if sd == 0:
                noise = np.random.uniform(rope[0], rope[1], size=int(num_samples / 2))
                sample_matrix.iloc[id, :int((num_samples / 2))] += noise
                # The +1 add takes care of the case of an odd list of samples, and does it work for even lists too.
                sample_matrix.iloc[id, int(((num_samples + 1) / 2)):] -= noise

        dataset_sds = sample_matrix.std(axis=1)
        mean_dataset_sd = dataset_sds.mean()

        if num_samples == 1:
            dataset_mean_sd = sample_matrix.std()
        else:
            dataset_mean_sd = sample_matrix.mean(axis=1).std()

        if d0_lower is None:
            d0_lower = sample_matrix.abs().to_numpy().max() * -1

        if d0_upper is None:
            d0_upper = sample_matrix.abs().to_numpy().max()

        stan_data = {
            "deltaLow": d0_lower,
            "deltaHi": d0_upper,
            "stdLow": 0,
            "stdHi": mean_dataset_sd * std_upper,
            "std0Low": 0,
            "std0Hi": dataset_mean_sd * std_upper,
            "Nsamples": num_samples,
            "q": num_datasets,
            "x": np.array(sample_matrix),
            "rho": rho,
            "upperAlpha": alpha_upper,
            "lowerAlpha": alpha_lower,
            "upperBeta": beta_upper,
            "lowerBeta": beta_lower
        }

        #posterior = stan.build(stan_code, data=stan_data, random_seed=seed)
        #fit = posterior.sample(num_chains=n_chains, num_samples=stan_samples)

        with tempfile.NamedTemporaryFile(suffix='.stan', delete=False) as temp:
            temp.write(stan_code.encode('utf-8'))
            temp_file_name = temp.name  # Save the filename to use later

        model = CmdStanModel(stan_file=temp_file_name)
        fit = model.sample(data=stan_data, chains=n_chains, iter_sampling=int(stan_samples/2),
                           iter_warmup=int(stan_samples/2), seed=42, show_progress=not suppress_output)

        results = fit.draws_pd()

        # The cols are named like: "diff.1.3", but we get diff as name.
        # iterate over the columns "templates" and detect them in the dataframe
        # Then, drop those columns from the df.
        cols_to_drop_df = []
        cols_to_drop = ["diff", "diagQuad", "oneOverSigma2", "nuMinusOne", "logLik"]

        for col in results.columns:
            for col_to_drop_template in cols_to_drop:
                if col_to_drop_template in col:
                    cols_to_drop_df.append(col)

        results = results.drop(cols_to_drop_df, axis=1)
        # Get the delta.X columns, that represent each of the datasets.
        delta_cols = [col for col in results.columns if ("delta" in col and col != "delta0")]
        delta_df = results[delta_cols]
        left = delta_df[delta_df < rope[0]].count() / delta_df.count()
        right = delta_df[delta_df > rope[1]].count() / delta_df.count()
        rope = delta_df[(delta_df > rope[0]) & (delta_df < rope[1])].count() / delta_df.count()

        probs_per_dataset = pd.DataFrame({"left": left, "rope": rope, "right": right})

        aux = HierarchicalBayesianTest.pt_scaled(rope[1], df=results["nu"], mean=results["delta0"], sd=results["std0"])
        cum_left = HierarchicalBayesianTest.pt_scaled(rope[0], df=results["nu"], mean=results["delta0"],
                                                      sd=results["std0"])
        cum_rope = aux - cum_left
        cum_right = 1 - aux

        posterior_distribution = pd.DataFrame({"left": cum_left, "rope": cum_rope, "right": cum_right})

        # Get the probabilities according to the counts
        left_wins = (cum_left > cum_right) & (cum_left > cum_rope)
        right_wins = (cum_right > cum_left) & (cum_right > cum_rope)
        rope_wins = (left_wins | right_wins)
        rope_wins = np.array([not x for x in rope_wins])
        prob_left_win = left_wins.mean()
        prob_right_win = right_wins.mean()
        prob_rope_win = rope_wins.mean()

        positive_d0 = results["delta0"] > 0
        prob_positive = positive_d0.mean()
        prob_negative = 1 - prob_positive

        # Get the results ready
        per_dataset = delta_df.mean(axis=0) * scale_factor
        per_dataset = pd.concat([per_dataset, probs_per_dataset], axis=1)
        left_str = "left (" + self.approaches[0] + " < " + self.approaches[1] + ")"
        right_str = "right (" + self.approaches[0] + " > " + self.approaches[1] + ")"
        rope_str = "rope (" + self.approaches[0] + " = " + self.approaches[1] + ")"
        per_dataset.rename(columns={
            per_dataset.columns[0]: "mean_delta",
            "left" : left_str,
            "rope" : rope_str,
            "right" : right_str
        }, inplace=True)
        per_dataset.index = self.datasets

        global_sign = pd.DataFrame({"negative": prob_negative, "positive": prob_positive}, index=[0])
        global_wins = pd.DataFrame({left_str: prob_left_win, rope_str: prob_rope_win, right_str: prob_right_win},
                                   index=[0])

        import os
        os.remove(temp_file_name)

        return BayesianHierarchicalResults(True, global_wins, posterior_distribution, per_dataset, global_sign, results)


if __name__ == "__main__":
    x_data = pd.DataFrame([[75.3, 78.3, 60.4], [68.5, 77.5, 76.9], [77.9, 74.5, 80.9], [90, 90, 90]])
    y_data = pd.DataFrame([[74.3, 75.3, 61.4], [65.5, 70.5, 80.9], [79.9, 76.2, 81.9], [90, 90, 90]])
    results = HierarchicalBayesianTest(x_data, y_data, approaches=["approach 1", "approach 2"],
                                       datasets=["d1", "d2", "d3", "d4"]).run([-1, 1])
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    print("Global wins: ", results.global_wins)
    print("Per dataset: ", results.per_dataset.iloc[:, 1:4])
