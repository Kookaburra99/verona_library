import tempfile
from typing import List, Tuple
import math

from cmdstanpy import CmdStanModel, install_cmdstan, cmdstan_path
import numpy as np

import pandas as pd
import matplotlib.ticker as ticker
import scipy.stats as stats

import os

from barro.evaluation.stattests.stan_codes import STAN_CODE


class HierarchicalBayesianTest:

    @staticmethod
    def pt_scaled(q, df, mean=0, sd=1):
        """
        This function emulates the function pt.scaled from the "metRology" R package. That function computes the
        cumulative distribution function of a T-Student for a given number of degrees of freedom (df). This computation
        is scaled and shifted by mean, and sd, respectively.

        Parameters
        ----------
        q : quantile
        df : degrees of freedom
        mean : scale (mean)
        sd : shift (standard deviation)

        Returns
        -------
        Cumulative distribution function shifted and scaled.
        """
        return stats.t.cdf((q - mean) / sd, df)

    def __init__(self, x_result: List[List[float]], y_result: List[List[float]], approaches: List[str],
                 datasets: List[str]):
        """

        Parameters
        ----------
        x_matrix : first sample, a matrix with the results obtained by the first algorithm (each dataset in a row)
        y_matrix : second sample, a matrix with the results obtained by the second algorithm (each dataset in a row)
        approaches : list of the names of the two approaches that are going to be compared. The first approach corresponds with x_matrix while the second one corresponds with y_matrix.
        datasets : list of the names of the datasets that are fed to the model
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

        if cmdstan_path() is None:
            install_cmdstan()

    def run(self, rope: List, rho=0.2, n_chains=4, num_samples=1000, std_upper=1000, alpha_lower=0.5, alpha_upper=5,
            beta_lower=0.05, beta_upper=0.15, d0_lower=None, d0_upper=None) -> (
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Bayesian hierarchical model for the analysis of two algorithms in multiple datasets.

        Parameters
        ----------
        rope : region of practical equivalence, i.e, interval for the difference considered as "irrelevant"
        n_chains : number of MC chains to be simulated. As half the simulations are used for the warm-up, the total number of simulations will be \code{nchain}*\code{nsim}/2
        num_samples : number of samples (per chain) used to estimate the posterior distribution. Note that, by default, half the simulations are used for the burn-in
        std_upper : factor to set the upper bound for both sigma_i and sigma_0 (see [1] for more details)
        alpha_lower : lower bound for the (uniform) prior for the alpha hyperparameter (see [1] for more details). Default value set at 0.5, as in the original paper
        alpha_upper : upper bound for the (uniform) prior for the alpha hyperparameter (see [1] for more details). Default value set at 5, as in the original paper
        beta_lower : lower bound for the (uniform) prior for the beta hyperparameter (see [1] for more details). Default value set at 0.05, as in the original paper
        beta_upper : upper bound for the (uniform) prior for the beta hyperparameter (see [1] for more details). Default value set at 0.15, as in the original paper
        d0_lower : lower bound for the prior for mu_0. If not provided, the smallest observed difference is used
        d0_upper : upper bound for the prior for mu_0. If not provided, the biggest observed difference is used
        seed : optional parameter used to fix the random seed

        Notes
        -----
        The results includes the typical information relative to the three areas of the posterior density (left, right and rope probabilities), both global and per dataset (in the additional information). Also, the simulation results are included.

        As for the prior parameters, they are set to the default values indicated in [1], except for the bound for the prior distribution of mu_0, which are set to the maximum and minimum values observed in the sample. You should not modify them unless you know what you are doing.

        [1] A. Benavoli, G. Corani, J. Demsar, M. Zaffalon (2017) Time for a Change: a Tutorial for Comparing Multiple Classifiers Through Bayesian Analysis. \emph{Journal of Machine Learning Research}, 18, 1-36.

        Returns
        -------
        global_wins : vector with the left, rope and right probabilities overall
        posterior_distribution : sampled probabilities (see details)
        per_dataset : vector with the left, rope and right probabilities for each dataset
        global_sign : sampled probabilities of mu_0 being positive or negative
        results : complete set of results produced by the Stan program

        Examples
        --------
        >>> x_data = pd.DataFrame([[75.3, 78.3, 60.4], [68.5, 77.5, 76.9], [77.9, 74.5, 80.9], [90, 90, 90]])
        >>> y_data = pd.DataFrame([[74.3, 75.3, 61.4], [65.5, 70.5, 80.9], [79.9, 76.2, 81.9], [90, 90, 90]])
        >>> global_wins, posterior_distribution, per_dataset, global_sign, results = HierarchicalBayesianTest(x_data, y_data)._run_stan(x_data, y_data, [-1, 1], n_chains=10, stan_samples=15000)
        """
        global_wins, posterior_distribution, per_dataset, global_sign, results = self._run_stan(x_matrix=self.x_result,
                                                                                                y_matrix=self.y_result,
                                                                                                rope=rope, rho=rho,
                                                                                                n_chains=n_chains,
                                                                                                stan_samples=num_samples,
                                                                                                std_upper=std_upper,
                                                                                                alpha_lower=alpha_lower,
                                                                                                alpha_upper=alpha_upper,
                                                                                                beta_lower=beta_lower,
                                                                                                beta_upper=beta_upper,
                                                                                                d0_lower=d0_lower,
                                                                                                d0_upper=d0_upper)
        return global_wins, posterior_distribution, per_dataset, global_sign, results

    def _run_stan(self, x_matrix: pd.DataFrame, y_matrix: pd.DataFrame, rope: List, rho=0.2, n_chains=4,
                  stan_samples=1000, std_upper=1000, alpha_lower=0.5, alpha_upper=5, beta_lower=0.05, beta_upper=0.15,
                  d0_lower=None, d0_upper=None, seed=42) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # TODO: refactor this method since it follows the structure of the original code, but it is not very pythonic
        # SCMAMP does not describe rho, so do I.

        assert len(rope) == 2, "The number of elements of rope must be 2"
        assert rope[0] < rope[1], "rope[0] must be less than rope[1]"

        stan_code = STAN_CODE.HIERARCHICAL_TEST

        num_samples = x_matrix.shape[1]
        num_datasets = x_matrix.shape[0]

        experiment_results = x_matrix - y_matrix

        dataset_sds = experiment_results.std(axis=1)
        mean_dataset_sd = dataset_sds.mean()
        scale_factor = mean_dataset_sd  # TODO: ???

        # Scale the crossvalidation results and the rope
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
        fit = model.sample(data=stan_data, chains=n_chains, iter_sampling=int(stan_samples/2), iter_warmup=int(stan_samples/2), seed=42)

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
        global_wins = pd.DataFrame({left_str : prob_left_win, rope_str : prob_rope_win, right_str: prob_right_win}, index=[0])

        import os
        os.remove(temp_file_name)

        return global_wins, posterior_distribution, per_dataset, global_sign, results


if __name__ == "__main__":
    x_data = pd.DataFrame([[75.3, 78.3, 62.4], [68.5, 77.5, 82.9], [79.9, 79.5, 83.9], [95, 93, 92]])
    y_data = pd.DataFrame([[74.3, 75.3, 61.4], [65.5, 70.5, 80.9], [79.9, 76.2, 81.9], [90, 90, 90]])
    global_wins, posterior_distribution, per_dataset, global_sign, results = HierarchicalBayesianTest(x_data, y_data,
                                                                                                      ["a1", "a2"],
                                                                                                      ["d1", "d2",
                                                                                                       "d3", "d4"])._run_stan(
        x_data, y_data, rope=[-1, 1], n_chains=10, stan_samples=90000)
    print("Global wins: ", global_wins)
    print("Posterior distribution: ", posterior_distribution)
    print("Per dataset: ", per_dataset)
    print("Glboal sign: ", global_sign)
    print("Results: ", results)
