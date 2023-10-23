import numpy as np
import pandas as pd
from scipy.stats import t

class CorrelatedBayesianTTest:
    def __init__(self, x, y, approaches):
        self.x = x
        self.y = y
        self.approaches = approaches

    def run(self, rho=0.2, rope=[-1, 1]):
        """
        Parameters
        ----------
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
        #posterior_probs = pd.DataFrame(posterior_probs)

        """
        additional = {
            "pposterior": ppos,
            "qposterior": qpos,
            "posterior_df": tdist_df,
            "posterior_mean": tdist_mean,
            "posterior_sd": tdist_sd
        }
        """

        return dpos, qpos, posterior_probs