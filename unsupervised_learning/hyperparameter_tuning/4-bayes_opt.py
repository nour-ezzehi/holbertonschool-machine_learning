#!/usr/bin/env python3
""" 4. Bayesian Optimization - Acquisition """
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """performs Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(
        self, f, X_init, Y_init, bounds,
        ac_samples,
        l=1,
        sigma_f=1,
        xsi=0.01,
        minimize=True,
    ):
        """"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(
            bounds[0], bounds[1], num=ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """calculates the next best sample location
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            Y_sample_opt = np.min(self.gp.Y)
            imp = Y_sample_opt - mu - self.xsi
        else:
            Y_sample_opt = np.max(self.gp.Y)
            imp = mu - Y_sample_opt - self.xsi

        A = imp / sigma

        EI = imp * norm.cdf(A) + sigma * norm.pdf(A)

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
