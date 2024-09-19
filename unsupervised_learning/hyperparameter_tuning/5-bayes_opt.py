#!/usr/bin/env python3
""" 5. Bayesian Optimization """
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

    def optimize(self, iterations=100):
        """optimizes the black-box function"""

        all_X = list()

        for i in range(iterations):
            X_next, _ = self.acquisition()
            if X_next in all_X:
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)
            all_X.append(X_next)

        if self.minimize is True:
            Y_opt = np.min(self.gp.Y)
            idx = np.argmin(self.gp.Y)
        else:
            Y_opt = np.max(self.gp.Y)
            idx = np.argmax(self.gp.Y)
        X_opt = self.gp.X[idx]

        return X_opt, Y_opt
