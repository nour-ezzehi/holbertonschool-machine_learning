#!/usr/bin/env python3
"""12. Binomial CDF"""


class Binomial():
    """represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = int(n)
            if 0 >= p or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)

            variance = sum((x - mean)**2 for x in data) / (len(data))

            self.n = round(mean / (1 - (variance / mean)))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”
        """

        if not isinstance(k, int):
            k = int(k)
        if k is None or k < 0 or k > self.n:
            return 0

        return (self.fact(self.n) / (self.fact(k) * self.fact(self.n - k))) * (
            (self.p ** k) * (1 - self.p) ** (self.n - k))

    def fact(self, k):
        """function that returns the factorial of k"""
        if k in [0, 1]:
            return 1
        return k * self.fact(k - 1)

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of “successes”
        """

        if not isinstance(k, int):
            k = int(k)
        if k is None or k < 0 or k > self.n:
            return 0

        return sum([(self.fact(self.n) /
                    (self.fact(i) * self.fact(self.n - i)))
                    * ((self.p ** i) * (1 - self.p) ** (self.n - i))
                    for i in range(0, k + 1)])
