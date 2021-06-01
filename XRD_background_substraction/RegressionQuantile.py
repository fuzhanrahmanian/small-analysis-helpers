import numpy as np

# defining the objective of regression quantile and its subgradient
class RegQuant:

    def __init__(self, q = 1/2):
        self.q = q

    def norm(self):
        return 0

    def negLogDen(self, x):
        ind = np.float64(x > 0)
        return (ind * self.q + (1-ind) * (1-self.q)) * np.abs(x)

    def gradX(self, x):
        ind = np.float64(x > 0)
        # return (ind * self.q + (1-ind) * (1-self.q)) * (np.float64(x > 0) - np.float64(x < 0))
        return (ind * self.q + (1-ind) * (1-self.q)) * np.sign(x)

    # only used if gradient should be normalized
    def gradX2(self, x):
        return np.ones(x.shape)
