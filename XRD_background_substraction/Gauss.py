import numpy as np
from LocationScaleFamily import LSF

############################## Standard Normal ################################
class STD():

    def negLogDen(self, x):
        return x**2/2 + np.log(2*np.pi) / 2

    def gradX(self, x):
        return x

    def gradX2(self, x):
        return np.ones(x.shape)

    def gradient(self, x):
        return self

    def laplacian(self, x):
        return self

    def scaledGradient(self, x, d = 1e-12):
        return self

    def print(self):
        return None

    def isValid(self):
        return True

    def makeValid(self):
        return self

    def assign(self, other):
        return None

    def genSamples(self, size = 1):
        return np.random.standard_normal(size)

    # define operators on standard normal distribution
    def __add__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __mul__(self, scalar):
        return self

    def __truediv__(self, other):
        return self

    def norm(self):
        return 0

    # functions regarding the standard normal distribution
    def density(self, x):
        return 1/(np.sqrt(2*np.pi)) * np.exp( - x**2 / 2 )

    def denGradX(self, x):
        return -x * self.density(x)

    def denGradX2(self, x):
        return - self.density(x) + (-x * self.denGradX(x))

########################## Gaussian Distribution ###############################
class Gauss( LSF ):

    def __init__(self, m = 0, s = 1, optM = False, optS = False):
        self.std = STD()
        self.m = m
        self.s = s

        self.optM = optM
        self.optS = optS

# ########################### Multivariate Normal ################################
# class MultiGauss():
#
#     def __init__(self, m = 0, var = np.diag(1), cov = None, optM = False, optVar = False, optCov):
#
#         self.std = STD()
#
#         self.d = 1 # dimension of space
#         self.m = m
#         self.var = s
#         self.cov = s
#
#         self.optM = optM
#         self.optVar = optVar
#         self.optCov = optS
#
#     def genSamples(self, size = 1):
#         return np.var * np.random.standard_normal(size)
#
#         # # covariance
#         # lowRank = 2*(np.random.random(k) < 1/2) -1
#         # # diag = np.eye(k)
#         # # CoVar = (.05)**2*diag + (.02)**2 * lowRank
#         # # print(CoVar)
#         # # rv = multivariate_normal(mean=m, cov=CoVar)
#         # # print(rv)
#         # # print(rv.rvs())
#         import matplotlib.pyplot as plt
#         # # plt.plot(x, M.densities(x), x, M.density(x))
#         # # for i in range(32):
#         # #     # ms = rv.rvs()
#         # #     ms = (.08) * lowRank * np.random.randn() + m
#         # #     ms = (.08) * np.random.standard_normal(m.shape) + m
#         # #     print(ms)
#         # #     M.setM(ms)
#         # #     plt.plot(x, M.densities(x), x, M.density(x))
#         # #     plt.show()

# G = Gauss(.5, .1)
# G.print()
# n = 128
# x = G.genSamples(n)
#
# G2 = Gauss()
# G2.setOptM(True)
# G2.setOptS(True)
# G2.optimize(x)
# G2.print()
