import numpy as np
import numpy.linalg as la
from scipy.stats import skew, moment, norm
from scipy.special import erfcx, erfc

# import self-made tools
import OPT as OPT
# import Tools.Gauss as Gauss
from LocationScaleFamily import LSF
from SpecialFunctions import log_erfc

################################ Standard EMG #################################
class STD:

    def __init__(self, a = 1, optA = False):
        self.a = a  # alpha = lambda * sigma in standard parameterization of EMG model
        self.optA = optA   # Boolean variables tracking whether or not a is optimized
        # optA could be used as a template parameter if we assume that it is known at compile time

    # should check for positivity
    def setA(self, a):
        self.a = a

    def getA(self):
        return self.a

    def setOptA(self, optA):
        self.optA = optA

    def getOptA(self):
        return self.optA

    # define operators on EMG distribution
    def __add__(self, other):
        return STD(self.a + other.a, self.optA)

    def __sub__(self, other):
        return STD(self.a - other.a, self.optA)

    def __mul__(self, scalar):
        return STD(self.a * scalar, self.optA)

    def __truediv__(self, other):
        return STD(self.a / other.a, self.optA)

    def norm(self):
        return la.norm(self.a)

    def print(self):
        print('a = ' + str(self.a))
        print('optA = ' + str(self.optA))

    def isValid(self):
        return True if self.a > 0 else False

    def makeValid(self, thres = 1e-6):
        self.a = max(self.a, thres)

    def assign(self, other):
        self.a = other.a

    # generate standard EMG distributed samples
    def genSamples(self, size = 1):
        return np.random.standard_normal(size) + np.random.exponential(scale = 1/self.a, size = size)

    # standard negative log EMG density
    # alpha = lambda * sigma
    def negLogDen(self, x):
        a = self.a
        nld = -np.log(a/2) - a**2 / 2 + a*x - log_erfc( (a-x) / np.sqrt(2) )
        return nld

    def density(self, x):
        return np.exp(-self.negLogDen(x))

    def isConvexInA(self):
        return (True if self.a < 1 else False)

    def _get_d(self, x):
        a = self.a
        d = (a-x) / np.sqrt(2)
        return d

    def _get_de(self, x):
        d = self._get_d(x)
        e = 1 / erfcx(d)
        return d, e

    # first derivative of negative log density w.r.t. x
    def gradX(self, x):
        a = self.a
        d, e = self._get_de(x)
        return a - np.sqrt(2/np.pi) * e

    # second derivative of standard negative log density w.r.t. x
    def gradX2(self, x):
        a = self.a
        d, e = self._get_de(x)
        return (2/np.pi * e**2 - 2/np.sqrt(np.pi) * e * d)

    # first derivative of standard negative log density w.r.t. a
    # a: alpha scalar
    def gradA(self, x):
        a = self.a
        d, e = self._get_de(x)
        return -(1/a + a) + x + np.sqrt(2/np.pi) * e

    # second derivative of standard negative log density w.r.t. a
    def gradA2(self, x):
        a = self.a
        d, e = self._get_de(x)
        return (1/a**2 - 1) + 2/np.pi * e**2 - 2/np.sqrt(np.pi) * e * d

    def gradient(self, x):
        return STD(np.sum(self.gradA(x)) if self.optA else 0)

    def laplacian(self, x):
        return STD(np.sum(self.gradA2(x)) if self.optA else 0)

    def scaledGradient(self, x, d = 1e-12):
        return STD(np.sum(self.gradA(x)) / (abs(np.sum(self.gradA2(x)) + d)) if self.optA else 0)

class EMG( LSF ):

    def __init__(self, a = 1, m = 0, s = 1, optA = False, optM = False, optS = False):
        self.m = m  # location parameter
        self.s = s  # scale parameter
        self.std = STD(a, optA)  # the standard distribution on which we are basing the LSFamily

        # Boolean variables tracking which varialbes are optimized
        self.optM = optM
        self.optS = optS

    def getAMS(self):
        return self.std.a, self.getMS()

    def setAMS(self, a, m, s):
        self.std.setA(a)
        self.m = m
        self.s = s

    def setOpt(self, optA, optM, optS):
        self.std.setOptA(optA)
        self.optM = optM
        self.optS = optS

    def setA(self, a):
        self.std.setA(a)

    def setOptA(self, optA):
        self.std.setOptA(optA)

    def gradA(self, x):
        m, s = self.getMS()
        return self.std.gradA((x-m)/s)

    def gradA2(self, x):
        m, s = self.getMS()
        return self.std.gradA2((x-m)/s)

    ###################### Method of Moments Estimates ###########################
    # TODO: Testing of MOM estimates
    # methods of moments estimates for all three parameters
    def MoM(self, x):
        return self.MoMA(x), self.MoMM(x), self.MoMS(x)

    # method of moments estimate mu
    def MoMM(self, x):
        mu = np.mean(x) - self.MoML(x)
        return mu

    # method of moments estimate sigma
    def MoMS(self, x):
        sig = np.sqrt( max( 0, np.var(x) * ( 1 - np.power(skew(x, None)/2, 2/3))))
        return sig

    # method of moments estimate lambda
    def MoML(self, x):
        lam = 1 / (np.std(x) * np.power(skew(x, None) / 2, 1/3))
        return lam

    def MoMA(self, x):
        return self.MoMS(x) * self.MoML(x)


################################# Demonstration ################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    E = EMG()
    E.print()
    n = 1024
    x = E.genSamples(n)


    E.setAMS(.5, 0, 1)
    E.print()

    E.setOptA(True)
    E.setOptM(False)
    E.setOptS(False)

    E.print()
    E.optimize(x)
    E.print()
