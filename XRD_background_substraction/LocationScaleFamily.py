# to be template class for arbitrary location scale probability distribution
import numpy as np
import numpy.linalg as la
import OPT as OPT

# takes in a standard distribution and computes all quantities related to its location scale family
class LSF:
# TODO: have subclasses for density and nll for LSF and GM
    def __init__(self, std, m = 0, s = 1, optM = False, optS = False):
        self.std = std  # the standard distribution on which we are basing the LSFamily
        self.m = m  # location parameter
        self.s = s  # scale parameter

        # Boolean variables tracking which varialbes are optimized
        self.optM = optM
        self.optS = optS

    def print(self):
        print('m = ' + str(self.m))
        print('optM =' + str(self.optM))
        print('s = ' + str(self.s))
        print('optS =' + str(self.optS))
        self.std.print()

    # checks if the distribution is valid
    def isValid(self):
        if np.all(self.s > 0) and self.std.isValid():
            return True
        else:
            return False

    def makeValid(self, thres = 1e-6):
        self.std.makeValid()
        self.s = np.maximum(thres, self.s)
        return self

    # assign the variables of another LSFamily object to the current one
    def assign(self, other):
        self.std.assign(other.std)
        self.m = other.m
        self.s = other.s

    # define operators on location scale family
    def __add__(self, other):
        return LSF(self.std + other.std, self.m + other.m, self.s + other.s, self.optM, self.optS)

    def __sub__(self, other):
        return LSF(self.std - other.std, self.m - other.m, self.s - other.s, self.optM, self.optS)

    # ideally would have scalar and elementwise multiplication
    def __mul__(self, scalar):
        return LSF(self.std * scalar, self.m * scalar, self.s * scalar, self.optM, self.optS)

    def __truediv__(self, other):
        return LSF(self.std / other.std, self.m / other.m, self.s / other.s, self.optM, self.optS)

    def norm(self):
        return np.sqrt(self.std.norm()**2 + la.norm(self.m)**2 + la.norm(self.s)**2)

    def getMS(self):
        return self.m, self.s

    def setMS(self, m, s):
        self.m = m
        self.s = s

    def setM(self, m):
        self.m = m

    def getM(self):
        return self.m

    def setS(self, s):
        self.s = s

    def getS(self):
        return self.s

    def setOptM(self, optM):
        self.optM = optM

    def setOptS(self, optS):
        self.optS = optS

    def setOpt(self, optM, optS):
        self.optM = optM
        self.optS = optS

    def genSamples(self, size = 1):
        if size > 1:
            return self.s * self.std.genSamples(size) + self.m
        elif size == 1:
            return self.s * self.std.genSamples(self.m.shape) + self.m

    # negative log density could be its own class
    def negLogDen(self, x):
        m, s = self.getMS()
        return self.std.negLogDen((x-m)/s) + np.log(s)

    # derivatives of negLogDen
    def gradM(self, x):
        m, s = self.getMS()
        return -1/s * self.std.gradX((x-m)/s)

    def gradM2(self, x):
        m, s = self.getMS()
        return 1/s**2 * self.std.gradX2((x-m)/s)

    def gradS(self, x):
        m, s = self.getMS()
        xm = (x-m)/s
        return self.std.gradX(xm) * -xm/s + 1/s

    def gradS2(self, x):
        m, s = self.getMS()
        xm = (x-m)/s
        return (self.std.gradX2(xm) * (xm/s)**2 + self.std.gradX(xm) * 2*xm/s**2) - 1/s**2

    def gradX(self, x):
        m, s = self.getMS()
        return 1/s * self.std.gradX((x-m)/s) # = - gradM

    def gradX2(self, x):
        m, s = self.getMS()
        return 1/s**2 * self.std.gradX2((x-m)/s) # = gradM2

    # to use this in a pipeline, have to move sum outwards
    def gradient(self, x):
        gradM = np.sum(self.gradM(x)) if self.optM else 0
        gradS = np.sum(self.gradS(x)) if self.optS else 0
        return LSF(self.std.gradient(x), gradM, gradS, self.optM, self.optS)

    def laplacian(self, x):
        gradM2 = np.sum(self.gradM2(x)) if self.optM else 0
        gradS2 = np.sum(self.gradS2(x)) if self.optS else 0
        return LSF(self.std.laplacian(x), gradM2, gradS2, self.optM, self.optS)

    def scaledGradient(self, x, d = 1e-12):
        gradM = np.sum(self.gradM(x)) / np.sum(self.gradM2(x)) if self.optM else 0
        gradS = np.sum(self.gradS(x)) / (abs(np.sum(self.gradS2(x))) + d) if self.optS else 0
        # in log domain, have to scale gradS by exp(log(s)) = s
        return LSF(self.std.scaledGradient(x), gradM, gradS, self.optM, self.optS)

    ########## Then there are specific gradients of a given standard distribution
    # In C++ could make this template function where the derivatives of the distribution specific parameters are passed
    # def stdGrad(self, x):
    #     m, s = self.getMS()
    #     return self.std.grad((x-m)/s)
    #
    # def stdGrad2(self, x):
    #     m, s = self.getMS()
    #     return self.std.grad2((x-m)/s)

    ##################### Negative Log Likelihood #############################
    def negLogLike(self, x):
        return np.sum(self.negLogDen(x))

    # def scaledGradient(self, x, optA, optM, optS):
    #     ga = np.sum(self.std.gradA(x)) / np.sum(self.std.gradA2(x)) if optA else 0
    #     gm = np.sum(self.gradM(x)) / np.sum(self.gradM2(x)) if optM else 0
    #     gs = np.sum(self.gradS(x)) / np.sum(self.gradS2(x)) if optS else 0
    #     return EMG(ga, gm, gs)

    ######################## Parameter Estimation ##################################
    # jointly optimize parameters given data x
    def optimize(self, x, maxIter = 32, plot = False):
        params = OPT.defineOptimizationParameters(maxIter = maxIter, minDecrease = 1e-5)
        obj = lambda E : E.negLogLike(x)
        grad = lambda E : E.scaledGradient(x)
        updateVariables = lambda E, dE, s : E - (dE * s)
        projection = lambda E : E.makeValid()
        E, normArr, stepArr = OPT.gradientDescent(self, obj, grad, projection, updateVariables, params)
        self.assign(E)
        if plot:
            import matplotlib.pyplot as plt
            plt.subplot(121)
            plt.plot(normArr)
            plt.subplot(122)
            plt.plot(stepArr)
            plt.show()
        return self

    ########################## Density #######################################

    def density(self, x):
        return np.exp(-self.negLogDen(x))

    # compute gradient of density given gradients of negative log-density
    # Should also do this with templates
    def denGrad(self, den, nllGrad):
        return den * -nllGrad

    def denGrad2(self, den, nllGrad, nllGrad2):
        return den * (nllGrad**2 - nllGrad2)

    def denGradX(self, x):
        return self.denGrad(self.density(x), self.gradX(x))

    def denGradX2(self, x):
        return self.denGrad2(self.density(x), self.gradX(x), self.gradX2(x))

    def denGradM(self, x):
        return self.denGrad(self.density(x), self.gradM(x))
        # return self.density(x) * -self.gradM(x)

    def denGradM2(self, x):
        return self.denGrad2(self.density(x), self.gradM(x), self.gradM2(x))

    def denGradS(self, x):
        return self.denGrad(self.density(x), self.gradS(x))

    def denGradS2(self, x):
        return self.denGrad2(self.density(x), self.gradS(x), self.gradS2(x))

        #
        # if np.any(self.std.a <= 0):
        #     raise ValueError('Negative alpha')
        # if np.any(self.s <= 0):
        #     raise ValueError('Negative sigma')
