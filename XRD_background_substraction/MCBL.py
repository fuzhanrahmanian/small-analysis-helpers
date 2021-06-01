import numpy as np
import numpy.linalg as la

# utility modules
from MatrixFactorization import PMF
import RKHS as RKHS # Reproducing Kernel Hilbert Space

# residual distributions to choose from
from Gauss import Gauss # Gaussian Distribution
from EMGM import EMGM   # Exponentially-modified Gaussian Mixture
from RegressionQuantile import RegQuant # Regression Quantile
from Outlier import Outlier # Outlier detection
from UGM import UGM # Uniform-Gauss Mixture

# Matrix Factorization Class compatible with any residual distribution given by resDis
class MCBL( PMF ):

    def __init__(self, U = 0, V = 0, optU = True, optV = True,
            resDis = Gauss(0,1), optResDis = True, B = None, phaseMask = None):

        # initializing PMF fields
        self.resDis = resDis # residual distribution (EMGM, Gauss, Outlier)
        self.U = U  # factor matrices
        self.V = V

        # boolean optimization flags
        self.optResDis = optResDis
        self.optU = optU
        self.optV = optV

        self.B = B # holds factorized RKHS basis
        self.phaseMask = phaseMask

    # basic arithmetic
    def __add__(self, other):
        return MCBL(self.U + other.U, self.V + other.V, self.optU, self.optV, self.resDis, self.optResDis, self.B, self.phaseMask)

    def __sub__(self, other):
        return MCBL(self.U - other.U, self.V - other.V, self.optU, self.optV, self.resDis, self.optResDis, self.B, self.phaseMask)

    def __mul__(self, scalar):
        return MCBL(self.U * scalar, self.V * scalar, self.optU, self.optV, self.resDis, self.optResDis, self.B, self.phaseMask)

    def __truediv__(self, other):
        return MCBL(self.U / other.U, self.V / other.V, self.optU, self.optV, self.resDis, self.optResDis, self.B, self.phaseMask)

    # initializes the basis of the background RKHS
    def initBasis(self, q, l = 1):
        # kernel = RKHS.RBF(l)    # could be replaced with arbitrary kernel
        # K = RKHS.getKernelMatrix(q, q, kernel)
        K = RKHS.getRBFKernelMatrix(q, q, l)
        self.B = RKHS.orthogonalBasis(K)

    # projection of V into RKHS
    def project(self):
        # self.U = np.maximum(self.U, 0) # can add non-negativity as a constraint
        self.V = RKHS.projectRows(self.V, self.B)
        return self

    def initFactors(self, x, rank):
        numSamples, numQ = x.shape
        # randomly
        self.U = np.random.rand(numSamples, rank)
        self.V = np.dot(np.random.rand(rank, self.B.shape[1]), self.B.T)
        self.V = RKHS.projectRows(self.V, self.B)

    def inferBackground(self, q, x, rank = 8, l = 1, maxIter = 16, minChange = 1e-6): #  mix = .5, optMix = True,
        # initialize background basis
        self.initBasis(q, l)
        # initialize factor matrices
        self.initFactors(x, rank)
        # solve PMF problem
        self.optimize(x, maxIter = maxIter, maxFactorizeIter = 512, maxResDisIter = 32, minChange = 1e-6)
        return self

    def scaledGradient(self, x):
        G = super(MCBL, self).scaledGradient(x)
        return MCBL(G.U, G.V, G.optU, G.optV, G.resDis, G.optResDis, self.B, self.phaseMask)

# ideally normalize data before calling
def MCBL_UGM(Q, I, rank, l, maxIter = 32):

    optS = True
    optMix = False
    z = np.ones(I.shape) # initially, all points belong to the Gaussian component
    s = 1 # initial standard deviation
    max = np.max(I)
    mix = .1 # mixture probability (probability of observing background signal)
    hard = False # if true: latent variables z are assigned maximum likelihood estimate
    resDis = UGM(z, s, max, mix = mix, optMix = optMix, optS = optS, hard = hard)
    print("Creating MCBL Object")
    P = MCBL(resDis = resDis)
    print("Initializing Background Basis")
    P.initBasis(Q, l)
    print("Initializing Background Factors")
    P.initFactors(I, rank)
    sampleID = 0
    print("Optimizing ...")
    for i in range(maxIter):
        P.optimize(I, maxIter = 0, maxFactorizeIter = 512, maxResDisIter = 16, minChange = 1e-6)
        print( str(i/maxIter) + "%")
    print("100%")
    return P

def MCBL_Out(Q, I, rank, l, maxIter = 32):
    resDis = Outlier(np.ones(I.shape))
    print("Creating MCBL Object")
    P = MCBL(resDis = resDis)
    print("Initializing Background Basis")
    P.initBasis(Q, l)
    print("Initializing Background Factors")
    P.initFactors(I, rank)
    print("Optimizing ...")
    P.optimize(I, maxIter = maxIter, maxFactorizeIter = 512, maxResDisIter = maxIter, minChange = 1e-6)
    return P

# using Regression Quantiles, and EMGM proceeds in a similar manner.
