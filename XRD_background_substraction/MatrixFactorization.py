import numpy as np
import numpy.linalg as la
import OPT as OPT
from Gauss import Gauss
from copy import deepcopy

###################### Probabilistic Matrix Factorization ######################
class PMF():

    def __init__(self, U, V, optU = True, optV = True, resDis = Gauss(), optResDis = False):
        # residual distribution requirements: negLogDen, gradX, gradX2, assign, addition etc.
        self.resDis = resDis

        self.U = U
        self.V = V

        self.optResDis = optResDis
        self.optU = optU
        self.optV = optV

    def print(self):
        self.resDis.print()
        print('optResDis = ' + str(self.optResDis))
        print('optU = ' + str(self.optU))
        print('optV = ' + str(self.optV))

    def assign(self, other):
        self.U = other.U
        self.V = other.V

    def __add__(self, other):
        return PMF(self.U + other.U, self.V + other.V, self.optU, self.optV, self.resDis, self.optResDis)

    def __sub__(self, other):
        return PMF(self.U - other.U, self.V - other.V, self.optU, self.optV, self.resDis, self.optResDis)

    def __mul__(self, scalar):
        return PMF(self.U * scalar, self.V * scalar, self.optU, self.optV, self.resDis, self.optResDis)

    def __truediv__(self, other):
        return PMF(self.U / other.U, self.V / other.V, self.optU, self.optV, self.resDis, self.optResDis)

    def norm(self):
        return np.sqrt(self.resDis.norm()**2 + la.norm(self.U)**2 + la.norm(self.V)**2)

    def UDotV(self):
        return self.U.dot(self.V)

    def getUV(self):
        return self.U, self.V

    def getRes(self, x):
        U, V = self.getUV()
        return x - U.dot(V)

    # initialize U, V randomly
    def initRandMat(self, n, m, rank):
        U, V = self.getUV()
        self.U = np.random.rand((n, rank))
        self.V = np.random.rand((rank, m))

    def initFromData(self, x, rank):
        n, m = x.shape
        self.U = np.random.rand((n, rank))
        self.V = np.random.rand((rank, m))

    def negLogLike(self, x):
        nll = np.sum( self.resDis.negLogDen( self.getRes(x) ) )
        return nll

    def scaledGradient(self, x, d = 1e-12):

        U, V = self.getUV()
        r = self.getRes(x)

        grad = self.resDis.gradX( r )
        grad2 = self.resDis.gradX2( r )

        gradU = - grad.dot(V.T) / (grad2.dot(V.T**2) + d) if self.optU else 0
        gradV = - (U.T).dot(grad) / ((U.T**2).dot(grad2) + d) if self.optV else 0

        return PMF(gradU, gradV, self.optU, self.optV)

    def project(self):
        return self

    def factorize(self, x, maxIter = 1024):
        params = OPT.defineOptimizationParameters(maxIter = maxIter)

        objective = lambda M : M.negLogLike(x)
        gradient = lambda M : M.scaledGradient(x)
        projection = lambda M : M.project()
        updateVariables = lambda E, dE, s : E - (dE * s)
        M, objArr, stepArr = OPT.gradientDescent(self, objective, gradient, projection = projection, updateVariables = updateVariables, params = params)
        # import matplotlib.pyplot as plt
        # plt.plot(objArr)
        # plt.show()
        # plt.semilogy(stepArr)
        # plt.show()
        self.assign(M)
        return self

    # optimize both factor matrices and residual distribution alternatingly
    # maxIter is the number of outer iterations
    # maxFactorizeIter is the number of iterations per factorization optimization
    # maxResDisIter is the number of iterations for the residual distribution optimization
    def optimize(self, x, maxIter = 16, minChange = 1e-6, maxFactorizeIter = 1024, maxResDisIter = 128):
        iter = 0
        converged = False
        while not converged:
            oldSelf = deepcopy(self)

            if self.optU or self.optV:
                # print('factorizing')
                self.factorize(x, maxIter = maxFactorizeIter)

            if self.optResDis:
                # print('optimizing residual model')
                self.resDis.optimize(self.getRes(x), maxIter = maxResDisIter)
            iter += 1

            if iter > maxIter or (oldSelf-self).norm() < minChange:
                converged = True

        return self

# def alternatingLS(X, U, V, maxIter = 16):
#     for i in range(maxIter):
#         V = np.linalg.solve(U, X)
#         U = np.linalg.solve(V.T, X.T).T
#     return U, V

################################# Demonstration ###############################
# n = 128
# m = n
# k = 1
# U = np.random.randn(n, k)
# V = np.random.randn(k, m)
# X = U.dot(V) + 1 * np.random.randn(n, m)
#
# ki = 1
# Ui = np.random.randn(n, ki)
# Vi = np.random.randn(ki, m)
#
# M = PMF(Ui, Vi)
# M.factorize(X)
#
# M.resDis.setOptS(True)
# M.resDis.optimize(M.getRes(X))
# M.resDis.print()
# print(la.norm(M.getRes(X)) / np.size(X))
