# Implementation of Reproducing Kernel Hilbert spaces (RKHS)
import numpy as np
import numpy.linalg as la
from numba import jit, prange
# import POC

def RBF(s = 1):
    return lambda x, y: np.exp( - np.abs(x-y)**2 / (2*s**2) )

@jit(nopython = True, fastmath = True)
def getRBFKernelMatrix(x, y, s = 1):
    K = np.zeros((len(x), len(y)))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            K[i,j] = np.exp( - np.abs(xi-yj)**2 / (2*s**2) )
    return K

def getKernelMatrix(x, y, kernel = RBF):
    K = np.zeros((len(x), len(y)))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            K[i,j] = kernel(xi, yj)
    return K

def getRBFKernelMatrix(x, y, l = 1):
    K = np.zeros((len(x), len(y)))
    dif = x.reshape(-1,1).dot(np.ones((1, len(y)))) - np.ones((len(x), 1)).dot(y.reshape(1,-1))
    K = np.exp( - dif**2 / (2*l**2) )
    return K

# inner product in RKHS with kernel K
def inner(x, y, K):
    return np.dot(np.dot(x, K), y)

def norm(x, K):
    return inner(x, x, K)

# U is an orthonormal basis for the RKHS
# projects column vectors
def project(x, U):
    return np.dot(U, np.dot(U.T, x))

# projects row vectors
def projectRows(x, U):
    return (x.dot(U)).dot(U.T)

def orthogonalBasis(K, thres = 1e-6):
    V, e = truncatedEig(K, thres = thres)
    return V

# low rank approximation of a positive definite Kernel matrix up to a given threshold
def truncatedEig(K, thres = 1e-6):
    e, v = la.eigh(K)
    r = 0
    # import matplotlib.pyplot as plt
    # plt.semilogy(e)
    # plt.show()
    while e[r] < thres:
        r += 1
    e = e[r:]
    V = v[:,r:]
    return V, e

################################# demonstration ###############################
# n = 128
# x = np.linspace(-1,1,n)
# v = np.sin(2*np.pi*x)
# vn = v.reshape(-1,1) + .2 * np.random.randn(n, 2)
#
# s = .5
# K = getKernelMatrix(x, x, RBF(s))
# P = orthogonalBasis(K)
#
# import matplotlib.pyplot as plt
# plt.plot(x, P[:,0], x, K[:,0])
# plt.show()
#
# print(vn.T.shape)
# print(project(vn.T,P).shape)
# print(x.shape)
# plt.plot(x, vn, x, project(vn.T, P).T, x, v)
# plt.show()

# # calculates a basis for the background signals based on a Kernel
# def getBackgroundBasis(Q, s = 2, thres = 1e-6):
#     kernel = RKHS.RBF(s)
#     K = RKHS.getKernelMatrix(Q, Q, kernel)
#     V = RKHS.orthogonalBasis(K, thres = thres)
#     return V
