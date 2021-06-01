import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

def defineOptimizationParameters(minDecrease = 1e-8, maxIter = 1024, minIter = 0,
                maxDampingIter = 32, dampingFactor = 7, increaseFactor = 2,
                initialStepSize = 1):
    params = { 'dampingFactor' : dampingFactor, 'increaseFactor': increaseFactor,
                'minDecrease' : minDecrease, 'initialStepSize' : initialStepSize,
                'maxIter': maxIter, 'minIter': minIter, 'maxDampingIter' : maxDampingIter}
    return params


# TODO: implement integral control of stepsize
# implement nonlinear conjugate gradient method
# stop optimization based on average decrease, not latest function decrease
# this guards against accidental small gradients at non-stationary points
# for gradient descent to work on objects, have to define subtraction and scalar multiplication
def gradientDescent(X, objective, gradient,
                        projection = (lambda x : x),
                        updateVariables = (lambda x, dx, s: x - s * dx),
                        params = defineOptimizationParameters(),
                        isInDomain = lambda x : True):

    X = projection(X)
    obj = objective(X)
    stepSize = params['initialStepSize']
    converged = False
    iter = 0
    objArr = np.zeros(params['maxIter']+1)
    stepArr = np.zeros(params['maxIter']+1)
    XTemp = deepcopy(X)
    while not converged:
        oldObj = obj

        deltaX = gradient(X)
        XTemp = updateVariables(X, deltaX, stepSize)
        XTemp = projection(XTemp)
        obj = objective(XTemp)

        dampingIter = 0
        while (obj > oldObj and dampingIter < params['maxDampingIter']) or not isInDomain(X):

            stepSize /= params['dampingFactor']

            XTemp = updateVariables(X, deltaX, stepSize)
            XTemp = projection(XTemp)
            obj = objective(XTemp)
            dampingIter += 1

        # print(dampingIter)
        # print(obj - oldObj)
        # print(oldObj - params['minDecrease'] < obj)
        stepSize *= params['increaseFactor'] # instead of increasing, could reset stepsize to 1 every few iterations (asymptotically 1 is the best value)
        X = deepcopy(XTemp)
        objArr[iter] = obj
        stepArr[iter] = stepSize
        iter += 1
        # print(iter)
        # print(obj)
        if (iter > params['maxIter'] or oldObj - params['minDecrease'] < obj) and iter > params['minIter']:
            converged = True
    return X, objArr[0:iter], stepArr[0:iter]

# def GaussNewton(X, objective, gradient,
#                         projection = (lambda x : x),
#                         updateVariables = (lambda x, dx, s: x - s * dx),
#                         params = defineOptimizationParameters(),
#                         isInDomain = lambda x : True):
#
#     X = projection(X)
#     obj = objective(X)
#     stepSize = params['initialStepSize']
#     converged = False
#     iter = 0
#     objArr = np.zeros(params['maxIter']+1)
#     stepArr = np.zeros(params['maxIter']+1)
#     XTemp = deepcopy(X)
#     while not converged:
#         oldObj = obj
#
#         deltaX = gradient(X)
#         XTemp = updateVariables(X, deltaX, stepSize)
#         XTemp = projection(XTemp)
#         obj = objective(XTemp)
#
#         dampingIter = 0
#         while (obj > oldObj and dampingIter < params['maxDampingIter']) or not isInDomain(X):
#
#             stepSize /= params['dampingFactor']
#
#             XTemp = updateVariables(X, deltaX, stepSize)
#             XTemp = projection(XTemp)
#             obj = objective(XTemp)
#             dampingIter += 1
#
#         # print(dampingIter)
#         # print(obj - oldObj)
#         # print(oldObj - params['minDecrease'] < obj)
#         stepSize *= params['increaseFactor']
#         X = deepcopy(XTemp)
#         objArr[iter] = obj
#         stepArr[iter] = stepSize
#         iter += 1
#         # print(iter)
#         # print(obj)
#         if (iter > params['maxIter'] or oldObj - params['minDecrease'] < obj) and iter > params['minIter']:
#             converged = True
#
#     return X, objArr[0:iter], stepArr[0:iter]

def gradientDescentFixedStep(X, objective, gradient,
                        projection = (lambda x : x),
                        updateVariables = (lambda x, dx, s: x - s * dx),
                        params = defineOptimizationParameters(),
                        isInDomain = lambda x : True):

    X = projection(X)
    obj = objective(X)
    stepSize = params['initialStepSize']
    converged = False
    iter = 0
    objArr = np.zeros(params['maxIter']+1)
    stepArr = np.zeros(params['maxIter']+1)
    XTemp = deepcopy(X)
    while not converged:
        oldObj = obj

        deltaX = gradient(X)
        XTemp = updateVariables(X, deltaX, stepSize)
        XTemp = projection(XTemp)
        obj = objective(XTemp)

        X = deepcopy(XTemp)
        stepArr[iter] = stepSize
        objArr[iter] = obj
        iter += 1
        # print(iter)
        # print(obj)
        if (iter > params['maxIter'] or oldObj - params['minDecrease'] < obj) and iter > params['minIter']:
            converged = True

    return X, objArr[0:iter], stepArr[0:iter]

# expectation maximization algorithm
def EM(p, expectationStep, maximizationStep, maxIter = 32, minDecrease = 1e-6):

    converged = False
    iter = 0
    while not converged:

        p = expectationStep(p)

        p = maximizationStep(p)

        iter += 1
        # TODO: convergence condition on decrease of negative log likelihood
        if np.linalg.norm(Z - oldZ) < minDecrease or iter > maxIter:
            converged = True

    return T, Z, iter


# coordinate descent algorithm, gradient List provides
def coordinateDescent(X, objective, gradientList,
                        updateVariablesList = (lambda x, dx, s: x - s * dx),
                        params = defineOptimizationParameters(),
                        innerParams = defineOptimizationParameters(),
                        projection = (lambda x : x)):

    numCoordinates = len(gradientList)

    obj = objective(X)
    converged = False
    iter = 0
    objArr = np.zeros(params['maxIter']+1)
    XTemp = X

    while not converged:

        for c in range(numCoordinates):

            oldObj = obj
            gradientC = gradientList[c]
            updateVariablesC = updateVariablesList[c]
            projectionC = projectionList[c]
            X, objArrC, stepArr = gradientDescent(X, objective, gradientC,
                updateVariables = updateVariablesC,
                params = innerParams,
                projection = projectionC)
            obj = objArrC[-1]
            objArr[iter] = obj
            iter += 1

        if iter > params['maxIter'] or oldObj - params['minDecrease'] < obj:
            converged = True

    return X, objArr[0:iter]

# stochastic gradient descent
def SDE(X, objective, gradient,
                        updateVariables = (lambda x, dx, s: x - s * dx),
                        params = defineOptimizationParameters(),
                        projection = (lambda x : x),
                        stepSizeUpdate = lambda n : 1/n**2):

    return 0

# to be used for interior point methods with constraint x < b
def logBarrier(x, b):
    return -np.log(b - x)

def logBarrierGradient(x, b):
    return 1 / (b-x), 1 / (b-x)**2

def totalVariation(x):
    diff = np.diff(x)
    TV = np.sum(np.abs(diff))
    return TV

def totalVariationGradient(x):
    grad = np.zeros(x.shape)
    for i in range(1,len(x)-1):
        grad[i] = np.sign(x[i+1] - x[i]) + np.sign(x[i] - x[i-1])
    return grad

def quadraticVariation(x):
    diff = np.diff(x)
    QV = np.sum(diff**2)
    return QV

def quadraticVariationGradient(x):
    grad = np.zeros(x.shape)
    grad[0] = - 2*(x[1] - x[0])
    for i in range(1,len(x)-1):
        grad[i] = 2*(-x[i+1] + 2*x[i] - x[i-1])# -2*(X[i+1] - X[i]) + 2*(X[i] - X[i-1])
    grad[-1] = 2*(x[-1] - x[-2])
    normGrad = 2
    return grad

# for a description, see "Sparse NMF - half-baked or well done?" by Le Roux, Weninger, and Hershey
# grad is the gradient of an objective function I(w) where w is a vector in R^n
# this function computes the gradient of I(w/norm(w))
def gradientOfNormalizedObjective(w, grad):
    N = len(w)
    # for general normaliztion with any vector norm: ( u is the derivative of the norm w.r.t. w)
    # gradN = 1/w * (np.eye(N) - u * w) * grad(w)
    # in the L2 case we simply have
    normW = np.linalg.norm(w)
    u = w / normW
    gradU = grad(u)
    gradN = 1 / normW * (gradU - u * u.dot(gradU))
    return gradN

# gradientDescent test
# import matplotlib.pyplot as plt
# x = 0
# f = lambda x : (x-np.pi)**2
# df = lambda x : 2*(x-np.pi)
# p = lambda x : x if x < np.exp(1) else np.exp(1)
#
# xf, norms = gradientDescent(x, f, df, projection = p)
# plt.plot(norms)
# plt.show()

# # total variation test
# import matplotlib.pyplot as plt
# N = 32
# yi = np.random.randn(2*N)
# x = np.linspace(-1,1,2*N)
# y0 = np.zeros(x.shape)
# y0[0:N] = -1
# y0[N:2*N] = 1
# y0 += .5*np.random.randn(2*N)
# l = 100
#
# f = lambda x : np.sum((x-y0)**2) + l * totalVariation(x)
# df = lambda x : 2*(x-y0) + l * totalVariationGradient(x)
# # print(quadraticVariation(x))
# print(yi)
# print(quadraticVariationGradient(yi))
# # print(f(x))
# print(df(yi))
# yf, norms, steps = gradientDescent(yi, f, df)
# n = range(0,2*N)
# plt.subplot(2,1,1)
# plt.plot(n, yf, n, y0)
# plt.subplot(2,1,2)
# plt.semilogy(norms)
# plt.show()

# quadratic variation test
# import matplotlib.pyplot as plt
# N = 32
# yi = np.random.randn(N)
# x = np.linspace(-1,1,N)
# y0 = np.sin(2*np.pi*x) + np.random.randn(N)
# l = 2
#
# f = lambda x : np.sum((x-y0)**2) + l * quadraticVariation(x)
# df = lambda x : 2*(x-y0) + l * quadraticVariationGradient(x)
# # print(quadraticVariation(x))
# print(yi)
# print(quadraticVariationGradient(yi))
# # print(f(x))
# print(df(yi))
# yf, norms, steps = gradientDescent(yi, f, df)
# n = range(0,N)
# plt.subplot(2,1,1)
# plt.plot(n, yf, n, y0)
# plt.subplot(2,1,2)
# plt.semilogy(norms)
# plt.show()

# # could write this as matrix vector product of Gaussian Kernel matrix
# def gaussianRKHSNorm(x, y, l = 1):
#     kernel = lambda r : np.exp(-r**2 / (2*l**2))
#     GV = RKHSNorm(x, y, kernel)
#     return GV
#
# def gaussianRKHSNormGradient(x, y, l = 1):
#     kernel = lambda r : np.exp(-r**2 / (2*l**2))
#     GV = RKHSNormGradient(x, y, kernel)
#     return GV
#
# # RKHS norm
# def RKHSNorm(x, y, kernel):
#     GV = 0
#     for i in range(len(x)):
#         for j in range(len(x)):
#             GV += kernel(x[i]-x[j]) * y[i] * y[j]
#     return GV
#
# # RKHS norm
# def RKHSNormGradient(x, y, kernel):
#     grad = np.zeros(x.shape)
#     # for i in range(len(x)):
#     #         grad += 2 * kernel(x[i]-x) * y[i]
#     for i in range(len(x)):
#         grad[i] = 2 * kernel(x[i]-x).dot(y)
#     return grad

# # RKHS test
# import matplotlib.pyplot as plt
# N = 128
# yi = np.random.randn(N)
# x = np.linspace(-1,1,N)
# y0 = np.sin(2*np.pi*x) + np.random.randn(N)
# l = 10
# s = .1
# f = lambda y : np.sum((y-y0)**2) + l * gaussianRKHSNorm(x, y, s)
# df = lambda y : 2*(y-y0) + l * gaussianRKHSNormGradient(x, y, s)
#
# # print(quadraticVariation(x))
# print(gaussianRKHSNorm(x, yi, s))
# print(gaussianRKHSNormGradient(x, yi, s))
#
# kernel = lambda r : np.exp(-r**2 / (2*s**2))
# print(x)
# print(kernel(x-x[4]))
#
# # print(f(x))
# print(df(yi))
# yf, norms, steps = gradientDescent(yi, f, df)
# plt.subplot(2,1,1)
# plt.plot(x, yf, x, y0)
# plt.subplot(2,1,2)
# plt.semilogy(norms)
# plt.show()
