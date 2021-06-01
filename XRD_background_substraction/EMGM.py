import numpy as np
import numpy.linalg as la

# import self-made tools
import OPT as OPT
import Gauss as Gauss
import EMG as EMG

from LocationScaleFamily import LSF
from copy import deepcopy

############################### Standard EMG Mixture ###########################
class STD():

    def __init__(self, a = 1, z = 0, optA = False, optZ = False):
        self.EMG = EMG.STD(a, optA)
        self.Gauss = Gauss.STD()
        self.z = z
        self.optZ = optZ

    def print(self):
        self.EMG.print()
        print('z = ' + str(self.z))
        print('optZ = ' + str(self.optZ))

    def getZ(self):
        return self.z

    def setZ(self, z):
        self.z = z

    def getAZ(self):
        return self.a, self.z

    def setAZ(self, a, z):
        self.EMG.setA(a)
        self.z = z

    def setOpt(self, optA, optZ):
        self.EMG.optA = optA
        self.optZ = optZ

    # checks if the distribution is valid
    def isValid(self):
        if self.EMG.isValid() and self.Gauss.isValid():
            return True
        else:
            return False

    def makeValid(self, thres = 1e-6):
        self.EMG.makeValid()
        self.Gauss.makeValid()
        return self

    # assign the variables of another LSFamily object to the current one
    def assign(self, other):
        self.EMG.assign(other.EMG)
        self.z = other.z

    # define operators on standard EMGM objects
    def __add__(self, other):
        return STD(self.EMG.getA() + other.EMG.getA(), self.z + other.z, self.EMG.getOptA(), self.optZ)

    def __sub__(self, other):
        return STD(self.EMG.getA() - other.EMG.getA(), self.z - other.z, self.EMG.getOptA(), self.optZ)

    # ideally would have scalar and elementwise multiplication
    def __mul__(self, scalar):
        return STD(self.EMG.getA() * scalar, self.z * scalar, self.EMG.getOptA(), self.optZ)

    def __truediv__(self, other):
        return STD(self.EMG.getA() / other.EMG.getA(), self.z / other.z, self.EMG.getOptA(), self.optZ)

    # calculate the norm of the object
    def norm(self):
        return np.sqrt(self.EMG.norm()**2 + la.norm(self.z)**2)

    def negLogDen(self, x):
        z = self.getZ()
        return (1-z) * self.Gauss.negLogDen(x) + z * self.EMG.negLogDen(x)

    def gradX(self, x):
        z = self.getZ()
        return (1-z) * self.Gauss.gradX(x) + z * self.EMG.gradX(x)

    def gradX2(self, x):
        z = self.getZ()
        return (1-z) * self.Gauss.gradX2(x) + z * self.EMG.gradX2(x)

    def gradA(self, x):
        z = self.getZ()
        return z * self.EMG.gradA(x)

    def gradA2(self, x):
        z = self.getZ()
        return z * self.EMG.gradA2(x)

    def gradient(self, x):
        return STD(np.sum(self.gradA(x)) if self.EMG.getOptA() else 0, 0)

    def laplacian(self, x):
        return STD(np.sum(self.gradA2(x)) if self.EMG.getOptA() else 0, 0)

    def scaledGradient(self, x, d = 1e-12):
        return STD(np.sum(self.gradA(x)) / (abs(np.sum(self.gradA2(x)) + d)) if self.EMG.getOptA() else 0, 0)

    # generate samples that follow a standard EMG mixture distribution
    def genSamples(self, size = 1):
        z = self.getZ()
        ind = np.random.random(size) < z
        return (1-ind) * self.Gauss.genSamples(size) + ind * self.EMG.genSamples(size)

    # calculate the expected value of z, given mixture probabilities mix
    def expectedZ(self, x, mix):

        # compute responsibilities
        GauDen = self.Gauss.density(x)
        EMGDen = self.EMG.density(x)

        # return (mix * EMGDen) / ((1-mix) * GauDen + mix * EMGDen)
        # for numerical stability
        ind = (mix*EMGDen + (1-mix)*GauDen) == 0
        notInd = np.logical_not(ind)
        z = np.zeros(x.shape)
        z[ notInd ] = (mix * EMGDen[notInd]) / (mix * EMGDen[notInd] + (1-mix) * GauDen[notInd])
        z[ np.logical_and(ind, x>0) ] = 1
        z[ np.logical_and(ind, x<0) ] = 0
        return z

########################## EMG Mixture model ##################################
class EMGM( LSF ):
# could be a child of an abstract mixture model class

    def __init__(self, a = 1, m = 0, s = 1, z = 0, optA = False, optM = False, optS = False, optZ = False):

        self.std = STD(a, z, optA, optZ)
        self.m = m
        self.s = s

        self.optM = optM
        self.optS = optS

    def getAMSZ(self):
        a, z = self.std.getAZ()
        return a, self.getMS(), z

     # set variables, makes sure that both distributions are updated
    def setAMSZ(self, a, m, s, z):
        self.std.setAZ(a, z)
        self.setMS(m, s)

    def getZ(self):
        return self.std.getZ()

    def getOptZ(self):
        return self.std.optZ

    def setZ(self, z):
        return self.std.setZ(z)

    # setting all optimization indicators
    def setOpt(self, optA, optM, optS, optZ):
        self.std.setOpt(optA, optZ)
        self.setOptM(optM)
        self.setOptS(optS)

    ############################## Optimization ################################
    def calculateMix(self):
        return np.mean(self.getZ())

    def expectedZ(self, x, mix):
        m, s = self.getMS()
        return self.std.expectedZ( (x-m)/s, mix)

    def expectationStep(self, x, mix):
        self.setZ(self.getOptZ() * self.expectedZ(x, mix))
        return self

    def maximizationStep(self, x, mix, optMix, maxIter):

        # optimize continuous parameters of EMGM model
        super(EMGM, self).optimize(x, maxIter = maxIter)

        # optimize mixture coefficient
        if optMix:
            mix = np.mean(self.getZ())

        return mix

    ################ EM Algorithm for optimization of EMG Mixture ##############
    def optimize(self, x, mix = 1/2, optMix = True, maxIter = 8, minChange = 1e-6, maxMaxIter = 128):
        converged = False
        iter = 0
        while not converged:

            oldSelf = deepcopy(self)

            mix = self.maximizationStep(x, mix, optMix, maxMaxIter)

            if np.any(self.getOptZ()):
                self.expectationStep(x, mix)

            iter += 1
            if iter > maxIter or (oldSelf-self).norm() < minChange:
                converged = True
        # print(self.s)
        # print(self.std.EMG.a)
        return self

############################## Demonstration ###################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    print('Creating object')
    mix = .1
    E = EMGM(a = 1, m = 1, s = 1, z = mix)
    E.print()


    print('Generating samples')
    n = 1024
    x = E.genSamples(n)

    import matplotlib.pyplot as plt
    dom = np.linspace(-1, 1, n)
    plt.plot(dom, x)
    plt.show()
    print('Resetting variables')
    E.setAMSZ(.5, 0, 1, mix)

    print('Setting variables to be inferred')
    E.setOpt(True, True, True, True)
    E.print()

    print('Maximum likelihood estimation')
    E.optimize(x)
    print('Inferred mixture probability:' + str(np.mean(E.getZ())))
    print('True mixture probability:' + str(mix))
    E.print()

    ind = E.getZ() > .5
    plt.plot(dom, x)
    plt.plot(dom[ind], x[ind], '*')
    plt.show()

    print('Plotting outlier probability ')
    r = np.linspace(min(x), max(x), n)
    plt.plot(r, E.expectedZ(r, mix))
    plt.show()
