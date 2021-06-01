import numpy as np

# Second type of distribution:
# Mixture of uniform outlier distribution with Gaussian noise distribution (UMGM)
# can be seen as a smoothed version of the outlier detection (amenable to EM)
class UGM:

    def __init__(self, z, s, max, mix = 1/2, optMix = False, optS = False, hard = True):
        self.z = z
        self.mix = mix # mixture probability
        self.s = s
        self.max = max # uniform component has support over [0, max]

        self.optS = optS
        self.optMix = optMix
        self.hard = hard # instead of soft assignment weighted by membership probability, uses maximum likelihood assignment

    def negLogDen(self, x):
        return self.z * ((x/self.s)**2 / 2) + (1-self.z) * np.log(self.max) # from Gaussian component: (np.sqrt(2*np.pi)*self.s)

    def negLogLike(self, x):
        return np.sum(self.negLogDen(x))

    def gradX(self, x):
        return self.z * x / self.s**2

    def gradX2(self, x):
        return self.z / self.s**2

    # updates sigma based on residual values x and current outlier information z
    def maximizationStep(self, x):
        if self.optS:
            self.s = np.sqrt(np.sum( (self.z*x - np.mean(self.z*x))**2 ) / np.sum(self.z))

        if self.optMix:
            mix = np.mean(z)

        return self.s

    def expectationStep(self, x):
        # mix is mixture probability
        GauDen = np.exp(-(x/self.s)**2/2) / (np.sqrt(2*np.pi)*self.s)
        UniDen = np.full(x.shape, 1/self.max) * (x > 0)

        sum = (self.mix * GauDen + (1-self.mix) * UniDen)

        ind = sum == 0
        notInd = np.logical_not(ind)
        # if np.sum(ind) > 0:
        #     print(np.sum(ind))
        z = np.zeros(x.shape)
        z[ notInd ] = self.mix * GauDen[notInd] / sum[notInd]

        # print(np.sum(np.logical_and(ind, x <= 0)))
        z[ np.logical_and(ind, x <= 0) ] = 1
        z[ np.logical_and(ind, x > 0) ] = 0

        self.z = z

        # choose maximum likelihood assignment
        if self.hard:
            self.z = self.z > 1/2

        # chance to do markov chain here!
        return self.z

    def optimize(self, x, maxIter = 1):
        for i in range(maxIter):
            self.expectationStep(x)
            self.maximizationStep(x)
        return self

    def norm(self):
        return np.sqrt(np.linalg.norm(self.z)**2 + self.s**2)
