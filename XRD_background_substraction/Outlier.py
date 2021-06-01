import numpy as np

# First type of objective function: iteratively detects outliers and removes
# them from the matrix factorization objective
class Outlier:

    def __init__(self, IsIn = 1, sigma = 1, sigmaCoeff = 1.5, Qmask = None):
        self.IsIn = IsIn # boolean matrix signaling whether a point is an inlier
        self.sigma = sigma # standard deviation of the fitting error
        self.Qmask = Qmask # can be used to force a point to belong to the background (more effective in the future using GUI)
        self.sigmaCoeff = sigmaCoeff

    def negLogDen(self, x):
        return np.sum(self.IsIn * x**2) / 2

    def gradX(self, x):
        return self.IsIn * x

    def gradX2(self, x):
        return self.IsIn

    # updates sigma based on residual values R and current outlier information IsIn
    def updateSigma(self, R):
        # sqrtN = np.sqrt(np.size(R))
        sqrtN = np.sqrt(np.sum(self.IsIn == 1))
        # self.sigma = np.sqrt(np.sum((self.IsIn*R - np.mean(self.IsIn*R))**2)) / sqrtN
        self.sigma = np.sqrt(np.sum( (self.IsIn*R - np.mean(self.IsIn*R))**2 )) / sqrtN
        return self.sigma

    # updates inlier information
    def updateIsIn(self, R):
        K = 3 # number of points to combine for probability calculation
        D = int((K-1)/2) # number of points to include on each side of current point
        IsPeak = np.ones(R.shape)
        for i in range(R.shape[0]):
            for j in range(D, R.shape[1]-D):
                for k in range(j-D, j+D):
                    if R[i,k] < self.sigmaCoeff * self.sigma:
                        IsPeak[i,j] = 0
        self.IsIn = (1 - IsPeak)
        if self.Qmask is not None:
            self.IsIn[:, self.Qmask] = 1
        return self.IsIn

    def optimize(self, R, maxIter = 0):
        self.updateSigma(R)
        self.updateIsIn(R)
        return self

    def norm(self):
        return np.sqrt(np.linalg.norm(self.IsIn)**2 + self.sigma**2)
