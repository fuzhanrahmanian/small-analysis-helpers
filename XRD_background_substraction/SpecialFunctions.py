# numerically stable python implementation of the log complementary error function
import numpy as np
from scipy.special import erfcx, erfc

def log_erfc( x ):
    fx = np.zeros(x.shape)
    ind = (x > 8)
    fx[ind] = np.log(erfcx(x[ind])) - x[ind]**2
    ind = (x <= 8)
    fx[ind] = np.log(erfc(x[ind]))
    # for i in range(len(x)):
    #     if x[i] > 8.0:
    #         fx[i] = np.log(np.erfcx(x[i])) - x[i]**2
    #     else :
    #         fx[i]  = np.log(np.erfc(x[i]))
    return fx
