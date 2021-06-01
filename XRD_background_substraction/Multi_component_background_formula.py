import math
import numpy as np
import random
import matplotlib.pyplot as plt
def EMG(mu, sigma, lamda, x):
    error_function = math.erf((mu+ (lamda * (sigma **2)) - x)/(np.sqrt(2) * sigma))
    emg = (lamda/2) * np.exp((lamda/2) * (2 * mu + (lamda * (sigma **2)) - 2 * x)) * error_function
    return emg

x = np.linspace(0, 10, 1000)
output = [EMG(1, 1, 1, x_) for x_ in x]


plt.plot(x, output)
plt.show()
#likelihood of the atual data

plt.hist(output)
plt.show()