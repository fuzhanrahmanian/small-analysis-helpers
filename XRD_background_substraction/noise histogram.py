from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap

viridis = cm.get_cmap('viridis', 7)
newcolors = viridis(np.linspace(0, 1, 256))
newcmp = ListedColormap(newcolors)



x = loadmat(r'C:\Users\Fuzhi\Documents\GitHub\XRD_background_substraction\20160728_MnFeCo.mat')
x.keys()
intensity = x['XRDData'][:, 35]


def xrdbg(input,c = 0.5,kmax = 30):
    def passthrough(inp):
        inputc = np.copy(inp)
        m = np.zeros(len(inputc)-1)
        for k in range(1,kmax): #number of iterations
            for l in range(1,len(inputc)-1): #go through all p_l
                m[l-1] = 0.5*inputc[l-1]+0.5*inputc[l+1]
            for l in range(1,len(inputc)-1): #check if larger according to Sonneveld & Visser 1975
               #adaptive curvature
                if inputc[l]>m[l]+c:
                    inputc[l]=m[l]
        return np.copy(inputc)
    return passthrough(input)

background = xrdbg(intensity)
net = intensity - background

xrd = [background, intensity, net]
cmap = plt.get_cmap('viridis', 3)
colors ={c:cmap(v) for c,v in zip(xrd,[0,2,4])}

plt.plot(background, label= 'background', alpha= 0.6) #, color='r'
plt.plot(intensity, label= 'measurement', alpha= 0.5) #, color= 'c'
plt.plot(net, label= 'net signal', alpha= 0.6) #, color= 'y'
plt.xlabel('Wavenuber ($cm^{-1}$)')
plt.ylabel('Intensity')
plt.legend()
plt.savefig('classical_method.svg', format='svg', quality=100)
plt.show()



plt.hist(background, 50, alpha= 0.4, label= 'background', rwidth=5, density=True)
plt.hist(intensity, 50, alpha= 0.5, label= 'measurement', rwidth=5, density=True)
plt.hist(net, 50, alpha= 0.6, label= 'net signal', rwidth=5, density=True)
plt.ylabel('Frequency')
plt.xlabel('Intensity')
plt.xlim([0, 180])
plt.legend()
plt.savefig('noise_histogram.svg', format='svg', quality=100)
plt.show()




