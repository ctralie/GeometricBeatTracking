import sys
sys.path.append("..")
from SyntheticFunctions import *
from SlidingWindow import *
from Laplacian import *
from CSMSSMTools import *
import matplotlib.pyplot as plt
from PulseExperiments import *
from CircularCoordinates import *
import glob
import os
from sys import exit

PREFIX = "temp"


if __name__ == '__main__':
    colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y']
    doAnimation = False
    NPerPeriod = 200
    amps = (1, 0.5, 1)
    
    plt.figure(figsize=(6, 2))
    AllFacs = [(1, 2), (1, 3), (1, 5), (1, 3, 9)]
    Strs = ["2 on 1", "3 on 1", "5 on 1", "9 on 3 on 1"]
    for i in range(len(AllFacs)):
        plt.subplot(1, 4, i+1)
        facs = AllFacs[i]
        n0 = lcm(facs)
        f = np.ceil(float(NPerPeriod)/n0)
        T = int(n0*f)
        Ts = [int(T/k) for k in facs]
        N = int(T*3.01)

        print "facs = %s, Ts = %s"%(facs, Ts)

        xR = getGaussianPulseTrain(N, Ts, amps, 0, 3)
        Win = T*2

        YR = getSlidingWindowNoInterp(xR, Win)
        D = getSSM(YR)
        plt.imshow(D, interpolation = 'nearest', cmap = 'afmhot')
        plt.title(Strs[i])
        plt.axis('off')
    plt.savefig("SSMs.svg", bbox_inches = 'tight')
