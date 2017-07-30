"""
Check what the answers should be for TDA and the graph laplacian
using synthetic pulse trains
"""
import sys
sys.path.append("..")
from SyntheticFunctions import *
from TDA import *
from SlidingWindow import *
from Laplacian import *
from CSMSSMTools import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io as sio
import fractions


def lcm2(a,b): return abs(a * b) / fractions.gcd(a,b) if a and b else 0

def lcm(xs):
    ret = lcm2(xs[0], xs[1])
    for i in range(2, len(xs)):
        ret = lcm2(ret, xs[i])
    return ret

def getFacStr(facs):
    ret = "%i"%facs[0]
    for i in range(1, len(facs)):
        ret = "%s_%i"%(ret, facs[i])
    return ret

if __name__ == "__main__":
    fields = [2, 3, 5, 7, 11]
    np.random.seed(100)
    noiseSigma = 0
    gaussSigma = 4
    NPerPeriod = 400 #Target number of samples per period
    
    for facs in [(1, 12)]:#[(1, 2), (3, 4), (1, 6), (1, 3, 9)]:#, (1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]:
        n0 = lcm(facs)
        f = np.ceil(float(NPerPeriod)/n0)
        T = int(n0*f)
        Ts = [int(T/k) for k in facs]
        N = T*3
        
        print "facs = %s, Ts = %s"%(facs, Ts)
        
        xP = getPerfectPulseTrain(N, Ts)
        xR = getGaussianPulseTrain(N, Ts, noiseSigma, gaussSigma)
        Win = T*2

        YP = getSlidingWindowNoInterp(xP, Win)
        YR = getSlidingWindowNoInterp(xR, Win)
        
        
        #Plot perfect pulse thresholds
        plt.figure(figsize=(12, 5))
        plt.subplot(121)
        plt.plot(np.arange(len(xP)), xP, 'b')
        plt.hold(True)
        plt.plot(np.arange(Win), xP[0:Win], 'r')
        plt.subplot(122)
        D = getSSM(YP)
        plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
        plt.title("%s"%np.unique(D))
        plt.savefig("SSM%s.png"%getFacStr(facs), bbox_inches='tight')
        
        
        plt.figure(figsize=((len(fields)+2)*4, 4))
        plt.subplot(1, len(fields)+1, 1)
        plt.plot(np.arange(len(xR)), xR, 'b')
        plt.hold(True)
        plt.plot(np.arange(Win), xR[0:Win], 'r')
        
        AllPDs = []
        
        for i in range(len(fields)):
            f = fields[i]
            print "Doing field %i"%f
            plt.subplot(1, len(fields)+1, 2+i)
            PDs = doRipsFiltration(YR, 1, coeff=f, relpath="../")
            AllPDs.append(PDs[1])
            plotDGM(PDs[1], color = 'b')
            plt.title("$\mathbb{Z}%i$"%f)
        
        plt.savefig("%s.svg"%getFacStr(facs), bbox_inches = 'tight')
