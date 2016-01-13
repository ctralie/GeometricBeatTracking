import numpy as np
import matplotlib.pyplot as plt
from SoundTools import *

def getPulseTrain(NSamples, TMin, TMax, AmpMin, AmpMax):
    x = np.zeros(NSamples)
    x[0] = 1
    i = TMin
    while i < NSamples:
        i += TMin + int(np.round(np.random.randn()*(TMax-TMin)))
        if i >= NSamples:
            break
        x[i] = AmpMin + (AmpMax-AmpMin)*np.random.randn()  
    return x

def getSyntheticPulseTrain(NSamples, T, noiseSigma, gaussSigma):
    x = np.zeros(NSamples)
    x[0::T] = 1
    gaussWidth = int(np.round(gaussSigma*3))
    g = np.exp(-(np.arange(-gaussWidth, gaussWidth+1, dtype=np.float64))**2/(2*gaussSigma**2))
    x = np.convolve(x, g, 'same')
    x = x + noiseSigma*np.random.randn(len(x))
    return x

def getSyntheticPulseTrainFreqAmpDrift(NSamples, TMin, TMax, AmpMin, AmpMax, noiseSigma, gaussSigma):
    x = getPulseTrain(NSamples, TMin, TMax, AmpMin, AmpMax)
    plt.show()
    gaussWidth = int(np.round(gaussSigma*3))
    g = np.exp(-(np.arange(-gaussWidth, gaussWidth+1, dtype=np.float64))**2/(2*gaussSigma**2))
    y = np.convolve(x, g, 'same')
    y = y + noiseSigma*np.random.randn(len(y))
    return (x, y)


if __name__ == "__main__":
    np.random.seed(100)
    T = 300
    NPCs = 20
    noiseSigma = 0.1
    gaussSigma = 3
    x = getSyntheticPulseTrain(10000, T, noiseSigma, gaussSigma)
    s = BeatingSound()
    s.novFn = x

    W = 200
    (Y, S) = s.getSlidingWindowLeftSVD(W)
    V = s.getSlidingWindowRightSVD(W, NPCs)
    
    plt.subplot(221)
    plt.imshow(Y[:, 0:40], interpolation = 'none', aspect = 'auto')
    plt.subplot(222)
    plt.plot(V[0, :], V[1, :])
    
    y = s.performDenoising(np.arange(2))
    plt.subplot(223)
    plt.plot(x)
    plt.subplot(224)
    plt.plot(y)
    
    plt.show()
