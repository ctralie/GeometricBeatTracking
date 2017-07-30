import numpy as np
import matplotlib.pyplot as plt
from SoundTools import *
from CircularCoordinates import *

def getPulseTrain(NSamples, TMin, TMax, AmpMin, AmpMax):
    x = np.zeros(NSamples)
    x[0] = 1
    i = 0
    while i < NSamples:
        i += TMin + int(np.round(np.random.randn()*(TMax-TMin)))
        if i >= NSamples:
            break
        x[i] = AmpMin + (AmpMax-AmpMin)*np.random.randn()  
    return x

def convolveAndAddNoise(x, gaussSigma, noiseSigma):
    gaussSigma = int(np.round(gaussSigma*3))
    g = np.exp(-(np.arange(-gaussSigma, gaussSigma+1, dtype=np.float64))**2/(2*gaussSigma**2))
    x = np.convolve(x, g, 'same')
    x = x + noiseSigma*np.random.randn(len(x))
    return x

def getSyntheticPulseTrain(NSamples, T, noiseSigma, gaussSigma):
    x = np.zeros(NSamples)
    x[0::T] = 1
    x = convolveAndAddNoise(x, gaussSigma, noiseSigma)
    return x

def getSyntheticPulseTrainFreqAmpDrift(NSamples, TMin, TMax, AmpMin, AmpMax, noiseSigma, gaussSigma):
    x = getPulseTrain(NSamples, TMin, TMax, AmpMin, AmpMax)
    y = convolveAndAddNoise(x, gaussSigma, noiseSigma)
    #return (original pulse train, gaussian convolved noisy version)
    return (x, y)

def getSyntheticPulseTrainRandMicrobeats(NSamples, T, noiseSigma, gaussSigma):
    x = getPulseTrain(NSamples, T, T, 1, 1)
    #Now make some random microbeats that are half the tempo
    i = 0
    while i < NSamples:
        i += T/2
        if i >= NSamples:
            break
        if np.random.rand() < 0.5:
            x[i] = 0.5
    y = convolveAndAddNoise(x, gaussSigma, noiseSigma)
    return (x, y)


def getSyntheticPulseTrainPerfectMicrobeats(NSamples, T, noiseSigma, gaussSigma):
    x = getPulseTrain(NSamples, T, T, 1, 1)
    x += 0.5*getPulseTrain(NSamples, T/2, T/2, 1, 1)
    y = convolveAndAddNoise(x, gaussSigma, noiseSigma)
    return (x, y)

if __name__ == "__main__":
    np.random.seed(100)
    T = 300
    NPCs = 20
    noiseSigma = 0.1
    gaussSigma = 2
    (x, y) = getSyntheticPulseTrainPerfectMicrobeats(10000, T, noiseSigma, gaussSigma)
    s = BeatingSound()
    s.novFn = y
    s.origNovFn = y
    s.hopSize = 128
    s.Fs = 44100
    
    NPCs = 0
    pca = None
    if NPCs > 0:
        pca = PCA(n_components = NPCs)
    gaussWin = 20
    W = 600
    
    for Kappa in [0.01, 0.025, 0.05, 0.1, 0.2]:
        theta = getCircularCoordinatesBlocks(s, W, pca, 600, 100, None, 3, denoise = True, doPlot = True, Kappa = Kappa)
        (onsets, score) = getOnsetsDP(theta, s, 6, 0.4)
        plt.clf()
        plt.subplot(211)
        plt.plot(y)
        plt.subplot(212)
        onsetsSec = onsets*float(s.hopSize)/s.Fs
        vals = theta[onsets]
        plt.stem(onsetsSec, vals)
        plt.show()
