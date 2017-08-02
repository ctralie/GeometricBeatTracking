import numpy as np
import matplotlib.pyplot as plt

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

def convolveGaussAndAddNoise(x, gaussSigma, noiseSigma):
    gaussSigma = int(np.round(gaussSigma*3))
    g = np.exp(-(np.arange(-gaussSigma, gaussSigma+1, dtype=np.float64))**2/(2*gaussSigma**2))
    x = np.convolve(x, g, 'same')
    x = x + noiseSigma*np.random.randn(len(x))
    return x

def getPerfectPulseTrain(NSamples, Ts, amps):
    x = np.zeros(NSamples)
    for i in range(len(Ts)):
        T = Ts[i]
        x[0::T] += amps[i]
    return x

def getGaussianPulseTrain(NSamples, Ts, amps, noiseSigma, gaussSigma):
    x = getPerfectPulseTrain(NSamples, Ts, amps)
    x = convolveGaussAndAddNoise(x, gaussSigma, noiseSigma)
    return x

def getRectPulseTrain(NSamples, Ts, amps, rectWidth = 3):
    x = getPerfectPulseTrain(NSamples, Ts, amps)
    g = np.ones(rectWidth)
    x = np.convolve(x, g, 'same')
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
