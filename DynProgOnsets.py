import numpy as np
import matplotlib.pyplot as plt

def getOnsetsDP(theta, s, tightness, alpha = 0.8):
    """
    :param theta: Circular coordinate estimates
    :param s: BeatingSound object (has Fs, nvFn, hopSize)
    :param tightness: How much to weight the beat penalty
    :param alpha: How much to weight the novelty function
    :return (onsets, score)
    """
    N = theta.size
    
    #Step 1: Estimate the slope at every point in radians per sample,
    #which is used to figure out search windows for each point
    sWin = int(np.round(s.Fs/s.hopSize)) #Estimate slope in 2 second windows
    slopes = np.zeros(N)
    deriv = np.zeros(sWin*2)
    deriv[0:sWin] = np.ones(sWin)
    deriv[sWin:] = -np.ones(sWin)
    slopes[sWin*2-1:] = np.convolve(theta, deriv, 'valid')/float(sWin**2)
    slopes[0:sWin*2-1] = slopes[sWin*2-1]
    #Use this to figure out search indices for each point
    i1 = np.array(np.round(-20*np.pi/slopes), dtype=np.int64) #4 beats before
    i1 = i1 + np.arange(N)
    i1 = np.maximum(i1, 0)
    i2 = np.array(np.round(-2*np.pi/(20*slopes)), dtype=np.int64) #Half of a beat before
    i2 = i2 + np.arange(N)
    i2 = np.maximum(i2, 0)
    
    #Step 2: Do dynamic programming
    backlink = -1*np.ones(N, dtype=np.int64) #Best predecessor for this point
    cscore = np.array(s.origNovFn[0:N]) #Cumulative score
    for i in range(N):
        if (i1[i] == i2[i]):
            continue
        #Log-penalize how far away from 2*pi the interval is
        indices = np.arange(i1[i], i2[i]+1)
        indices = indices[(theta[i]-theta[indices]) > 0]
        if (len(indices) == 0):
            continue
        txcost = -tightness*(np.log(np.abs(theta[i]-theta[indices])/(2*np.pi))**2)
        scores = cscore[indices] + txcost
        #Find best predecessor location
        maxidxrel = np.argmax(scores)
        maxidx = indices[maxidxrel]
        backlink[i] = maxidx
        cscore[i] = scores[maxidxrel] + alpha*s.novFn[maxidx]
    
    #Step 3: Backtrace to extract final onsets
    b = [np.argmax(cscore)]
    while not (b[-1] == -1):
        b.append(backlink[b[-1]])
    b = np.array(b[0:-1])
    b = np.fliplr(b[None, :]).flatten()
    return (b, cscore[b[-1]])

def searchTempoRange(theta, s, tightness, alpha):
    """
    Run dynamic programming several times on different integer scalings
    """
    NScales = 9
    AllScores = np.zeros(NScales)
    AllOnsets = []
    for i in range(1, NScales+1):
        (onsets, score) = getOnsetsDP(theta/i, s, tightness, alpha)
        x = np.zeros(len(s.novFn))
        x[onsets] = 1
        x = np.convolve(x, np.exp(np.arange(-5, 6)**2/9), 'same')
        AllScores[i-1] = i*np.sum(s.novFn*x)
        AllOnsets.append(onsets)
    return (AllOnsets, AllScores)

def evalThetas(theta, novFn, W):
    """
    Project sliding windows of the novelty function onto sliding windows
    of cosine of theta and return the sum of the magnitudes of all projections
    """
    N = len(theta)
    M = N-W+1
    ret = 0.0
    ctheta = np.cos(theta)
    f = novFn[0:N]
    for i in range(M):
        s = np.array(novFn[i:i+W])
        c = np.array(ctheta[i:i+W])
        #Normalize both
        s = s/np.sqrt(np.sum(s**2))
        c = c/np.sqrt(np.sum(c**2))
        ret += np.abs(np.sum(c*s))
    return ret
