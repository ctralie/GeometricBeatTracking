import numpy as np

#theta: Circular coordinate estimates
#s: BeatingSound object (has Fs, nvFn, hopSize)
#tightness: How much to weight the beat penalty
#alpha: How much to weight the novelty function
def getOnsetsDP(theta, s, tightness, alpha = 0.8):
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
    i1 = np.array(np.round(-2*np.pi/slopes), dtype=np.int64) #2 beats before
    i1 = i1 + np.arange(N)
    i1 = np.maximum(i1, 0)
    i2 = np.array(np.round(-np.pi/(2*slopes)), dtype=np.int64) #Half of a beat before
    i2 = i2 + np.arange(N)
    i2 = np.maximum(i2, 0)
    
    #Step 2: Do dynamic programming
    backlink = -1*np.ones(N, dtype=np.int64) #Best predecessor for this point
    cscore = np.array(s.novFn[0:N]) #Cumulative score
    for i in range(N):
        if (i1[i] == i2[i]):
            continue
        #Log-penalize how far away from 2*pi the interval is
        indices = np.arange(i1[i], i2[i]+1)
        indices = indices[(theta[i]-theta[indices]) > 0]
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
    return b
