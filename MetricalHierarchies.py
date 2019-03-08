import numpy as np
import matplotlib.pyplot as plt
import librosa
import sys
import scipy.interpolate as interp

## TODO: Pre-multiply by Hamming window
## TODO: Get the hop size as small as possible with madmom to maximize frequency resolution

def get_dftacf_hadamard_blocks(X, win, hop, do_plot = False):
    """
    Given an audio novelty function, compute the product of the
    STFT magnitude and the frequency-warped ACF in blocks
    Parameters
    ----------
    X: ndarray(N)
        Samples in audio novelty function
    win: int
        Number of samples in each block
    hop: int
        Hop length between blocks
    do_plot: boolean
        Whether to plot the DFT, ACF, and result for each block
    Returns
    -------
    hadamards: ndarray(n_freqs, n_blocks)
        An array of all of the hadamard products in blocks
    freqs: ndarray(n_freqs)
        Frequencies in cycles/sample for every hadamard element
    """
    N = len(X)
    #Exclude last block if it doesn't include a full BlockLen
    #samples so that all frames are the same length
    NBlocks = int(np.floor(1 + (N - win)/hop))
    hadamards = []
    dft_freq = np.array([0])
    for i in range(NBlocks):
        x = X[i*hop:i*hop+win]
        x = x - np.mean(x)
        x = x/np.std(x)
        n = len(x)
        #DFT
        dft = np.abs(np.fft.fft(x))
        if n%2 == 0:
            dft = dft[0:int(n/2)+1]
        else:
            dft = dft[0:int((n-1)/2)+1]
        dft_freq = np.arange(dft.size)/float(n)
        #Autocorrelation
        acf = np.correlate(x, x, 'full')[-n+1::]
        acf[acf < 0] = 0 # Half-wave rectify
        acf_freq = 1.0/(1+np.arange(acf.size))
        acf[0] = 0
        dft[0] = 0

        #acfwarp = interp.spline(acf_freq, acf, dft_freq)
        acfwarp = np.interp(dft_freq, acf_freq[::-1], acf[::-1])
        h = dft*acfwarp
        if do_plot:
            plt.subplot(411)
            plt.plot(x)
            plt.title('x')
            plt.subplot(412)
            plt.plot(dft_freq, acfwarp)
            plt.xlim([0, np.max(dft_freq)])
            plt.title("Frequence Warped Autocorrelation")
            plt.subplot(413)
            plt.plot(dft_freq, dft)
            plt.xlim([0, np.max(dft_freq)])
            plt.title("DFT")
            plt.subplot(414)
            plt.plot(1.0/dft_freq, h)
            plt.title("Hadamard Product")
            plt.tight_layout()
            plt.show()
        hadamards.append(h[1::])
    return np.array(hadamards).T, dft_freq[1::]

def isint(r, tol):
    return np.abs(r-np.round(r))/r < tol

def ppk(j, Ms, peaks, ascending=True, tol=0.05):
    """
    Peak-picking kernel
    fj is the level under analysis and M is the
    metrical structure candidates
    """
    print("j = %g, Ms = %s"%(peaks[j], Ms))
    MsRet = [M + [] for M in Ms]
    if ascending:
        cmpfn = lambda f: float(peaks[f])/peaks[j]
        nextfn = lambda i: i+1
    else:
        cmpfn = lambda f: peaks[j]/float(peaks[f])
        nextfn = lambda i: i-1
    q = nextfn(j)
    # While f_{q+1}/f_q not int
    while q > 0 and q < len(peaks)-1 and not isint(cmpfn(q), tol):
        q = nextfn(q)
    if q < 0 or q >= len(peaks):
        return q, MsRet
    if q == 0 or q == len(peaks)-1:
        if (not (q == j)) and isint(cmpfn(q), tol):
            print("Boundary: fj = %g, fq = %g"%(peaks[j], peaks[q]))
            for M in MsRet:
                M.append(peaks[q])
        return q, MsRet
    qnext = nextfn(q)
    if isint(cmpfn(qnext), tol): #If f_{q+1}/fj is int
        if ascending:
            r = peaks[qnext]/peaks[q]
        else:
            r = peaks[q]/peaks[qnext]
        if not isint(r, tol): #If f_{q+1}/f_q not int
            print("Splitting: fj = %g, fq=%g, fq+1=%g"%(peaks[j], peaks[q], peaks[qnext]))
            Ms1 = [[] + M for M in MsRet]
            Ms2 = [[] + M for M in MsRet]
            j = q
            j, Ms1 = ppk(j, Ms1, peaks, ascending, tol)
            MsRet = MsRet + Ms1
            j = qnext
            j, Ms2 = ppk(j, Ms2, peaks, ascending, tol)
            MsRet = MsRet + Ms2
        else:
            print("Case 2: fj = %g, fq=%g, fq+1=%g"%(peaks[j], peaks[q], peaks[qnext]))
            for M in MsRet:
                M.append(peaks[q])
            j = q
    else:
        print("Case 3: fj = %g, fq=%g, fq+1=%g"%(peaks[j], peaks[q], peaks[qnext]))
        for M in MsRet:
            M.append(peaks[q])
        j = q
    return ppk(j, MsRet, peaks, ascending, tol)

def get_metrical_hierarchy(x, sr, hop=0.36, win=12, peakmin=0.005, bmin=30, bmax=800, tol=0.01, verbose=False):
    """
    Compute a metrical hierarchy for a given onset function
    Parameters
    ----------
    x: ndarray(N)
        Samples in audio novelty function
    sr: int
        The sample rate in hz of the audio novelty function
    win: float
        The length in seconds of each hadamard window
    hop: float
        The hop length in seconds between each hadamard window 
    peakmin: float
        The minimum allowable ratio between a peak and the
        maximum peak
    bmin: float
        Minimum allowable metrical level in beats per minute
    bmax: float
        Maximum allowable metrical level in beats per minute
    tol: float
        The difference factor allowed when considering a tempo
        level to be an integer ratio
    verbose: boolean
        Whether to print progress as the peaks are being filtered
    """
    # Compute hadamard onset frequency blocks
    wini = int(np.round(win*sr))
    hopi = int(np.round(hop*sr))
    h, f = get_dftacf_hadamard_blocks(x, wini, hopi)
    h = np.mean(h, 1)
    h = h/np.max(h)
    h_bpm = sr*f*60
    h[h_bpm < bmin] = 0
    h[h_bpm > bmax] = 0

    # Find the peaks
    peaksidx = np.arange(h.size-2)+1
    peaksidx = peaksidx[(h[1:-1:] > h[0:-2])*(h[1:-1:] > h[2::])]
    peaksidx = peaksidx[h[peaksidx] >= peakmin]
    fmax = peaksidx[np.argmax(h[peaksidx])]
    # Express all peaks in bpm
    peaks = h_bpm[peaksidx]
    fmax = h_bpm[fmax]
    # Filter out all peaks that are not integer ratios of fmax
    if verbose:
        print("fmax = %i"%fmax)
        print(peaks)
    tokeep = np.zeros(peaks.size)
    for i in range(tokeep.size):
        if peaks[i] < fmax:
            r = fmax/peaks[i]
        else:
            r = peaks[i]/fmax
        if isint(r, tol):
            tokeep[i] = 1
    peaks = peaks[tokeep==1]
    if verbose:
        print(peaks)

    
    plt.subplot(211)
    plt.plot(x)
    plt.subplot(212)
    plt.plot(h_bpm, h)
    plt.scatter(h_bpm[peaksidx], h[peaksidx])
    plt.show()


def test_2_1():
    sys.path.append("TheoryValidation")
    from SyntheticFunctions import getGaussianPulseTrain
    N = 4000
    Ts = [100, 200, 50, 150] # 120, 60, 240, 80
    amps = [1.0, 0.5, 0.2, 0.3]
    noiseSigma=0.2
    gaussSigma=0.5
    x = getGaussianPulseTrain(N, Ts, amps, noiseSigma, gaussSigma)
    get_metrical_hierarchy(x, sr=200, verbose=True)

def test_ppk():
    #j, Ms, peaks
    peaks = np.arange(1, 6)
    j = 0
    j, Ms = ppk(j, [[peaks[j]]], peaks)
    print(Ms)

if __name__ == '__main__':
    #test_2_1()
    test_ppk()