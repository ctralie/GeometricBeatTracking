import sys
sys.path.append("..")
from SyntheticFunctions import *
from TDA import *
from SlidingWindow import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io as sio


if __name__ == "__main__":
    fields = [2, 3, 5, 7, 11]
    np.random.seed(100)
    N = 500
    T = 50
    noiseSigma = 0
    gaussSigma = 1
    #t = np.linspace(0, 2*np.pi, N+1)[0:N]
    #X = 0.6*np.cos(4*t) + 0.8*np.cos(8*t)

    #(_, X) = getSyntheticPulseTrainPerfectMicrobeats(N, T, noiseSigma, gaussSigma)
    
    #x = 0.5*getSyntheticPulseTrain(N, T, noiseSigma, gaussSigma)
    #x += getSyntheticPulseTrain(N, T*5, noiseSigma, gaussSigma)

    
    plt.figure(figsize=((len(fields)+1)*4, 4))
    t = np.linspace(0, 2*np.pi*N/T, N+1)[0:N]
    for fac in range(2, 10):
        plt.clf()
        thisT = int(round(float(T)/fac))
        x = np.zeros(N)
        #x[0::thisT] = 1
        #x[0::fac*thisT] += 1
        x = getSyntheticPulseTrain(N, thisT, noiseSigma, gaussSigma) + getSyntheticPulseTrain(N, fac*thisT, noiseSigma, gaussSigma)
        Win = fac*thisT*2

        plt.subplot(1, len(fields)+1, 1)
        plt.plot(np.arange(len(x)), x, 'b')
        plt.hold(True)
        plt.plot(np.arange(Win), x[0:Win], 'r')
        
        Y = getSlidingWindowNoInterp(x, Win)
        Y = Y - np.mean(Y, 1)[:, None]

        pca = PCA(n_components=2)
        Z = pca.fit_transform(Y)

        for i in range(len(fields)):
            f = fields[i]
            print "Doing field %i"%f
            plt.subplot(1, len(fields)+1, 2+i)
            PDs = doRipsFiltration(Y, 1, coeff=f, relpath="../")
            sio.savemat("%i_%i.mat"%(fac, fields[i]), {"Y":Y, "I":PDs[1]})
            plotDGM(PDs[1], color = 'b')
            plt.title("$\mathbb{Z}%i$"%f)
        
        plt.savefig("%i.svg"%fac, bbox_inches = 'tight')
