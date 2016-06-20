import numpy as np
import scipy.io as sio
import networkx as nx
import sys
sys.path.append("..")
from CircularCoordinates import *
from SyntheticFunctions import *
from SoundTools import *
import sklearn.decomposition

EIG1 = 1
EIG2 = 2

def plotResults(X, D, A, v, theta, Kappa):    
    pca = sklearn.decomposition.PCA(n_components = 2)
    pca.fit(X.T)
    Y = pca.transform(X.T)
    Y = Y.T
    
    plt.subplot(231)
    plt.imshow(D)
    plt.title('SSM')
    
    plt.subplot(232)
    plt.spy(A)
    plt.title("Adjacency Matrix")
    
    plt.subplot(233)
    N = A.shape[0]
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    I = I[A > 0]
    J = J[A > 0]
    plt.scatter(Y[0, :], Y[1, :], 20, c = theta, edgecolors = 'none')
    plt.hold(True)
    for i in range(len(I)):
        x1 = Y[:, I[i]]
        x2 = Y[:, J[i]]
        plt.plot([x1[0], x2[0]], [x1[1], x2[1]], 'k')
    plt.title("Kappa = %g"%Kappa)
    
    plt.subplot(234)
    plt.plot(v[:, EIG1], 'b')
    plt.hold(True)
    plt.plot(v[:, EIG2], 'r')
    plt.title('Eigenvectors %i and %i'%(EIG1, EIG2))
    
    plt.subplot(235)
    plt.scatter(v[:, EIG1], v[:, EIG2], c=np.arange(v.shape[0]), edgecolors = 'none')
    plt.title('Eigenvectors %i and %i'%(EIG1, EIG2))
    
    plt.subplot(236)
    plt.plot(theta % (2*np.pi))
    plt.title('Circular Coordinates')


if __name__ == '__main__':
    np.random.seed(100)
    T = 150
    W = 125
    noiseSigma = 0
    gaussSigma = 2
    NSamples = 250

    y = np.cos(4*np.pi*np.arange(NSamples)/NSamples)
    y += np.cos(8*np.pi*np.arange(NSamples)/NSamples)
    
#    x = getPulseTrain(NSamples, T, T, 1, 1)
#    x += getPulseTrain(NSamples, T/3, T/3, 1, 1)
#    y = convolveAndAddNoise(x, gaussSigma, noiseSigma)
    
    s = BeatingSound()
    s.novFn = y
    plt.plot(y)
    plt.show()
    X = s.getSlidingWindowFull(W)
#    sio.savemat("X.mat", {"X":X.T})
#    I = sio.loadmat("I.mat")['I']
#    ds = np.sort(I.flatten())
#    #Put a distance threshold right before and right after each event
#    newds = []
#    eps = 0.01
#    for i in range(len(ds)):
#        newds.append(ds[i]-eps)
#        newds.append(ds[i])
#        newds.append(ds[i]+eps)
#    ds = np.array(newds)
    i = 0
    for Kappa in [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        print "Kappa = ", Kappa
        res = getLaplacianBlock((X, True, -Kappa, None))
        [D, A, L, v] = [res['D'], res['A'], res['L'], res['v']]
        theta = getThetas(v)
        sio.savemat("L%i.mat"%i, {"L":L})
        plt.clf()
        plotResults(X, D, A, v, theta, Kappa)
        plt.savefig("%i.png"%i, dpi=150)
        i = i + 1    
