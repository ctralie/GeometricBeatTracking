import numpy as np
import scipy.io as sio
import networkx as nx
import sys
sys.path.append("..")
from CircularCoordinates import *

EIG1 = 1
EIG2 = 2

def plotResults(X, D, A, v, theta, Kappa):    
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
    plt.scatter(X[0, :], X[1, :], 20, c = theta, edgecolors = 'none')
    plt.hold(True)
    for i in range(len(I)):
        x1 = X[:, I[i]]
        x2 = X[:, J[i]]
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

#Circle example
def CircleExample(N = 300):
    t = np.linspace(0, 2*np.pi, N+1)
    t = t[0:N]
    X = np.zeros((2, N))
    X[0, :] = np.cos(t)
    X[1, :] = np.sin(t)
    return X

#Noisy circle example
def CircleExampleNoisy(N = 300):
    X = CircleExample(N)
    np.random.seed(10)
    X = X + 0.1*np.random.randn(X.shape[0], X.shape[1])
    return X

def TwoCircleExample():
    r1 = 1
    r2 = 0.5
    X1 = r1*CircleExample(30)
    X2 = np.array([[r1+r2],[0]]) + r2*CircleExample(30)
    X = np.concatenate((X1, X2), 1)
    np.random.seed(10)
    X = X + 0.1*np.random.randn(X.shape[0], X.shape[1])
    return X

if __name__ == '__main__':
    X = TwoCircleExample()
    sio.savemat("X.mat", {"X":X.T})
    I = sio.loadmat("I.mat")['I']
    ds = np.sort(I.flatten())
    #Put a distance threshold right before and right after each event
    newds = []
    eps = 0.01
    for i in range(len(ds)):
        newds.append(ds[i]-eps)
        newds.append(ds[i])
        newds.append(ds[i]+eps)
    ds = np.array(newds)
    i = 0
    for Kappa in ds:
        print "Kappa = ", Kappa
        (D, A, L, v, theta) = getCircularCoordinatesBlock((X, False, Kappa, None))
        sio.savemat("L%i.mat"%i, {"L":L})
        plt.clf()
        plotResults(X, D, A, v, theta, Kappa)
        plt.savefig("%i.png"%i, dpi=150)
        i = i + 1
