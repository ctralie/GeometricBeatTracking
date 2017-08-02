import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import sys
sys.path.append("..")
from Laplacian import *

def getCirculantAdj(N, lags):
    #Setup circular parts
    I = range(N)*(len(lags)+2)
    J = range(1, N+1) + range(-1, N-1)
    J[N-1] = 0
    J[N] = N-1
    for lag in lags:
        J = J + (np.mod(np.arange(N) + lag, N)).tolist()
    V = np.ones(len(I))
    return sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

def getOneOnK(N, k):
    lags = [i*N/k for i in range(1, k)]
    return getCirculantAdj(N, lags)

def getCircleEigs(N):
    lambdas = np.zeros(N)
    for i in range(1, N/2+1):
        val = 2 - 2*np.cos(2*np.pi*i/N)
        i1 = i*2-1
        i2 = i*2
        lambdas[i1] = val
        if i2 < N:
            lambdas[i2] = val
    return lambdas

def getMoebiusEigs(N):
    lambdas = np.zeros(N)
    for i in range(1, N/2+1):
        val = 3 - 2*np.cos(2*np.pi*i/N) - (-1)**i
        i1 = i*2-1
        i2 = i*2
        lambdas[i1] = val
        if i2 < N:
            lambdas[i2] = val
    return (lambdas, np.sort(lambdas))

def get3WayEigs(N):
    lambdas = np.zeros(N)
    for i in range(1, N/2+1):
        val = 4 - 2*np.cos(2*np.pi*i/N) - 2*np.cos(2*np.pi*i/3)
        i1 = i*2-1
        i2 = i*2
        lambdas[i1] = val
        if i2 < N:
            lambdas[i2] = val
    return (lambdas, np.sort(lambdas))

if __name__ == '__main__':
    N = 100
    A = getOneOnK(N, 2)
    #A = getCirculantAdj(N, [30, 60, 80])
    A = A.toarray()
    (w, v, L) = getLaplacianEigsDense(A, A.shape[0])
    
    (lambdas, lambdassorted) = get3WayEigs(N)
    
    plt.figure(figsize=(15, 4))
    plt.subplot(132)
    plt.plot(lambdas)
    plt.title("Eigenvalues")
    plt.xlabel("Eigenvalue Number")
    plt.ylabel("Eigenvalue")
    
#    plt.subplot(224)
#    plt.scatter(w, lambdassorted)
#    plt.xlabel("Numerically Computed")
#    plt.ylabel("Analytic")
#    plt.axis('equal')
#    plt.title("Checking accuracy")
    
    plt.subplot(131)
    plt.imshow(A, interpolation = 'nearest', cmap = 'gray')
    plt.title("Adjacency Matrix")
    
    plt.subplot(133)
    plt.imshow(v, cmap = 'afmhot', aspect = 'auto', interpolation = 'nearest')
    plt.xlabel("k-th Smallest Eigenvector")
    plt.title("Eigenvectors")
    
    plt.savefig("Eigs.svg", bbox_inches = 'tight')
