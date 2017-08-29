"""
Programmer: Chris Tralie
Purpose: An assortment of geometry tools, such as mean shift,
greedy permutation, etc
"""
import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

def getMeanShift(X, theta = np.pi/16):
    N = X.shape[0]
    eps = np.cos(theta)
    XS = X/np.sqrt(np.sum(X**2, 1))[:, None]
    D = XS.dot(XS.T)
    J, I = np.meshgrid(np.arange(N), np.arange(N))
    J = J[D >= eps]
    I = I[D >= eps]
    V = np.ones(I.size)
    D = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    XMean = np.zeros(X.shape)
    for i in range(N):
        idx = D[i, :].nonzero()[1]
        print idx
        XMean[i, :] = np.mean(X[idx, :], 0)
    return XMean

def getMeanShiftKNN(X, K):
    N = X.shape[0]
    D = np.sum(X**2, 1)[:, None]
    D = D + D.T - 2*X.dot(X.T)
    allidx = np.argsort(D, 1)
    XMean = np.zeros(X.shape)
    for i in range(N):
        idx = allidx[i, 0:K]
        print idx
        XMean[i, :] = np.mean(X[idx, :], 0)
    return XMean

def getGreedyPerm(D):
    """
    Naive O(N^2) algorithm to do the greedy permutation
    param: D (NxN distance matrix for points)
    returns: (permutation (N-length array of indices), 
              lambdas (N-length array of insertion radii))
    """
    N = D.shape[0]
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)

def get2DCircleScore(X, sigma = 0.5):
    Y = X - np.mean(X, 0)[None, :]
    Y = Y/np.std(Y, 0) #Normalize by standard deviation
    Z = Y/np.sqrt(np.sum(Y**2, 1))[:, None]
    V = Y - Z
    return np.exp(-np.sum(V**2)/(sigma**2*X.shape[0]))

if __name__ == '__main__':
    t = np.linspace(0, 10*np.pi, 301)[0:300]
    X = np.zeros((len(t), 2))
    X[:, 0] = np.cos(t)
    X[:, 1] = np.sin(t)
    np.random.seed(420)
    X = X + 0.1*np.random.randn(X.shape[0], 2)
    XMean = getMeanShiftKNN(X, 10)

    plt.plot(X[:, 0], X[:, 1], '.')
    plt.hold(True)
    plt.scatter(XMean[:, 0], XMean[:, 1], 20)
    plt.show()
