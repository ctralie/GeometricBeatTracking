from SyntheticFunctions import *
from TDA import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.io as sio

def getSlidingWindow(x, dim, Tau, dT):
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim))
    idx = np.arange(N)
    for i in range(NWindows):
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))+2
        if end >= len(x):
            X = X[0:i, :]
            break
        X[i, :] = interp.spline(idx[start:end+1], x[start:end+1], idxx)
    return X

def getSlidingWindowNoInterp(x, dim):
    N = len(x)
    NWindows = N - dim + 1
    X = np.zeros((NWindows, dim))
    idx = np.arange(N)
    for i in range(NWindows):
        X[i, :] = x[i:i+dim]
    return X
