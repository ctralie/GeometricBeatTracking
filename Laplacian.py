import sys
import scipy.sparse as sparse
import scipy.stats
import scipy.sparse.linalg as slinalg
import numpy as np
import numpy.linalg as linalg
import scipy.io as sio
import matplotlib.pyplot as plt
from CSMSSMTools import *

def getLaplacianEigs(A, NEigs):
    DEG = sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG - A
    w, v = slinalg.eigsh(L, k=NEigs, sigma = 0, which = 'LM')
    return (w, v, L)

def getLaplacianEigsDense(A, NEigs):
    DEG = scipy.sparse.dia_matrix((A.sum(1).flatten(), 0), A.shape)
    L = DEG.toarray() - A
    w, v = linalg.eigh(L)
    return (w[0:NEigs], v[:, 0:NEigs], L)

def getLapThetas(pv, eig1, eig2):
    """
    Use arctangent of mean-centered eigenvectors as estimates of
    circular coordinates
    """
    v = np.array(pv[:, [eig1, eig2]])
    v = v - np.mean(v, 0, keepdims=True)
    theta = np.arctan2(v[:, 1], v[:, 0])
    thetau = np.unwrap(theta)
    #Without loss of generality, switch theta to overall increasing
    if thetau[-1] - thetau[0] < 0:
        thetau = -thetau
    return (theta, thetau - thetau[0])

def getSlopes(thetas, sWin = 10):
    """
    Estimate smoothed versions of slopes in radians per sample
    2*sWin is the size of the rectangular window used to smooth
    """
    N = len(thetas)
    slopes = np.zeros(N)
    deriv = np.zeros(sWin*2)
    deriv[0:sWin] = np.ones(sWin)
    deriv[sWin:] = -np.ones(sWin)
    slopes[sWin-1:-sWin] = np.convolve(thetas, deriv, 'valid')/float(sWin**2)
    slopes[0:sWin-1] = slopes[sWin-1]
    slopes[-(sWin+1):] = slopes[-(sWin+1)]
    return slopes

def getLapCircularCoordinatesSigma(X, sigma, weighted = False):
    D = getSSM(X)
    if weighted:
        A = np.exp(-D*D/(2*sigma**2))
        (w, v, L) = getLaplacianEigsDense(A, 10)
    else:
        A = getNNAdj(X, Kappa)
        (w, v, L) = getLaplacianEigs(A, 10)
    (theta, thetau) = getLapThetas(v, 1, 2)
    return {'w':w, 'v':v, 'theta':theta, 'thetau':thetau, 'A':A, 'D':D}

def getLapCircularCoordinatesThresh(pD, thresh):
    D = np.array(pD)
    np.fill_diagonal(D, np.inf)
    A = np.zeros(D.shape)
    A[D <= thresh] = 1
    (w, v, L) = getLaplacianEigsDense(A, 10)
    (theta, thetau) = getLapThetas(v, 1, 2)
    return {'w':w, 'v':v, 'theta':theta, 'thetau':thetau, 'A':A, 'D':D}

