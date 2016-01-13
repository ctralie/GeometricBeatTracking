import numpy as np
import scipy.sparse
import scipy.sparse.linalg as slinalg
import matplotlib.pyplot as plt
from SyntheticFunctions import *
from SoundTools import *

def SSMToBinary(D, Kappa):
    N = D.shape[0]
    if Kappa == 0:
        return np.ones((N, N))
    elif Kappa < 1:
        NNeighbs = int(np.round(Kappa*N))
    else:
        NNeighbs = Kappa
    cols = np.argsort(D, 1)
    temp, rows = np.meshgrid(np.arange(N), np.arange(N))
    cols = cols[:, 0:NNeighbs].flatten()
    rows = rows[:, 0:NNeighbs].flatten()
    ret = np.zeros((N, N))
    ret[rows, cols] = 1
    return ret

def SSMToBinaryMutual(D, Kappa):
    B1 = SSMToBinary(D, Kappa)
    B2 = SSMToBinary(D.T, Kappa)
    return B1*B2.T

def plotAdjacencyEdgesPCA(D, A, s):
    #Plot the adjacency matrix on top of the point cloud after PCA
    [I, J] = np.meshgrid(np.arange(A.shape[0]), np.arange(A.shape[0]))
    I = I[A > 0]
    J = J[A > 0]
    Y = (s.S[0:2])[:, None]*s.V[0:2, :]  
    plt.subplot(223)
    plt.spy(scipy.sparse.coo_matrix(A))
    plt.title('Adjacency Matrix')
    plt.subplot(224)
    plt.plot(Y[0, :], Y[1, :], '.')
    plt.title('2D PCA + Adjacency Edges')
    plt.hold(True)
    for i in range(len(I)):
        x1 = Y[:, I[i]]
        x2 = Y[:, J[i]]
        plt.plot([x1[0], x2[0]], [x1[1], x2[1]], 'b')
    plt.subplot(221)
    plt.imshow(D)
    plt.title('SSM')
    plt.subplot(222)
    plt.plot(s.novFn)
    plt.title('Original Function')
    plt.show()

def plotEigenvectors(v, NEig):
    NEig = 10
    k = int(np.ceil(np.sqrt(NEig)))
    for i in range(NEig):
        plt.subplot(k, k, i+1)
        plt.plot(v[:, i])
    plt.show()

if __name__ == "__main__":
    T = 300
    NPCs = 10
    noiseSigma = 0.1
    gaussSigma = 3
    Normalize = False
    (_, x) = getSyntheticPulseTrainFreqAmpDrift(5000, T-30, T+30, 1, 1, noiseSigma, gaussSigma)
    plt.plot(x)
    plt.show()
    x = x - np.mean(x)
    s = BeatingSound()
    s.novFn = x

    W = 200
    (U, S) = s.getSlidingWindowLeftSVD(W)
    SMat = np.eye(NPCs)
    np.fill_diagonal(SMat, S[0:NPCs])
    V = s.getSlidingWindowRightSVD(W, NPCs)
    X = U[:, 0:NPCs].dot(SMat).dot(V)

    #Compute SSM
    XSum = np.sum(X**2, 0)
    if Normalize:
        X = X/np.sqrt(XSum)
        XSum = np.ones(XSum.shape)
    D = XSum[:, None] + XSum[None, :] - 2*(X.T.dot(X))
    
    #Compute spectral decomposition
    A = SSMToBinaryMutual(D, 0.01)
    A[range(1, D.shape[0]), range(D.shape[0]-1)] = 1
    A[range(D.shape[0]-1), range(1, D.shape[0])] = 1
    #plotAdjacencyEdgesPCA(D, A, s)
    
    deg = np.sum(A, 1)
    DEG = np.eye(D.shape[0])
    np.fill_diagonal(DEG, deg)
    L = DEG - A    
    L = scipy.sparse.csc_matrix(L)
    v0 = np.random.randn(L.shape[0], 3)
    try:
        #http://stackoverflow.com/questions/12125952/scipys-sparse-eigsh-for-small-eigenvalues
        tic = time.time()
        w, v = slinalg.eigsh(L, k=3, sigma = 0, which = 'LM')
        toc = time.time()
        print "Time computing eigenvectors: ", toc-tic
    except Exception as err:
        print err
        w = err.eigenvalues
        v = err.eigenvectors
    
    plt.subplot(231)
    plt.plot(s.novFn)
    plt.title("Original Function")
    plt.subplot(232)
    plt.imshow(D)
    plt.title('SSM')
    plt.subplot(233)
    plt.spy(L)
    plt.title('Laplacian Matrix')
    plt.subplot(234)
    plt.plot(v[:, 1], 'b')
    plt.hold(True)
    plt.plot(v[:, 2], 'r')
    plt.title('Eigenvectors 2 and 3')
    plt.subplot(235)
    plt.plot(v[:, 1], v[:, 2])
    plt.title('Eigenvectors 2-3')
    plt.subplot(236)
    plt.plot(np.arctan2(v[:, 2], v[:, 1]))
    plt.title('Circular Coordinates')
    plt.show()
    
