import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
from SyntheticFunctions import *
from SlidingWindow import *
from Laplacian import *

if __name__ == '__main__':
    plt.figure(figsize=(9, 4))

    N = 8
    X = np.zeros((N*2, 2))
    X[0:N, 0] = np.cos(2*np.pi*np.arange(N)/N)
    X[0:N, 1] = np.sin(2*np.pi*np.arange(N)/N)
    X[N::, 0] = 0.6*np.cos(2*np.pi*np.arange(N)/N)
    X[N::, 1] = 0.6*np.sin(2*np.pi*np.arange(N)/N)

    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, X.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    
    
    plt.subplot(251)
    for i in range(N*2):
        plt.plot(X[[i, (i+1)%(2*N)], 0], X[[i, (i+1)%(2*N)], 1], 'k', lineWidth=2)
    plt.scatter(X[:, 0], X[:, 1], 100, c=C)
    plt.axis('off')
    plt.axis('equal')
    plt.subplot(256)
    for i in range(N):
        plt.plot(X[[i, (i+N)%(2*N)], 0], X[[i, (i+N)%(2*N)], 1], 'k', lineWidth=2)
    for i in range(N*2):
        plt.plot(X[[i, (i+1)%(2*N)], 0], X[[i, (i+1)%(2*N)], 1], 'k', lineWidth=2)
    plt.scatter(X[:, 0], X[:, 1], 100, c=C)
    plt.axis('off')
    plt.axis('equal')
    
    
    N = N*2
    X[:, 0] = np.cos(2*np.pi*np.arange(N)/N)
    X[:, 1] = np.sin(2*np.pi*np.arange(N)/N)
    plt.subplot(252)
    for i in range(N):
        plt.plot(X[[i, (i+1)%N], 0], X[[i, (i+1)%N], 1], 'k', lineWidth=2)     
    plt.scatter(X[:, 0], X[:, 1], 100, c=C)   
    plt.axis('off')
    plt.axis('equal')
    plt.subplot(257)
    for i in range(N/2):
        plt.plot(X[[i, (i+N/2)%N], 0], X[[i, (i+N/2)%N], 1], 'k', lineWidth=2)
    for i in range(N):
        plt.plot(X[[i, (i+1)%N], 0], X[[i, (i+1)%N], 1], 'k', lineWidth=2)     
    plt.scatter(X[:, 0], X[:, 1], 100, c=C)   
    plt.axis('off')
    plt.axis('equal')
    
    
    A = np.zeros((N, N))
    for i in range(N):
        A[i, (i+1)%N] = 1
        A[i, (i-1)%N] = 1
    plt.subplot(253)
    plt.imshow(A, interpolation = 'nearest', cmap = 'gray')
    plt.scatter(-1*np.ones(A.shape[0]), np.arange(A.shape[0]), 30, c=C)
    plt.scatter(np.arange(A.shape[0]), -1*np.ones(A.shape[0]), 30, c=C)
    plt.axis('off')

    A2 = np.array(A)
    for i in range(N):
        A2[i, (i+N/2)%N] = 1
    plt.subplot(258)
    plt.imshow(A2, interpolation = 'nearest', cmap = 'gray')
    plt.scatter(-1*np.ones(A.shape[0]), np.arange(A.shape[0]), 30, c=C)
    plt.scatter(np.arange(A.shape[0]), -1*np.ones(A.shape[0]), 30, c=C)
    plt.axis('off')
    
    #Get eigenvectors
    (w, v, L) = getLaplacianEigsDense(A, 3)
    (w2, v2, L2) = getLaplacianEigsDense(A2, 3)
    plt.subplot(254)
    plt.plot(v[:, 1:3])
    plt.axis('off')
    plt.scatter(np.arange(A.shape[0]), 1.2*np.min(v[:, 1:3])*np.ones(A.shape[0]), 30, c=C)
    plt.xlim([-1, N])
    plt.subplot(259)
    plt.axis('off')
    plt.scatter(np.arange(A.shape[0]), 1.2*np.min(v[:, 1:3])*np.ones(A.shape[0]), 30, c=C)
    plt.plot(v2[:, 1:3])
    
    """
    plt.subplot(255)
    plt.scatter(v[:, 1], v[:, 2], 100, c=C) 
    plt.axis('off')
    plt.axis('equal')
    plt.subplot(2, 5, 10)
    scale = np.linspace(1, 1.4, v2.shape[0])
    v2 = v2*scale[:, None]
    plt.scatter(v2[:, 1], v2[:, 2], 100, c=C)
    plt.axis('off')
    plt.axis('equal')
    """
    plt.subplot(255)
    plt.plot(np.arctan2(v[:, 2], v[:, 1]))
    plt.scatter(np.arange(A.shape[0]), (-np.pi*1.1)*np.ones(A.shape[0]), 30, c=C)
    plt.xlim([-1, v.shape[0]])
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(["$-\pi$", "0", "$\pi$"])
    
    plt.subplot(2, 5, 10)
    plt.plot(np.arctan2(v2[:, 2], v2[:, 1]))
    plt.scatter(np.arange(A.shape[0]), (-np.pi*1.1)*np.ones(A.shape[0]), 30, c=C)
    plt.xlim([-1, v.shape[0]])
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(["$-\pi$", "0", "$\pi$"])
    
    plt.savefig("MoebiusLadder.svg", bbox_inches = 'tight')
