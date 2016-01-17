import numpy as np
import scipy.sparse
import scipy.sparse.linalg as slinalg
import matplotlib.pyplot as plt
import prox_tv as ptv
import time
from SyntheticFunctions import *
from SoundTools import *

EIG1 = 1
EIG2 = 2

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
    k = int(np.ceil(np.sqrt(NEig)))
    for i in range(NEig):
        plt.subplot(k, k, i+1)
        plt.plot(v[:, i])
    plt.show()

def plotCircularCoordinates(s, D, v, theta, onsets, gtOnsets):
    plt.subplot(231)
    plt.plot(s.novFn)
    plt.title("Original Function")
    
    plt.subplot(232)
    plt.imshow(D)
    plt.title('SSM')
    
    plt.subplot(233)
    plt.stem(onsets, np.ones(len(onsets)), 'b')
    plt.hold(True)
    if len(gtOnsets) > 0:
        plt.stem(gtOnsets, 0.5*np.ones(len(gtOnsets)), 'r')
    plt.title('Onsets')
    
    plt.subplot(234)
    plt.plot(v[:, EIG1], 'b')
    plt.hold(True)
    plt.plot(v[:, EIG2], 'r')
    plt.title('Eigenvectors %i and %i'%(EIG1, EIG2))
    
    plt.subplot(235)
    plt.plot(v[:, EIG1], v[:, EIG2])
    plt.title('Eigenvectors %i and %i'%(EIG1, EIG2))
    
    plt.subplot(236)
    plt.plot(theta % 2*np.pi)
    plt.title('Circular Coordinates')
    plt.show()

def getCircularCoordinates(s, W, NPCs, Normalize = True):
    NEig = 16
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
    #A = SSMToBinaryMutual(D, 0.02)
    A = np.exp(-D*10)
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
        w, v = slinalg.eigsh(L, k=NEig, sigma = 0, which = 'LM')
        plotEigenvectors(v, NEig)
        toc = time.time()
        print "Time computing eigenvectors: ", toc-tic
    except Exception as err:
        print err
        w = err.eigenvalues
        v = err.eigenvectors
    theta = np.unwrap(np.arctan2(v[:, EIG2], v[:, EIG1]))
    #Without loss of generality, switch theta to overall increasing
    if theta[-1] - theta[0] < 0:
        theta = -theta
    theta = theta - theta[0]
    return (D, L, v, theta)

#Give different angles a score based on the energy of the novelty function
#that occurs around those angles
def scoreAngles(s, theta, NAngles):
    angles = np.linspace(0, 2*np.pi, NAngles+1)
    angles = angles[0:NAngles]
    scores = np.zeros(NAngles)
    sigma = angles[1]-angles[0]
    
    dtheta = theta[None, :] - angles[:, None]
    dtheta = np.mod(dtheta, 2*np.pi)
    dtheta[dtheta > np.pi] = 2*np.pi - dtheta[dtheta > np.pi] #Ensure proper wraparound
    weights = np.exp(-dtheta**2/(2*sigma**2))
    novFn = s.novFn[0:len(theta)]
    scores = np.abs(weights*novFn[None, :])
    scoresFinal = np.sum(scores, 1)
    return (angles, scores, scoresFinal)

#Determine when the unwrapped angles "t" pass some 2pi offset of "angle"
#Assumes the angle has been unwrapped and is overall increasing
def getOnsetsPassingAngle(t, angle):
    #Find whether each theta is above or below the closest 2pi multiple of angle
    diff = (angle - t) % (2*np.pi)
    diff[diff > np.pi] = -1 #These angles are above the closest theta
    diff[diff >= 0] = 1
    idx = np.arange(len(diff)-1)
    idx = idx[diff[1::] - diff[0:-1] == 2]
    #TODO: Denoise noisy transitions
    return idx

if __name__ == "__main__":
    np.random.seed(100)
    T = 200
    NPCs = 10
    noiseSigma = 0.05
    gaussSigma = 3
    (gtPulses, x) = getSyntheticPulseTrainFreqAmpDrift(5000, T-30, T+30, 1, 1, noiseSigma, gaussSigma)
    #(gtPulses2, x2) = getSyntheticPulseTrainFreqAmpDrift(5000, T/2, T/2, 1, 1, 0, gaussSigma)
    #gtPulses += gtPulses2
    #x += 0.5*x2
    plt.plot(gtPulses)
    plt.show()
    x = x - np.mean(x)
    s = BeatingSound()
    s.novFn = x

    W = 300
    (D, L, v, theta) = getCircularCoordinates(s, W, NPCs)
    plt.subplot(311)
    plt.plot(theta)
    plt.title('Original Theta')
    theta2 = ptv.tv1_1d(theta, 1)
    plt.subplot(312)
    plt.plot(theta2)
    plt.title('TV Theta')
    plt.subplot(313)
    plt.plot(theta2 - theta)
    plt.title('Difference')
    plt.show()
    theta = theta2
    
    (angles, scores, scoresFinal) = scoreAngles(s, theta, 1000)
    transitionAngle = angles[np.argmax(scoresFinal)]
    
    print "transitionAngle = ", transitionAngle*180/np.pi
    onsets = getOnsetsPassingAngle(theta, transitionAngle)
    gtOnsets = np.arange(len(gtPulses))
    gtOnsets = gtOnsets[gtPulses > 0]
    
#    plt.subplot(121)
#    plt.plot(np.arange(len(theta)), 180*(theta%2*np.pi)/np.pi, 'b')
#    plt.hold(True)
#    plt.plot([0, len(theta)], [transitionAngle*180/np.pi, transitionAngle*180/np.pi], 'r')
    
    plotCircularCoordinates(s, D, v, theta, onsets, gtOnsets)
    
