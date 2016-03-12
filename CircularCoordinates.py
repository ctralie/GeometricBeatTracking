import numpy as np
import scipy.sparse
import scipy.stats
import scipy.sparse.linalg as slinalg
import matplotlib.pyplot as plt
import prox_tv as ptv
import time
from SyntheticFunctions import *
from SoundTools import *
from DynProgOnsets import *
from multiprocessing import Pool
from sklearn.decomposition import PCA

EIG1 = 1
EIG2 = 2
VALIDTEMPOMIN = 0.2

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

def plotCircularCoordinates2(s, theta, onsets, gtOnsets):
    plt.subplot(311)
    plt.plot(s.novFn)
    plt.title("Original Function")
    
    plt.subplot(312)
    plt.plot(theta % 2*np.pi)
    plt.title('Circular Coordinates')

    plt.subplot(313)
    plt.stem(onsets, np.ones(len(onsets)), 'b')
    plt.hold(True)
    if len(gtOnsets) > 0:
        plt.stem(gtOnsets, 0.5*np.ones(len(gtOnsets)), 'r')
    plt.title('Onsets')
    plt.show()

#Give different angles a score based on the energy of the novelty function
#that occurs around those angles
def scoreAngles(s, theta, NAngles):
    angles = np.linspace(0, 2*np.pi, NAngles+1)
    angles = angles[0:NAngles]
    scores = np.zeros(NAngles)
    sigma = np.pi/20
    
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

def getCircularCoordinatesBlock(args):
    (X, Normalize, Kappa) = args
    NEig = 4
    #Compute SSM
    XSum = np.sum(X**2, 0)
    if Normalize:
        X = X/np.sqrt(XSum)
        XSum = np.ones(XSum.shape)
    D = XSum[:, None] + XSum[None, :] - 2*(X.T.dot(X))
    
    #Compute spectral decomposition
    A = SSMToBinaryMutual(D, Kappa)
    #A = np.exp(-D*10)
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
        w, v = slinalg.eigsh(L, k=NEig, sigma = 0, which = 'LM')
        #plotEigenvectors(v, NEig)
    except Exception as err:
        print err
        #w = err.eigenvalues
        #v = err.eigenvectors
        w = np.zeros(2)
        v = np.zeros((L.shape[0], max(EIG2, EIG1)+1))
    theta = np.unwrap(np.arctan2(v[:, EIG2], v[:, EIG1]))
    #Without loss of generality, switch theta to overall increasing
    if theta[-1] - theta[0] < 0:
        theta = -theta
    theta = theta - theta[0]
    return (D, L, v, theta)

#Throw away blocks that have a very low slope after linear regression
def discardBadBlocks(s, idxs, BlockAngles):
    A = np.ones((len(idxs[0]), 2))
    validIdx = []
    for i in range(len(idxs)):
        A[:, 0] = idxs[i]
        m, c = np.linalg.lstsq(A, np.array(BlockAngles[i]))[0]
        tempo = m*(s.Fs/float(s.hopSize))/(2*np.pi) #Convert to cycles per second
        if tempo > VALIDTEMPOMIN:
            validIdx.append(i)
    idxs = [idxs[i] for i in validIdx]
    BlockAngles = [BlockAngles[i] for i in validIdx]
    return (idxs, BlockAngles)

#Align adjacent blocks by aligning the median of the y coordinates in their 
#overlapping region
def medianAlignBlocks(idxs, BlockAngles):
    N = len(idxs[0])
    for i in range(1, len(idxs)):
        k = 0
        #Find the point where these sequences overlap
        while k < N:
            #TODO: Could do this faster with binary search
            if idxs[i-1][k] == idxs[i][0]:
                break
            k += 1
        if k == N:
            print "ERROR: Blocks at incides %i and %i don't overlap"%(i-1, i)
        else:
            med1 = np.median(BlockAngles[i-1][k::])
            med2 = np.median(BlockAngles[i][0:N-k])
            BlockAngles[i] += (med1-med2)

#Merge overlapping blocks by taking the median of the circular coordinates
#in overlapping regions after alignment
def medianMergeBlocks(idxs, BlockAngles, N, BlockLen, BlockHop):
    theta = np.nan*np.ones((BlockLen/BlockHop, N))
    currIdx = np.zeros(N, dtype=np.int64)
    for i in range(len(idxs)):
        theta[currIdx[idxs[i]], idxs[i]] = BlockAngles[i]
        currIdx[idxs[i]] += 1
    return scipy.stats.nanmedian(theta, 0)

def extendThetas(theta, N):
    N1 = len(theta)
    theta2 = np.zeros(N)
    theta2[0:N1] = theta
    slope = (theta[-1]-theta[0])/float(N1)
    theta2[N1:N] = theta[-1] + slope*np.arange(N-N1)
    return theta2

#Do circular coordinates in a sliding window and aggregate the results
#in a consistent way
def getCircularCoordinatesBlocks(s, W, pca, BlockLen, BlockHop, parpool, gaussWin = 1, Normalize = True, denoise = True, doPlot = True, Kappa = 0.1):
    #Step 0: Perform Sliding Window Denoising
    novFnOrig = np.array(s.novFn)
    if denoise:
        orig = s.origNovFn
        if gaussWin > 1:            
            s.smoothNovFn(gaussWin)
            if doPlot:
                plt.clf()
                plt.plot(s.origNovFn, 'b')
                plt.hold(True)
                plt.plot(s.novFn, 'g')
                plt.title('Smoothed by %i'%gaussWin)
                plt.show()
    
    #Step 1: Perform PCA on a sliding window over the entire song
    X = s.getSlidingWindowFull(W)
    if pca:
        if doPlot:
            (U, S) = s.getSlidingWindowLeftSVD(W)
            plt.clf()
            plt.plot(np.cumsum(S)/np.sum(S))
            plt.hold(True)
            plt.plot(np.arange(len(S)), 0.4*np.ones(len(S)), 'g')
            plt.plot(np.arange(len(S)), 0.8*np.ones(len(S)), 'g')
            plt.title('Singular Values')
            plt.show()
        X = pca.fit_transform(X.T).T
                
    Ds = []
    Ls = []
    vs = []
    idxs = []
    N = X.shape[1]
    NBlocks = int(np.ceil(1 + (N - BlockLen)/BlockHop))
    print "NBlocks = ", NBlocks
    BlockAngles = [] #Holds the circular coordinates for each different block
    #Step 2: Get the circular coordinates in blocks
    tic = time.time()
    for i in range(NBlocks):
        thisidxs = np.arange(i*BlockHop, i*BlockHop+BlockLen, dtype=np.int64)
        thisidxs = thisidxs[thisidxs < N]
        idxs.append(thisidxs)
    #Pull out all blocks and prepare for parallel processing
    Blocks = []
    for i in range(NBlocks):
        Blocks.append(np.array(X[:, idxs[i]]))
    Normalize = True
    args = zip(Blocks, [Normalize]*len(Blocks), [Kappa]*len(Blocks))
    #res = parpool.map(getCircularCoordinatesBlock, args)
    for i in range(NBlocks):
        (D, L, v, theta) = getCircularCoordinatesBlock((Blocks[i], Normalize, Kappa))
        #(D, L, v, theta) = res[i]
        BlockAngles.append(theta)
        Ds.append(D)
        Ls.append(L)
        vs.append(v)
    toc = time.time()
    print "Elapsed time circular coordinates computation: ", toc-tic
    
    if doPlot:
        plt.hold(True)
        for i in range(len(idxs)):
            plt.plot(idxs[i]*s.hopSize/float(s.Fs), BlockAngles[i])
        plt.title("All Blocks")
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Theta (Radians)")
        plt.show()
    #Step 3: Discard bad blocks and plot the blocks before alignment
    (idxs, BlockAngles) = discardBadBlocks(s, idxs, BlockAngles)
    if doPlot:
        plt.hold(True)
        for i in range(len(idxs)):
            plt.plot(idxs[i]*s.hopSize/float(s.Fs), BlockAngles[i])
        plt.title("Good Blocks Before Alignment")
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Theta (Radians)")
        plt.show()
    #Step 4: Median align the blocks and plot the result
    medianAlignBlocks(idxs, BlockAngles)
    if doPlot:
        plt.hold(True)
        for i in range(len(idxs)):
            plt.plot(idxs[i]*s.hopSize/float(s.Fs), BlockAngles[i])
        plt.title("Blocks After Alignment")
        plt.xlabel("Time (Seconds)")
        plt.ylabel("Theta (Radians)")
        plt.show()

    #Step 4: Finish merging all of the blocks by taking the median of values at the same time coordinate
    theta = medianMergeBlocks(idxs, BlockAngles, N, BlockLen, BlockHop)
    theta = extendThetas(theta, len(s.novFn))
    #Fill in nan values with zero order hold (not the best but gets the job done)
    if np.isnan(theta[0]):
        theta[0] = 0
    for i in range(1, len(theta)):
        if np.isnan(theta[i]):
            theta[i] = theta[i-1]
    if doPlot:
        plt.plot(theta)
        plt.show()
    return theta
