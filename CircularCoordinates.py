import numpy as np
import scipy.io as sio
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
    #Take one additional neighbor to account for the fact
    #that the diagonal of the SSM is all zeros
    elif Kappa < 1:
        NNeighbs = int(np.round(Kappa*N))+1
    else:
        NNeighbs = Kappa+1
    NNeighbs = min(N, NNeighbs)
    cols = np.argsort(D, 1)
    temp, rows = np.meshgrid(np.arange(N), np.arange(N))
    cols = cols[:, 0:NNeighbs].flatten()
    rows = rows[:, 0:NNeighbs].flatten()
    ret = np.zeros((N, N))
    ret[rows, cols] = 1
    return ret

def getAdjacencyKappa(D, Kappa):
    B1 = SSMToBinary(D, Kappa)
    B2 = SSMToBinary(D.T, Kappa)
    ret = B1*B2.T
    np.fill_diagonal(ret, 0)
    return ret

def getAdjacencySigma(D, Sigma):
    ret = np.array(D < Sigma, dtype=np.float64)
    np.fill_diagonal(ret, 0)
    return ret

def getLaplacianBlock(args):
    AddTimeEdges = False
    (X, Normalize, Sigma, s) = args
    NEig = 3
    #Compute SSM
    XSum = np.sum(X**2, 0)
    if Normalize:
        X = X/np.sqrt(XSum)
        XSum = np.ones(XSum.shape)
    D = XSum[:, None] + XSum[None, :] - 2*(X.T.dot(X))
    D[D < 0] = 0
    D = np.sqrt(D)
    
    #Compute spectral decomposition
    #If Sigma is negative, treat it as "Kappa" (proportion of mutual nearest neighbors)
    if Sigma > 0:
        A = getAdjacencySigma(D, Sigma)
    else:
        A = getAdjacencyKappa(D, -Sigma)
    #A = np.exp(-D*10)
    if AddTimeEdges:
        A[range(1, D.shape[0]), range(D.shape[0]-1)] = 1
        A[range(D.shape[0]-1), range(1, D.shape[0])] = 1
    #plotAdjacencyEdgesGraph(A)
    
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
    return {'D':D, 'A':A, 'L':L, 'v':v}

def RMSScoreBlock(pv):
    v = np.array(pv[:, [EIG1, EIG2]])
    N = v.shape[0]
    #Center on centroid
    v = v - np.mean(v, 0, keepdims=True)
    #RMS Normalize
    v = v*np.sqrt(N/np.sum(v**2))
    #Compute mean distance from circle
    ds = np.sqrt(np.sum(v**2, 1))
    score = np.sum(np.abs(ds - 1))/float(N)
    #TODO: Fit parabola?
    return score

#Use arctangent of mean-centered eigenvectors as estimates of
#circular coordinates
def getThetas(pv):
    v = np.array(pv[:, [EIG1, EIG2]])
    v = v - np.mean(v, 0, keepdims=True)
    theta = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    #Without loss of generality, switch theta to overall increasing
    if theta[-1] - theta[0] < 0:
        theta = -theta
    return theta - theta[0]

#Estimate smoothed versions of slopes in radians per sample
#2*sWin is the size of the rectangular window used to smooth
def getSlopes(thetas, sWin = 10):
    N = len(thetas)
    slopes = np.zeros(N)
    deriv = np.zeros(sWin*2)
    deriv[0:sWin] = np.ones(sWin)
    deriv[sWin:] = -np.ones(sWin)
    slopes[sWin-1:-sWin] = np.convolve(thetas, deriv, 'valid')/float(sWin**2)
    slopes[0:sWin-1] = slopes[sWin-1]
    slopes[-(sWin+1):] = slopes[-(sWin+1)]
    return slopes

#Do circular coordinates in a sliding window with different nearest neighbor thresholds
#for the graph laplacian
#Returns: AllResults[].  Each element is an array corresponding to one of the Kappa values
#Each array is an array of dictionary objects with fields D, A, L, v, score, thetas, and slopes
def getCircularCoordinatesBlocks(s, W, BlockLen, BlockHop, Normalize = True, Kappas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25]):
    #Step 1: Get the sliding window embedding of the audio novelty function
    X = s.getSlidingWindowFull(W)

    #Step 2: Set up the blocks
    idxs = []
    N = X.shape[1]
    NBlocks = int(np.ceil(1 + (N - BlockLen)/BlockHop))
    print "NBlocks = ", NBlocks
    for i in range(NBlocks):
        thisidxs = np.arange(i*BlockHop, i*BlockHop+BlockLen, dtype=np.int64)
        thisidxs = thisidxs[thisidxs < N]
        idxs.append(thisidxs)
    #Pull out all blocks
    Blocks = []
    for i in range(NBlocks):
        Blocks.append(np.array(X[:, idxs[i]]))
    
    #Step 3: Compute the Laplacian in all blocks for each Kappa
    AllResults = []
    for Kappa in Kappas:
        print "Doing Kappa = ", Kappa, "..."
        tic = time.time()
        Results = []
        ##Prepare for parallel processing
        #args = zip(Blocks, [Normalize]*len(Blocks), [Kappa]*len(Blocks), [s]*len(Blocks))
        #Results = parpool.map(getCircularCoordinatesBlock, args)
        for i in range(NBlocks):
            res = getLaplacianBlock((Blocks[i], Normalize, -Kappa, s))
            Results.append(res)
        toc = time.time()
        print "Elapsed time circular coordinates computation: ", toc-tic
        AllResults.append(Results)
    
    #Step 4: Compute and score circular coordinates in all blocks for each
    #Kappa, and compute slopes
    for Results in AllResults:
        for res in Results:
            #Step 3: Score each block as circular or not
            res['score'] = RMSScoreBlock(res['v'])
            #Step 4: Convert to circular coordinates
            res['thetas'] = getThetas(res['v'])
            #Step 5: Use circular coordinates to estimate slope
            res['slopes'] = getSlopes(res['thetas'])
    return AllResults

#Aggregate all of the slopes into one place with their confidences
def getInstantaneousTempoArray(s, AllResults, BlockLen, BlockHop):
    N = len(s.novFn)
    Tempos = []
    Scores = []
    for i in range(N):
        Tempos.append([])
        Scores.append([])
    timeScale = (float(s.Fs)/s.hopSize)*60.0/(2*np.pi)
    for Results in AllResults:
        for i in range(len(Results)):
            score = Results[i]['score']
            slopes = Results[i]['slopes']
            thetas = Results[i]['thetas']
            slope = (thetas[-1] - thetas[0])/float(len(thetas))
            for j in range(len(slopes)):
                idx = i*BlockHop + j
                #Tempos[idx].append(timeScale*slopes[j])
                Tempos[idx].append(timeScale*slope)
                Scores[idx].append(np.exp(-score/0.1))
    #Now copy into arrays for convenience
    TemposArr = np.zeros((N, (BlockLen/BlockHop)*len(AllResults)))
    ScoresArr = np.zeros(TemposArr.shape)
    for i in range(len(Tempos)):
        for j in range(len(Tempos[i])):
            TemposArr[i, j] = Tempos[i][j]
            ScoresArr[i, j] = Scores[i][j]
    return (TemposArr, ScoresArr)
