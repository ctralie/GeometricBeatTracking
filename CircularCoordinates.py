import numpy as np
import matplotlib.pyplot as plt
from CSMSSMTools import *
from Laplacian import *
from SlidingWindow import *
from SimilarityFusion import *
from GeometryTools import *

def getThresholdsFromPDs(PDs, thresh = 0.0, useDeaths = False):
    N = len(PDs)
    ts = []
    idxs = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            D = getCSM(PDs[i], PDs[j])
            x = np.min(D, 1)
            pers = PDs[i][:, 1] - PDs[i][:, 0]
            bs = PDs[i][:, 0]
            ts = ts + bs[(x > 0)*(pers > thresh)].tolist()
            idx = np.arange(len(x))
            idx = idx[(x > 0)*(pers > thresh)]
            for ii in idx:
                idxs.append([i, ii])
            if useDeaths:
                ds = PDs[i][:, 1]
                ts = ts + ds[(x > 0)*(pers > thresh)].tolist()
    #ts = np.unique(np.array(ts))
    ts = np.array(ts)
    return (ts, idxs)

def massageThresholds(ts):
    ret = np.array(ts)
    ret[0:-1] = 0.9*ret[0:-1] + 0.1*ret[1::]
    return ret

def getCircularCoordsFromThreshs(D, ts, doPlot = False):
    ret = []
    for t in ts:
        if doPlot:
            plt.clf()
        ret.append(getLapCircularCoordinatesThresh(D, t, doPlot))
        if doPlot:
            plt.show()
    return ret

def getAverageSlope(thetas):
    N = len(thetas)
    idx = np.arange(N)
    Ds = thetas[None, :] - thetas[:, None]
    norm = idx[None, :] - idx[:, None]
    [J, I] = np.meshgrid(idx, idx)
    return np.mean(Ds[I < J]/norm[I < J])


def getPDsFields(D, times, fields = [2, 3, 5, 7], doPlot = False):
    AllPDs = []
    plt.figure(figsize=((len(fields)+1)*5, 5))
    
    if doPlot:
        plt.subplot(1, len(fields)+1, 1)
        plt.imshow(D, extent = (0, times[D.shape[1]-1], times[D.shape[1]-1], 0), cmap = 'afmhot', interpolation = 'nearest')
        plt.title("Sliding Window SSM")
        plt.xlabel("times (Seconds)")
        plt.ylabel("times (Seconds)")
    for i in range(len(fields)):
        f = fields[i]
        print "Doing field %i"%f
        PDs = doRipsFiltrationDM(D, 1, coeff=f)
        if PDs[1].size > 0:
            AllPDs.append(PDs[1])
            if doPlot:
                plt.subplot(1, len(fields)+1, 2+i)
                plotDGM(PDs[1], color = 'b')
                plt.xlim([0, ymax])
                plt.ylim([0, ymax])
        if doPlot:
            plt.title("$\mathbb{Z}%i$"%f)
    return AllPDs

def getTemposBlocks(s, Win, BlockWin, BlockHop, NFuseBlocks, SkipBlocks, QuantLevels = 10, plotIndividualBlocks = False, plotFusion = False, plotCoords = False):
    """
    Get tempos in blocks of audio
    :param s: A BeatingSound object holding a computed novelty function / window size / hop size
    :param Win: Window size within block
    :param BlockWin: Length of sliding window within a block, in novelty audio samples
    :param BlockHop: Hop length between overlapping blocks
    :param NFuseBlocks: The number of blocks to fuse within a superblock
    :param SkipBlocks: The number of blocks to skip between superblocks
    :param QuantLevels: 
    """
    x = s.novFn
    Fs = s.Fs
    hopSize = s.hopSize
    winSize = s.winSize
    NBlocks = (len(x) - BlockWin)/BlockHop + 1
    Ds = []
    if plotIndividualBlocks:
        plt.figure(figsize=(10, 10))
    for i in range(NBlocks):
        thisx = x[i*BlockHop:i*BlockHop+BlockWin]
        YR = getSlidingWindowNoInterp(thisx, Win)
        #YR = getSlidingWindow(thisx, Win, 1, dT)
        YR = YR - np.mean(YR, 1)[:, None]
        YR = YR/np.sqrt(np.sum(YR**2, 1))[:, None]
        D = getSSM(YR)
        #DQuantized = quantizeCSM(D, 10)
        ymax = int(np.ceil(np.max(D)*1.1)) #For plotting everything on the same scale
        if plotIndividualBlocks:
            times = (i*BlockHop+np.arange(len(thisx)))*hopSize/float(Fs)
            plt.clf()
            plt.subplot(221)
            plt.plot(times, thisx)
            plt.xlabel("times (sec)")
            plt.ylabel("Audio Novelty")
            plt.title("Audio Novelty Block %i"%i)
            plt.subplot(222)
            plt.scatter(np.arange(D.shape[0]), D[0, :], 20, c = 'b')
            plt.scatter(np.arange(D.shape[0]), D[0, :], 5, c = 'r', edgecolor = 'none')
            plt.title("SSM Row 0")
            plt.subplot(223)
            plt.imshow(D, extent = (times[0], times[int(D.shape[1]*dT)-1], times[int(D.shape[1]*dT)-1], times[0]), cmap = 'afmhot', interpolation = 'nearest')
            plt.title("SSM")
            plt.subplot(224)
            #plt.imshow(DQuantized, extent = (times[0], times[int(D.shape[1]*dT)-1], times[int(D.shape[1]*dT)-1], times[0]), cmap = 'afmhot', interpolation = 'nearest')
            plt.title("SSM Quantized")
            plt.savefig("Block%i.svg"%i, bbox_inches = 'tight')
        Ds.append(D)
    

    if plotCoords:
        plt.figure(figsize=(10, 10))
    
    times = np.arange(Ds[0].shape[0])*hopSize/float(Fs)
    NFused = (NBlocks - NFuseBlocks)/SkipBlocks + 1
    AllTempos = []
    for b in range(NFused):
        print "Doing similarity fusion on %i matrices of size %ix%i..."%(len(Ds), Ds[0].shape[0], Ds[0].shape[1])
        tic = time.time()
        thisDs = Ds[b*SkipBlocks:b*SkipBlocks+NFuseBlocks]
        PlotNames = []
        if plotFusion:
            PlotNames = ["%i_%i"%(b, i) for i in range(len(DsFused))]
        DFused = doSimilarityFusion(thisDs, K=QuantLevels, PlotNames = PlotNames)
        np.fill_diagonal(DFused, 0)
        np.fill_diagonal(DFused, np.max(DFused))
        DFused = np.max(DFused) - DFused
        D = quantizeCSM(DFused, 10)
        D = 2*D/np.max(D)
        toc = time.time()
        print "Finished similarity fusion for superblock %i"%b
        print "Elapsed times: ", toc - tic, " seconds"

        threshs = np.unique(D.flatten())
        threshs = threshs[0:-1]
        CircCoords = getCircularCoordsFromThreshs(D, threshs)
        for i in range(len(threshs)):
            V = CircCoords[i]['v'][:, 1:3]
            theta = CircCoords[i]['thetau']
            estT = (hopSize/float(Fs))*(2*np.pi) / (getAverageSlope(theta))
            bmp = 60.0/estT
            score = get2DCircleScore(V)
            AllTempos.append([bmp, score])
            if plotCoords:
                plt.clf()
                t = times[0:YR.shape[0]]
                plt.subplot2grid((2, 2), (0, 0), colspan = 2)
                plt.plot(t, V)
                plt.xlim([t[0], t[-1]])
                plt.title("Thresh = %.3g, Estimated T = %g bmp, Score = %.3g"%(threshs[i], bmp, score))
                plt.subplot(223)
                plt.scatter(V[:, 0], V[:, 1])
                plt.subplot(224)
                D = CircCoords[i]['A']
                plt.imshow(D, cmap = 'gray', interpolation = 'none')
                plt.savefig("CoordsReal%i_%i.svg"%(b, i), bbox_inches = 'tight')
    return np.array(AllTempos)
