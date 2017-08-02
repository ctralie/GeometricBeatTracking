import sys
sys.path.append("..")
from SyntheticFunctions import *
from SlidingWindow import *
from Laplacian import *
from CSMSSMTools import *
import matplotlib.pyplot as plt
from PulseExperiments import *
from CircularCoordinates import *
import glob
import os
from sys import exit

PREFIX = "temp"

def getLocalMins(x):
    N = len(x)
    idx = np.arange(1, N-1)
    idx = idx[(x[idx] < x[idx+1])*(x[idx] < x[idx-1])]
    return idx

def removeCloseValues(xs, prec = 5):
    ys = np.round(xs*(10**5))
    return np.unique(ys/(10**5))

if __name__ == '__main__':
    colors = ['r', 'g', 'b', 'c', 'm', 'k', 'y']
    doAnimation = False

    NPerPeriod = 200
    facs = (1, 3, 9)
    amps = (2, 1, 1)
    gaussSigma = 1
    noiseSigma = 0

    n0 = lcm(facs)
    f = np.ceil(float(NPerPeriod)/n0)
    T = int(n0*f)
    Ts = [int(T/k) for k in facs]
    N = T*3

    print "facs = %s, Ts = %s"%(facs, Ts)

    xR = getGaussianPulseTrain(N, Ts, amps, noiseSigma, gaussSigma)
    Win = T*2

    YR = getSlidingWindowNoInterp(xR, Win)
    D = getSSM(YR)
    
    plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
    plt.title("SSM %s"%getFacStr(facs, " on "))
    plt.xlabel("Window Number")
    plt.ylabel("Window Number")
    plt.savefig("SSM%s.svg"%getFacStr(facs), bbox_inches = 'tight')
    
    D[D < 1e-5] = 0
    mins = getLocalMins(D[0, :])
    mins = np.unique(D[0, mins])
    mins = np.array([D[0, 1]] + mins.tolist())
    mins = removeCloseValues(mins)
    print "mins = ", mins
    ymax = int(np.ceil(np.max(D)*1.1)) #For plotting everything on the same scale

    #Save SSM and persistence diagrams
    AllPDs = []
    fields = [2, 3, 5, 7]#, 5, 7, 11]
    plt.figure(figsize=((len(fields)+2)*5, 5))
    plt.subplot(1, len(fields)+2, 1)
    plt.plot(np.arange(len(xR)), xR, 'b')
    plt.plot(np.arange(Win), xR[0:Win], 'r')
    plt.title(getFacAmpStr(facs, amps))
    plt.subplot(1, len(fields)+2, 2)
    #plt.imshow(D, interpolation = 'nearest', cmap = 'afmhot')
    plt.plot(D[0, :])
    for k in range(len(mins)):
        x = mins[k]
        plt.plot([0, D.shape[1]], [x, x], color = colors[k%len(colors)])
    plt.xlim([0, D.shape[1]])
    plt.ylim([0, ymax])
    plt.title("SSM Row 0")
    for i in range(len(fields)):
        f = fields[i]
        print "Doing field %i"%f
        plt.subplot(1, len(fields)+2, 3+i)
        PDs = doRipsFiltration(YR, 1, coeff=f, relpath="../")
        AllPDs.append(PDs[1])
        for k in range(len(mins)):
            x = mins[k]
            plt.plot([0, ymax], [x, x], color = colors[k%len(colors)])
            plt.plot([x, x], [0, ymax], color = colors[k%len(colors)])
        plotDGM(PDs[1], color = 'b')
        plt.xlim([0, ymax])
        plt.ylim([0, ymax])
        plt.title("$\mathbb{Z}%i$"%f)
    plt.savefig("%s.svg"%getFacStr(facs), bbox_inches = 'tight')

    threshs = getThresholdsFromPDs(AllPDs, True)
    print "threshs = ", threshs
    threshs = massageThresholds(threshs)
    CircCoords = getCircularCoordsFromThreshs(D, threshs)
    plt.figure(figsize=(12, 3*len(threshs)))
    for i in range(len(threshs)):
        plt.subplot(len(threshs), 1, i+1)
        theta = CircCoords[i]['thetau']
        estT = (2*np.pi) / (getAverageSlope(theta))
        plt.plot(CircCoords[i]['v'][:, 1:3])
        plt.title("Thresh = %.3g, Estimated T = %g"%(threshs[i], estT))
    plt.savefig("%sCoords.svg"%getFacStr(facs), bbox_inches = 'tight')

    #Make animation
    if not doAnimation:
        exit(0)
    plt.figure(figsize=(12, 6))
    idx = 0
    for i in range(T+1):
        plt.clf()
        plt.subplot(211)
        plt.plot(YR[0, :], 'b')
        plt.title('Window 0 vs Window %i, Distance = %.3g'%(i, D[0, i]))
        plt.plot(YR[i, :], 'r')
        plt.subplot(212)
        plt.plot(D[0, 0:T+1])
        plt.xlim([0, T+1])
        plt.title("Distance")
        plt.xlabel("Lag")
        count = 1
        #Pause for a second with local mins
        if i > 0 and i < T-1:
            if D[0, i] < D[0, i-1] and D[0, i] < D[0, i+1]:
                count = NPerPeriod/4
        if count == 1:
            plt.stem([i, i], [0, D[0, i]], 'k')
        else:
            plt.stem([i, i], [0, D[0, i]], 'r', lineWidth=4)
        for c in range(count):
            plt.savefig("%s%i.png"%(PREFIX, idx), bbox_inches = 'tight')
            idx += 1

    #Make movie with avconv
    s = getFacStr(facs)
    filename = "%s.avi"%s
    if os.path.exists(filename):
        os.remove(filename)
    rate = NPerPeriod / 2
    subprocess.call(["avconv", "-r", "%i"%rate, "-i", "%s%s.png"%(PREFIX, "%d"), "-r", "%i"%rate, "-b", "30000k", filename])
    for f in glob.glob("%s*.png"%PREFIX):
        os.remove(f)
