import numpy as np
import matplotlib.pyplot as plt
import subprocess
import sys
sys.path.append("..")
from TDA import *
from TDAPlotting import *
from PulseExperiments import *
from matplotlib.patches import Polygon
import glob

PREFIX = "temp"

def getPos(Ts, idx, spacing):
    T = int(Ts[-1])
    k = Ts[0]/Ts[-1]
    period = idx/T
    ypos = spacing*(idx % T) + 8*spacing*np.sin(2*np.pi*period/(k))
    if idx == Ts[0]:
        period = (idx-1)/T
        ypos = spacing*T + 8*spacing*np.sin(2*np.pi*period/(k))
    xpos = 4*spacing*np.cos(2*np.pi*period/(k))
    return (-xpos, ypos)

if __name__ == '__main__':
    NPerPeriod = 36
    facs = (1, 4)
    amps = (1, 1)
    s = getFacStr(facs)
    gaussSigma = 1

    n0 = lcm(facs)
    f = np.ceil(float(NPerPeriod)/n0)
    T = int(n0*f)
    Ts = [int(T/k) for k in facs]
    N = T*3

    print "facs = %s, Ts = %s"%(facs, Ts)

    xR = getGaussianPulseTrain(N, Ts, amps, 0, gaussSigma)
    Win = T*2

    YR = getSlidingWindowNoInterp(xR, Win)
    c = plt.get_cmap('Spectral')
    C = c(np.array(np.round(np.linspace(0, 255, YR.shape[0])), dtype=np.int32))
    C = C[:, 0:3]
    
    print "T = ", T
    print "YR.shape = ", YR.shape
    D = getSSM(YR)
    dmax = int(np.ceil(np.max(D)*1.1)) 
    print "D[0, -1] = ", D[0, -1]
    
    plt.figure(figsize=(12, 5))
    plt.subplot(122)
    plt.imshow(D, interpolation = 'nearest', cmap = 'afmhot')
    plt.title("Distance Matrix")
    plt.subplot(121)
    plt.plot(xR)
    plt.title("Pulse Train")
    
    plt.savefig("%sFiltration.svg"%s, bbox_inches = 'tight')
    
    
    PDs = doRipsFiltrationDM(D, 1, coeff=2, relpath="../")
    I = PDs[1]
    ts = np.unique(D.flatten())
    plt.figure(figsize=(12, 4))
    np.fill_diagonal(D, np.inf)
    for i in range(len(ts)):
        d = ts[i]
        
        plt.clf()
        plt.subplot(131)
        A = np.zeros(D.shape)
        A[D <= ts[i]] = 1
        plt.imshow(A, cmap = 'gray', interpolation = 'nearest')
        plt.scatter(-1*np.ones(A.shape[0]), np.arange(A.shape[1]), 20, c=C, edgecolor='none')
        plt.scatter(np.arange(A.shape[1]), -1*np.ones(A.shape[0]), 20, c=C, edgecolor='none')
        plt.xlim([-2, A.shape[0]])
        plt.ylim([A.shape[0], -2])
        plt.title("Adjacency Matrix")
        
        plt.subplot(132)
        plotDGM(I)
        plt.plot([0, dmax], [d, d], color = 'k')
        plt.plot([d, d], [0, dmax], color = 'k')
        plt.xlim([0, dmax])
        plt.ylim([0, dmax])
        
        plt.subplot(133)
        spacing = 10
        for i1 in range(D.shape[0]):
            (x1, y1) = getPos(Ts, i1, spacing)
            if i == 0:
                print i1, ": ", x1, ", ", y1
            plt.scatter([x1], [y1], 40, c=C[i1, :])
            for i2 in range(i1+1, D.shape[1]):
                (x2, y2) = getPos(Ts, i2, spacing)
                if D[i1, i2] <= d:
                    plt.plot([x1, x2], [y1, y2], 'k')
        plt.axis('off')
        plt.savefig("%s%i.png"%(PREFIX, i), bbox_inches = 'tight')

    filename = "%sFiltration.avi"%s
    if os.path.exists(filename):
        os.remove(filename)
    rate = 6
    subprocess.call(["avconv", "-r", "%i"%rate, "-i", "%s%s.png"%(PREFIX, "%d"), "-r", "%i"%rate, "-b", "30000k", filename])
    for f in glob.glob("%s*.png"%PREFIX):
        os.remove(f)
