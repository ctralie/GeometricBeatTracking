#Purpose: To visualize the circular coordinate slopes and their scores
import numpy as np
import scipy.io as sio
import scipy.io.wavfile
import sys
#sys.path.append("..")
from CircularCoordinates import *
from SyntheticFunctions import *
from SoundTools import *
import time

def plotRes(res):
    plt.clf()
    
    plt.subplot(231)
    plt.imshow(res['D'])
    plt.title('SSM')
    
    plt.subplot(232)
    plt.spy(res['A'])
    plt.title("Adjacency Matrix")
    
    plt.subplot(235)
    timeScale = (float(s.Fs)/s.hopSize)*60.0/(2*np.pi)
    plt.plot(res['slopes']*timeScale)
    plt.title('Slopes')
    
    v = res['v'][:, [1, 2]]
    v = v - np.mean(v, 0, keepdims=True)
    #RMS Normalize
    v = v*np.sqrt(v.shape[0]/np.sum(v**2))
    plt.subplot(233)
    plt.plot(v[:, 0], v[:, 1], 'b.')
    plt.hold(True)
    t = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(t), np.sin(t), 'r')
    plt.title('RMS Circle Fitting: Score = %g'%res['score'])
    
    plt.subplot(236)
    thetas = res['thetas']
    plt.plot(thetas % (2*np.pi))
    #plt.plot(np.cumsum(res['slopes']))
    
    plt.subplot(234)
    plt.plot(thetas)
    plt.title("Theta Unwrapped")
        

if __name__ == '__main__':
    PLOTBLOCKS = False
    hopSize = 128
    winSize = 2*2048
    gaussWin = 20
    
    #filename = "examples1/train2.wav"
    
    #Difficult waltz examples
    #filename = "Datasets/BallroomData/Waltz/Media-105102.wav"
    #filename = 'MissionImpossible.wav'
    #filename = 'trillingandsome.wav'
    
    #GTzan examples
    filename = "Datasets/GTzan/blues/blues.00013.au"
    
    s = BeatingSound()
    s.loadAudio(filename)
    s.getLibrosaNoveltyFn(winSize, hopSize)
    
    X = s.novFn
    plt.plot(np.arange(len(X))*float(s.hopSize)/s.Fs, X)
    plt.xlabel('Seconds')
    plt.title("Novelty Function")
    plt.show()
    
    W = 690
    print "W = ", W
    
    BlockLen = W
    BlockHop = W/6
    Kappas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
    AllResults = getCircularCoordinatesBlocks(s, W, BlockLen, BlockHop, True, Kappas)
    
    if PLOTBLOCKS:
        for i in range(len(AllResults)):
            Results = AllResults[i]
            for j in range(len(Results)):
                plt.clf()
                plotResults(Results[j])
                plt.savefig("%i_%i_%g.png"%(i, j, Kappas[i]))
        plt.clf()
    
    tempos = aggregateTempoScores(s, AllResults)
    sio.savemat("tempos.mat", {"tempos":tempos})
    plt.plot(tempos)
    plt.title('Tempos')
    plt.show()
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    plt.hold(True)
    ax = plt.subplot(111)
    cutoff = 0
    for i in range(len(AllResults)):
        (TemposArr, ScoresArr) = getInstantaneousTempoArray(s, [AllResults[i]], BlockLen, BlockHop)
        [Y, X] = np.meshgrid(np.arange(TemposArr.shape[1]), np.arange(TemposArr.shape[0]))
        X = X.flatten()
        TemposArr = TemposArr.flatten()
        ScoresArr = ScoresArr.flatten()
        X = X[ScoresArr > cutoff]
        TemposArr = TemposArr[ScoresArr > cutoff]
        ScoresArr = ScoresArr[ScoresArr > cutoff]
        plt.scatter(X*float(s.hopSize)/s.Fs, TemposArr, 30*ScoresArr, colors[i], label = '%g'%Kappas[i], edgecolor='none')
    plt.legend()
    ax.set_ylim([0, 600])
    #ax.set_xlim([0, 30])
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Tempo")
    plt.show()
