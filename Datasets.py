from CircularCoordinates import *
from DynProgOnsets import *
from SoundTools import *
import sys
sys.path.append("Beat-Tracking-Evaluation-Toolbox")
import beat_evaluation_toolbox as be
import matplotlib.pyplot as plt
import pickle

def getBallroomData():
    audiopath = "Datasets/BallroomData"
    fin = open("%s/allBallroomFiles"%audiopath)
    audioFiles = fin.readlines()
    audioFiles = [f[2:].rstrip() for f in audioFiles]
    annotations = []
    fin.close()
    for i in range(len(audioFiles)):
        af = audioFiles[i]
        fields = af.split("/")
        annFile = "Datasets/BallroomAnnotations/%s.beats"%(fields[1][0:-4])
        fin = open(annFile)
        lines = fin.readlines()
        fin.close()
        a = [float(l.split()[0]) for l in lines]
        annotations.append(a)
    audioFiles = ["Datasets/BallroomData/%s"%af for af in audioFiles]
    return (audioFiles, annotations)

def runTests(audioFiles, annotations):
    hopSize = 128
    winSize = 2*2048
    NPCs = 20
    N = len(audioFiles)
    myonsets = []
    alldponsets = []
    gaussWin = 20
    for i in range(N):
        print "Doing %s"%audioFiles[i]
        s = BeatingSound()
        s.loadAudio(audioFiles[i])
        s.getMFCCNoveltyFn(winSize, hopSize, 8000)
        W = 2*s.Fs/hopSize
        theta = getCircularCoordinatesBlocks(s, W, NPCs, 600, 100, gaussWin, denoise = True, doPlot = False)
        (onsets, score) = getOnsetsDP(theta, s, 6, 0.4)
        dponsets =  s.getDynamicProgOnsets()
        #Output clicks
        s.exportOnsetClicks("%s_My.ogg"%audioFiles[i], onsets)
        s.exportOnsetClicks("%s_DP.ogg"%audioFiles[i], dponsets)
        #Convert to seconds
        onsets = onsets*s.hopSize/float(s.Fs)
        dponsets = dponsets*s.hopSize/float(s.Fs)
        sio.savemat("%s.mat"%audioFiles[i], {"onsets":onsets, "dponsets":dponsets})
        #Do evaluation
        myonsets.append(onsets)
        alldponsets.append(dponsets)
        print onsets
    MyRes = be.evaluate_db(annotations, myonsets, 'all', False)
    DpRes = be.evaluate_db(annotations, alldponsets, 'all', False)
    pickle.dump(open("MyRes.txt", "w"), {"MyRes":MyRes})
    pickle.dump(open("DpRes.txt", "w"), {"DpRes":DpRes})

if __name__ == '__main__':
    (audioFiles, annotations) = getBallroomData()
    runTests(audioFiles, annotations)
