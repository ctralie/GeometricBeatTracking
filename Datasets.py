from CircularCoordinates import *
from DynProgOnsets import *
from SoundTools import *
import sys
sys.path.append("Beat-Tracking-Evaluation-Toolbox")
import beat_evaluation_toolbox as be
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
import os

DO_SKIP = True #Skip files that have already been computed

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
        a = np.array([float(l.split()[0]) for l in lines]).flatten()
        annotations.append(a)
    audioFiles = ["Datasets/BallroomData/%s"%af for af in audioFiles]
    return (audioFiles, annotations)

def getSMCData():
    NFILES = 281
    audiopath = "Datasets/SMC_MIREX/SMC_MIREX_Audio"
    annopath = "Datasets/SMC_MIREX/SMC_MIREX_Annotations_05_08_2014"
    annofiles = os.listdir(annopath)
    songsDict = {}
    for f in annofiles:
        fields = f.split("_")
        idx = int(fields[1])
        audiofilename = audiopath + "/SMC_" + fields[1] + ".wav"
        fin = open(annopath + "/" + f)
        annotations = np.array([float(s) for s in fin.readlines()]).flatten()
        fin.close()
        songsDict[idx] = (audiofilename, annotations)
    audioFiles = []
    annotations = []
    for i in songsDict:
        (af, an) = songsDict[i]
        audioFiles.append(af)
        annotations.append(an)
    return (audioFiles, annotations)
        

def exportGTClicks(audioFiles, annotations):
    hopSize = 128
    for i in range(len(audioFiles)):
        print "Exporting GT Clicks for %s"%audioFiles[i]
        s = BeatingSound()
        s.loadAudio(audioFiles[i])
        s.hopSize = hopSize
        a = np.array(annotations[i])
        a = np.array(np.round(a*s.Fs/s.hopSize), dtype=np.int64)
        s.exportOnsetClicks("%s_GT.ogg"%audioFiles[i], a)

def checkForDirectory(filename):
    #Check to see if this directory exists and create folders if necessary
    paths = filename.split("/")
    for i in range(len(paths)):
        p = paths[0]
        for j in range(1, i):
            p += "/" + paths[j]
        if not os.path.exists(p):
            os.mkdir(p)

def runTests(audioFiles, annotations, resprefix, toskip, parpool):
    hopSize = 128
    winSize = 2*2048
    NPCs = 60
    N = len(audioFiles)
    myonsets = []
    alldponsets = []
    gtannotations = []
    gaussWin = 20
    for i in range(N):
        if i in toskip:
            continue
        gtannotations.append(annotations[i])
        matfile = "%s/%s.mat"%(resprefix, audioFiles[i])
        checkForDirectory(matfile)
        if DO_SKIP and os.path.exists(matfile):
            res = sio.loadmat(matfile)
            myonsets.append(res['onsets'])
            alldponsets.append(res['dponsets'])
            print "Skipping %s"%audioFiles[i]
            continue
        print "Doing %s"%audioFiles[i]
        s = BeatingSound()
        s.loadAudio(audioFiles[i])
        #s.getMFCCNoveltyFn(winSize, hopSize, 8000)
        s.getLibrosaNoveltyFn(winSize, hopSize)
        W = 2*s.Fs/hopSize
        theta = getCircularCoordinatesBlocks(s, W, NPCs, 600, 100, parpool, gaussWin, denoise = True, doPlot = False)
        (onsets, score) = getOnsetsDP(theta, s, 6, 0.4)
        dponsets =  s.getDynamicProgOnsets()
        #Output clicks
        s.exportOnsetClicks("%s/%s_My.ogg"%(resprefix, audioFiles[i]), onsets)
        s.exportOnsetClicks("%s_DP.ogg"%audioFiles[i], dponsets)
        #Convert to seconds
        onsets = onsets*s.hopSize/float(s.Fs)
        dponsets = dponsets*s.hopSize/float(s.Fs)
        sio.savemat(matfile, {"onsets":onsets, "dponsets":dponsets})
        #Do evaluation
        myonsets.append(onsets)
        alldponsets.append(dponsets)
    for i in range(len(myonsets)):
        myonsets[i] = myonsets[i].flatten()
    MyRes = be.evaluate_db(gtannotations, myonsets, 'all', False)
    DpRes = be.evaluate_db(gtannotations, alldponsets, 'all', False)
    return (MyRes, DpRes)

if __name__ == '__main__':
    parpool = Pool(processes = 8)
    (audioFiles, annotations) = getBallroomData()
    #(audioFiles, annotations) = getSMCData()
    runTests(audioFiles, annotations, "Results/Test1_LibrosaOnset_FixedPCA_Extending", [212], parpool)
