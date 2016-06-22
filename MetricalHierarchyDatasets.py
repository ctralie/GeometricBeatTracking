from CircularCoordinates import *
from TDATools import *
from SoundTools import *
import sys
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
import os
import json

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def getRhythmHierarchy(filename, cutoff = 3.0):
    hopSize = 128
    winSize = 2*2048
    s = BeatingSound()
    s.loadAudio(filename)
    s.getLibrosaNoveltyFn(winSize, hopSize)
    
    W = 690
    BlockLen = W
    BlockHop = W/6
    Kappas = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25]
    AllResults = getCircularCoordinatesBlocks(s, W, BlockLen, BlockHop, True, Kappas)
    tempoScores = aggregateTempoScores(s, AllResults)
    #Filter from top to bottom
    I = getMorseFiltration(-tempoScores)
    tempos = I[I[:, 1] - I[:, 0] > cutoff, 2]
    return (tempos, tempoScores)

def evalPerformance(tempos, groundTruth, eps = 0.15):
    #By default just compare to the first annotation
    gt = groundTruth[0]['metrical_levels_pulse_rates']
    tP = 0
    for i in range(len(gt)):
        t = gt[i]
        if not t: #Strange bug on blues 32
            gt[i] = -1
            continue
        diff = np.exp(np.abs(np.log(tempos) - np.log(t))) - 1
        #At least one of the detected tempos has to be within
        #an epsilon factor of a ground truth tempo
        if np.sum(diff <= eps) > 0:
            tP += 1
    precision = 0
    if len(tempos) > 0:
        precision = float(tP)/len(tempos)
    recall = 0
    if len(gt) > 0:
        recall = float(tP)/len(gt)
    return (precision, recall, gt)
    

if __name__ == '__main__':
    SkipFiles = True
    with open('Datasets/annotations_metricalStructure_GTZAN/annotations_MetricalStructure_GTZAN.json') as data_file:
        data = json.load(data_file)
    
    genreDict = {}
    for i in range(len(data)):
        genreDict[str(data[i]['trackName'])] = data[i]['data']
    
    totalPrecision = 0.0
    totalRecall = 0.0
    
    count = 0
    for genre in GENRES:
        for i in range(100):
            count += 1
            soundfile = "Datasets/GTzan/%s/%s.%.5i.au"%(genre, genre, i)
            outfile = "Datasets/GTzan/%s/%s.%.5i.mat"%(genre, genre, i)
            if SkipFiles and os.path.exists(outfile):
                res = sio.loadmat(outfile)
                totalPrecision += res['precision']
                totalRecall += res['recall']
                print "Skipping %s %i"%(genre, i)
                continue
            print "Doing %s %i..."%(genre, i)
            (tempos, tempoScores) = getRhythmHierarchy(soundfile)
            groundTruth = genreDict["%s.%.5i.wav"%(genre, i)]
            (precision, recall, gt) = evalPerformance(tempos, groundTruth)
            print "Precision = %g, Recall = %g"%(precision, recall)
            totalPrecision += precision
            totalRecall += recall
            print "Running Precision = ", totalPrecision/float(count)
            print "Running Recall = ", totalRecall/float(count)
            sio.savemat(outfile, {"tempos":tempos, "tempoScores":tempoScores, "gt":gt, "precision":precision, "recall":recall})
