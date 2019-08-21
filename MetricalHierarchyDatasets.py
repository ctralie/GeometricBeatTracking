from SoundTools import *
import sys
import matplotlib.pyplot as plt
import pickle
from multiprocessing import Pool
import os
import json

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

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
            # TODO: Finish this