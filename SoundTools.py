import librosa
import numpy as np
import numpy.linalg as linalg
import scipy.io as sio
from scipy import sparse
import matplotlib.pyplot as plt
import subprocess
import os
import time

BATCHSIZE = 50 #Batch size for computing left hand singular vectors, tradeoff between memory and speed

class BeatingSound(object):
    def __init__(self):
        self.filename = ""
        self.XAudio = np.array([])
        self.Fs = 22050
        self.winSize = 1024
        self.hopSize = 512
        self.fmax = 8000
        self.S = np.array([[]]) #Spectrogram
        self.M = np.array([[]]) #Mel filterbank
        self.X = np.array([[]]) #Mel spectrogram
        self.novFn = np.array([]) #Novelty function
        
    def loadAudio(self, filename):
        self.filename = filename
        fileparts = filename.split(".")
        if not fileparts[-1] == "wav":
            os.remove("temp.wav")
            subprocess.call(["avconv", "-i", filename, "temp.wav"])
            #self.XAudio, self.Fs = librosa.load("temp.wav")
            self.Fs, self.XAudio = sio.wavfile.read("temp.wav")
            self.XAudio = self.XAudio.T
        else:
            #self.XAudio, self.Fs = librosa.load(filename)
            self.Fs, self.XAudio = sio.wavfile.read(filename)
            self.XAudio = self.XAudio.T
            print self.XAudio.shape
        print "Fs = %i"%self.Fs
        self.XAudio = librosa.core.to_mono(self.XAudio)

    def processSpecgram(self, winSize, hopSize, pfmax):
        [self.winSize, self.hopSize, self.fmax] = [winSize, hopSize, pfmax]
        self.S = librosa.core.stft(self.XAudio, winSize, hopSize)
        self.M = librosa.filters.mel(self.Fs, winSize, fmax = pfmax)
        self.X = 20*np.log(self.M.dot(np.abs(self.S)))
    
    def getMFCCNoveltyFn(self, winSize, hopSize, pfmax):
        self.processSpecgram(winSize, hopSize, pfmax)
        self.novFn = np.sum(np.abs(self.X[:, 1:] - self.X[:, 0:-1]), 0).flatten()
    
    def getSampleDelay(self, i):
        if i == -1:
            i = self.Y.shape[0]-1
        return float(i)*self.hopSize/self.Fs

    def exportToFnViewer(self, filename, X = None):
        if not X:
            #By default, export novelty function
            Fn = self.novFn
        SampleDelays = (float(self.hopSize)/self.Fs)*np.arange(Fn.size)
        X = np.zeros((Fn.size, 2))
        X[:, 0] = np.arange(X.shape[0])
        X[:, 1] = Fn
        sio.savemat(filename, {"soundfilename":self.filename, "SampleDelays":SampleDelays, "Fs":self.Fs, "X":X})

    #Export to web interface for synchronized visualization
    #This function expects that SlidingWindowRightSVD has been computed
    def exportToLoopDitty(self, outprefix):
        #Output information text file
        fout = open("%s.txt"%outprefix, "w")
        for i in range(self.V.shape[1]):
            fout.write("%g,%g,%g,%g,"%(self.V[0, i], self.V[1, i], self.V[2, i], i*float(self.hopSize)/self.Fs))
        fout.write("%g"%(np.sum(self.S[0:3])/np.sum(self.S)))
        fout.close()
        #Output audio information
        sio.wavfile.write("%s.wav"%outprefix, self.Fs, self.XAudio)
    
    def getSlidingWindowLeftSVD(self, W):
        #Calculate the left hand singular vectors of the sliding window of
        #the novelty function
        N = len(self.novFn)
        M = N-W+1
        #Step 1: Calculate the mean delay window
        self.Mu = np.zeros(W)
        for i in range(W):
            self.Mu[i] = np.mean(self.novFn[i:i+M])
            
        #Calculate the principal components (left hand singular vectors)
        #of the embedding
        start_time = time.time()
        AAT = np.zeros((W, W))
        NBatches = int(np.ceil(W/float(BATCHSIZE)))
        for i in range(NBatches):
            idxsi = range(i*BATCHSIZE, min((i+1)*BATCHSIZE, W))
            xi = np.zeros((len(idxsi), M))
            for k in range(len(idxsi)):
                xi[k, :] = self.novFn[idxsi[k]:idxsi[k]+M] - self.Mu[idxsi[k]]
            for j in range(NBatches):
                idxsj = range(j*BATCHSIZE, min((j+1)*BATCHSIZE, W))
                xj = np.zeros((M, len(idxsj)))
                for k in range(len(idxsj)):
                    xj[:, k] = self.novFn[idxsj[k]:idxsj[k]+M] - self.Mu[idxsj[k]]
                AAT[idxsi[0]:idxsi[-1]+1, idxsj[0]:idxsj[-1]+1] = xi.dot(xj)
        end_time = time.time()
        print "Elapsed time mutliplication: ", end_time - start_time
        [S, Y] = linalg.eigh(AAT)
        idx = np.argsort(-S)
        S[S < 0] = 0 #Numerical precision
        self.S = np.sqrt(S[idx])
        self.Y = Y[:, idx]        
        return (self.Y, self.S)
    
    def getSlidingWindowRightSVD(self, W, NPCs):
        #Calculate "NPCs" scaled right hand singular vectors (i.e. "principal coordinates")
        N = len(self.novFn)
        M = N-W+1
        self.V = np.zeros((NPCs, M))
        UT = self.Y.T[0:NPCs, :]
        for i in range(W):
            #Sum together outer products
            UTi = UT[:, i].flatten()
            x = self.novFn[i:i+M]
            self.V += UTi[:, None].dot(x[None, :])
        return self.V
        
if __name__ == '__main__':
    s = BeatingSound()
    s.loadAudio("examples1/train1.wav")
    s.getMFCCNoveltyFn(2048, 128, 8000)
    s.exportToFnViewer("train1.mat")
    W = 200
    (Y, S) = s.getSlidingWindowLeftSVD(W)
    V = s.getSlidingWindowRightSVD(W, 3)
    plt.subplot(121)
    plt.imshow(Y[:, 0:40], interpolation = 'none', aspect = 'auto')
    plt.subplot(122)
    plt.plot(V[0, :], V[1, :])
    plt.show()
    s.exportToLoopDitty("train1")
