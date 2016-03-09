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
        self.origNovFn = np.array([]) #Original novelty function before smoothing/denoising
        
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
        arg = self.M.dot(np.abs(self.S))
        self.X = librosa.core.logamplitude(arg)
    
    def getMFCCNoveltyFn(self, winSize, hopSize, pfmax):
        self.processSpecgram(winSize, hopSize, pfmax)
        diff = self.X[:, 1:] - self.X[:, 0:-1]
        diff[diff < 0] = 0
        self.novFn = np.sum(diff, 0).flatten()
        self.origNovFn = np.array(self.novFn)
    
    #Call librosa to get the dynamic programming onsets for comparison
    def getDynamicProgOnsets(self):
        (tempo, beats) = librosa.beat.beat_track(self.XAudio, self.Fs, hop_length = self.hopSize)
        return beats
    
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
    
    def exportToCircCoordinatesGUI(self, theta, outprefix):
        fout = open("%s.txt"%outprefix, "w")
        for i in range(len(theta)):
            fout.write("%g,%g,"%(theta[i], i*float(self.hopSize)/self.Fs))
        fout.close()
        #Output audio information
        sio.wavfile.write("%s.wav"%outprefix, self.Fs, self.XAudio)        
    
    def exportOnsetClicks(self, outname, onsets):
        YAudio = np.array(self.XAudio)
        blip = np.cos(2*np.pi*np.arange(self.hopSize*4)*440.0/self.Fs)
        blip = np.array(blip*np.max(np.abs(YAudio)), dtype=YAudio.dtype)
        for idx in onsets:
            l = len(YAudio[idx*self.hopSize:(idx+4)*self.hopSize])
            YAudio[idx*self.hopSize:(idx+4)*self.hopSize] = blip[0:l]
        sio.wavfile.write(outname, self.Fs, YAudio)
    
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
    
    #Apply Gaussian smoothing to the novelty function
    def smoothNovFn(self, W):
        t = np.linspace(-1, 1, W)
        g = np.exp(-t**2/0.25)
        g = g/np.sum(g)
        self.novFn = np.convolve(self.origNovFn, g, 'same')
    
    #Find the nearest valid delay embedding only using certain principal components
    #idxs is the indices of the principal components to use
    def slidingWindowDenoising(self, idxs):
        N = len(self.novFn)
        M = self.V.shape[1]
        NPCs = self.V.shape[0]
        U = self.Y[:, idxs]
        W = U.shape[0]
        V = self.V
        S = np.eye(U.shape[1])
        np.fill_diagonal(S, self.S[idxs])
        res = np.zeros(N)
        counts = np.zeros(N)
        for i in range(W):
            Window = (U[i, :].dot(S)).dot(V[idxs, :])
            res[i:i+M] += Window/np.sqrt(np.sum(Window**2))
            counts[i:i+M] += 1.0
        return res/counts
        
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
