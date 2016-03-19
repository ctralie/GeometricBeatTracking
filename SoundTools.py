import numpy as np
import numpy.linalg as linalg
import scipy.io as sio
from scipy import sparse
import matplotlib.pyplot as plt
import subprocess
import os
import time

import librosa

import essentia
from essentia import Pool, array
from essentia.standard import *



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
    
    ######################################################
    ##               Novelty Functions                  ##
    ######################################################
    def getMFCCNoveltyFn(self, winSize, hopSize, pfmax):
        self.processSpecgram(winSize, hopSize, pfmax)
        diff = self.X[:, 1:] - self.X[:, 0:-1]
        diff[diff < 0] = 0
        self.novFn = np.sum(diff, 0).flatten()
        self.origNovFn = np.array(self.novFn)
    
    def getLibrosaNoveltyFn(self, winSize, hopSize):
        [self.winSize, self.hopSize] = [winSize, hopSize]
        self.novFn = librosa.onset.onset_strength(y = self.XAudio, sr = self.Fs, hop_length = hopSize)
        self.origNovFn = np.array(self.novFn)
    
    def getWeightedPhaseNoveltyFn(self, winSize, hopSize):
        self.S = librosa.core.stft(self.XAudio, winSize, hopSize)
        (mag, phase) = librosa.magphase(self.S)
        insfreq = phase[:, 1:] - phase[:, 0:-1]
        insfreqp = insfreq[:, 1:] - insfreq[:, 0:-1]
        WP = mag[:, 0:insfreqp.shape[1]]*insfreqp
        self.novFn = np.mean(np.abs(WP), 0)
        self.novFn = self.novFn/np.max(self.novFn)
        self.origNovFn = np.array(self.novFn)
    
    def getEssentiaNoveltyFn(self, hopSize, m):
        #Options: hfc, complex, complex_phase, flux, melflux, rms
        od = OnsetDetection(method=m)
        w = Windowing(type='hann')
        fft = FFT()
        c2p = CartesianToPolar()
        X = essentia.array(self.XAudio)
        self.novFn = []
        for frame in FrameGenerator(X, frameSize = 1024, hopSize = hopSize):
            mag, phase, = c2p(fft(w(frame)))
            self.novFn.append(od(mag, phase))
        self.novFn = self.novFn/np.max(self.novFn)
        self.origNovFn = np.array(self.novFn)
    
    
    ######################################################
    ##           External  Onset Functions              ##
    ######################################################
    #Other implementations that I'm comparing to
    
    #Call librosa to get the dynamic programming onsets for comparison
    def getEllisLibrosaOnsets(self):
        (tempo, beats) = librosa.beat.beat_track(self.XAudio, self.Fs, hop_length = self.hopSize)
        return beats
    
    #Call Essentia's implementation of Degara's technique
    def getDegaraOnsets(self):
        X = essentia.array(self.XAudio)
        b = BeatTrackerDegara()
        beats = np.array(np.round(b(X)*self.Fs/self.hopSize), dtype=np.int64)
        return beats
    
    #Call the multi feature beat tracker in Essentia
    def getMultiFeatureOnsets(self):
        X = essentia.array(self.XAudio)
        b = BeatTrackerMultiFeature()
        beats = np.array(np.round(b(X)[0]*self.Fs/self.hopSize), dtype=np.int64)
        return beats        
    
    def getSampleDelay(self, i):
        if i == -1:
            i = self.Y.shape[0]-1
        return float(i)*self.hopSize/self.Fs


    ######################################################
    ##               Exporting Functions                ##
    ######################################################
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
        print "Fs = ", self.Fs
        YAudio = np.array(self.XAudio)
        blipsamples = int(np.round(0.02*self.Fs))
        blip = np.cos(2*np.pi*np.arange(blipsamples)*440.0/self.Fs)
        blip = np.array(blip*np.max(np.abs(YAudio)), dtype=YAudio.dtype)
        for idx in onsets:
            l = len(YAudio[idx*self.hopSize:idx*self.hopSize+blipsamples])
            YAudio[idx*self.hopSize:idx*self.hopSize+blipsamples] = blip[0:l]
        sio.wavfile.write("temp.wav", self.Fs, YAudio)
        if os.path.exists(outname):
            os.remove(outname)
        subprocess.call(["avconv", "-i", "temp.wav", outname])
    
    def plotOnsets(self, onsets, theta):
        #Plot a stem plot of the onsets on x-axis, and corresponding
        #circular coordinates on y-axis
        plt.clf()
        onsetsSec = onsets*float(self.hopSize)/self.Fs
        vals = theta[onsets]
        plt.stem(onsetsSec, vals)
    
    ######################################################
    ##       Sliding Window PCA/SVD Functions           ##
    ######################################################
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
        tic = time.time()
        for i in range(W):
            #Sum together outer products
            UTi = UT[:, i].flatten()
            x = self.novFn[i:i+M]
            self.V += UTi[:, None].dot(x[None, :])
        self.V = (1/self.S[0:NPCs, None])*self.V
        toc = time.time()
        print "Elapsed time right SVD: ", toc-tic
        return self.V
    
    def getSlidingWindowFull(self, W):
        #Return the mean-centered sliding window
        N = len(self.novFn)
        M = N-W+1
        X = np.zeros((W, M))
        for i in range(W):
            X[i, :] = self.novFn[i:i+M]
        X = X - np.mean(X, 1, keepdims = True)
        return X
    
    ######################################################
    ##               Novelty Smoothing                  ##
    ######################################################
    #Apply Gaussian smoothing to the novelty function
    def smoothNovFn(self, W):
        t = np.linspace(-1, 1, W)
        g = np.exp(-t**2/0.25)
        g = g/np.sum(g)
        self.novFn = np.convolve(self.origNovFn, g, 'same')
    
    def lowpassNovFn(self, W):
        #Do an ideal lowpass filter on blocks of the novelty function
        f = np.fft.fft(self.origNovFn)
        N = len(f)
        fidx = int(np.round((1.0/W)*N))
        f[fidx:-fidx] = 0
        self.novFn = np.abs(np.fft.ifft(f))
        
    
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
