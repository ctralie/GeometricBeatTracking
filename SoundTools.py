import numpy as np
import numpy.linalg as linalg
import scipy.io as sio
from scipy import sparse
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import subprocess
import os
import time

import librosa

import essentia
from essentia import Pool, array
from essentia.standard import *

import sys
sys.path.append("SuperFlux")
import SuperFlux

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
            if os.path.exists("temp.wav"):
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
    
    def getSuperfluxNoveltyFn(self, winSize, hopSize):
        w = SuperFlux.Wav(self.filename)
#        args = SuperFlux.parser()
#        # normalize audio
#        if args.norm:
#            w.normalize()
#            args.online = False  # switch to offline mode
        # down-mix to mono
        if w.num_channels > 1:
            w.downmix()
#        # attenuate signal
#        if args.att:
#            w.attenuate(args.att)
        # create filterbank if needed
        # re-create filterbank if the sample rate of the audio changes
        if filt is None or filt.fs != w.sample_rate:
            filt = SuperFlux.Filter(args.frame_size / 2, w.sample_rate,
                          args.bands, args.fmin, args.fmax, args.equal)
            filterbank = filt.filterbank
        # spectrogram
        s = SuperFlux.Spectrogram(w, frame_size=args.frame_size, fps=args.fps,
                        filterbank=filterbank, log=args.log,
                        mul=args.mul, add=args.add, online=args.online,
                        block_size=args.block_size, lgd=args.lgd)
        # use the spectrogram to create an SpectralODF object
        sodf = SuperFlux.SpectralODF(s, ratio=args.ratio, max_bins=args.max_bins,
                           diff_frames=args.diff_frames)
        act = sodf.superflux()
        s.novFn = act
    
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

    ######################################################
    ##         Other Signal Processing Stuff            ##
    ######################################################
    
    #For comparison with the Quinton paper
    #TODO: Finish this
    def getDFTACFHadamardBlocks(self, BlockLen, BlockHop):
        X = self.novFn
        N = len(X)
        #Exclude last block if it doesn't include a full BlockLen
        #samples so that all frames are the same length
        NBlocks = int(np.floor(1 + (N - BlockLen)/BlockHop))
        res = np.zeros((NBlocks, BlockLen))
        for i in range(NBlocks):
            thisidxs = np.arange(i*BlockHop, i*BlockHop+BlockLen, dtype=np.int64)
            x = X[thisidxs]
            n = len(x)
            #DFT
            dft = np.abs(np.fft.fft(x))
            #Autocorrelation
            acf = np.correlate(x, x, 'full')[len(dft)-1:]
            dft = dft[1:]
            acf = acf[1:]
            idx = len(x)/np.arange(1, len(x), dtype=np.float64)
            idx = idx[::-1]
            acf = acf[::-1]
            idxx = np.arange(1, len(x), dtype=np.float64)
            acfwarp = interp.spline(idx, acf, idxx)
            plt.subplot(411)
            plt.plot(x)
            plt.subplot(412)
            plt.plot(dft)
            plt.subplot(413)
            plt.plot(acfwarp)
            plt.subplot(414)
            plt.plot(dft*acfwarp)
            plt.show()
        print "TODO"

#Save a file which auralizes a number of tempos
#tempos: an array of tempos in beats per minute
#Fs: Sample rate, NSeconds: Number of seconds to go
#filename: Output filename (must be a .wav file)
def makeMetronomeSound(tempos, Fs, NSeconds, filename):
    blipsamples = int(np.round(0.02*Fs))
    blip = np.cos(2*np.pi*np.arange(blipsamples)*440.0/Fs)
    #blip = np.array(blip*np.max(np.abs(YAudio)), dtype=YAudio.dtype)
    X = np.zeros(int(round(NSeconds*Fs)))
    for tempo in tempos:
        #tempos are in beats per minute, convert to samples per beat
        T = 1.0/(tempo/(60.0*Fs))
        print "T = ", T
        i = 0
        while True:
            i1 = int(i*T)
            i2 = i1 + blipsamples
            if i2 >= len(X):
                break
            X[i1:i2] += blip
            i += 1
    X = X/np.max(X)
    sio.wavfile.write(filename, Fs, X)  

if __name__ == '__main__2':
    s = BeatingSound()
    s.loadAudio("examples1/train1.wav")
    #s.getMFCCNoveltyFn(2048, 128, 8000)
    #s.getDFTACFHadamardBlocks(600, 100)
    s.getSuperfluxNoveltyFn(2048, 128)
    plt.plot(s.novFn)
    plt.show()

if __name__ == '__main__':
    s = BeatingSound()
    s.loadAudio("examples1/train4.wav")
    s.getMFCCNoveltyFn(2048, 256, 8000)
    s.exportToFnViewer("train4.mat")
