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

    def loadAudio(self, filename, Fs=44100):
        self.filename = filename
        self.XAudio, self.Fs = librosa.load(filename, sr=Fs)
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

    #http://madmom.readthedocs.io/en/latest/modules/features/onsets.html
    def getSuperfluxNoveltyFn(self):
        import madmom
        log_filt_spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(self.filename, num_bands = 24)
        self.novFn = madmom.features.onsets.superflux(log_filt_spec)
        #Madmom default hopsize/window size
        self.hopSize = 441
        self.winSize = 2048

    def getComplexfluxNoveltyFn(self):
        import madmom
        log_filt_spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(self.filename, num_bands = 24)
        self.novFn = madmom.features.onsets.complex_flux(log_filt_spec)
        #Madmom default hopsize/window size
        self.hopSize = 441
        self.winSize = 2048

    def getRNNNoveltyFn(self):
        from madmom.features.beats import RNNBeatProcessor
        self.novFn = RNNBeatProcessor()(self.filename)
        self.hopSize = 441

    ######################################################
    ##           External  Onset Functions              ##
    ######################################################
    #Other implementations that I'm comparing to

    def getEllisLibrosaOnsets(self):
        """
        Call librosa to get the dynamic programming onsets for comparison
        """
        (tempo, beats) = librosa.beat.beat_track(self.XAudio, self.Fs, hop_length = self.hopSize)
        return beats

    def getDegaraOnsets(self, tempo = None):
        """
        Call Essentia's implementation of Degara's technique
        """
        X = essentia.array(self.XAudio)
        if tempo:
            t1 = int(np.round(tempo*0.9))
            t2 = int(np.round(tempo*1.1))
            if t2 - t1 < 20:
                t2 = t1 + 20
            b = BeatTrackerDegara(minTempo = t1, maxTempo = t2)
        else:
            b = BeatTrackerDegara()
        beats = np.array(np.round(b(X)*self.Fs/self.hopSize), dtype=np.int64)
        return beats

    def getMultiFeatureOnsets(self):
        """
        Call the multi feature beat tracker in Essentia
        """
        X = essentia.array(self.XAudio)
        b = BeatTrackerMultiFeature()
        beats = np.array(np.round(b(X)[0]*self.Fs/self.hopSize), dtype=np.int64)
        return beats

    def getMadmomOnsets(self):
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        proc = DBNBeatTrackingProcessor(fps=100)
        act = RNNBeatProcessor()(self.filename)
        b = proc(act)
        beats = np.array(np.round(b*self.Fs/self.hopSize), dtype=np.int64)
        return beats

    def getSampleDelay(self, i):
        if i == -1:
            i = self.Y.shape[0]-1
        return float(i)*self.hopSize/self.Fs

    def getMadmomTempo(self, do_plot=False):
        """
        Call Madmom Tempo Estimation
        :return: Array of tempos sorted in decreasing order of strength
        """
        import time
        from madmom.features.beats import RNNBeatProcessor
        from madmom.features.tempo import TempoEstimationProcessor
        tic = time.time()
        act = RNNBeatProcessor()(self.filename)
        print("Elapsed Time novfn: %.3g"%(time.time()-tic))
        tic = time.time()
        proc = TempoEstimationProcessor(fps=100)
        res = proc(act)
        print("Elapsed time tempo: %.3g"%(time.time()-tic))
        if do_plot:
            plt.stem(res[:, 0], res[:, 1])
            plt.xlabel("bpm")
            plt.ylabel("confidence")
        return res


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
        print("Fs = %i"%self.Fs)
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
        subprocess.call(["ffmpeg", "-i", "temp.wav", outname])

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
    def smoothNovFn(self, W):
        """
        Apply Gaussian smoothing to the novelty function
        """
        t = np.linspace(-1, 1, W)
        g = np.exp(-t**2/0.25)
        g = g/np.sum(g)
        self.novFn = np.convolve(self.origNovFn, g, 'same')

    def lowpassNovFn(self, W):
        """
        Do an ideal lowpass filter on blocks of the novelty function
        """
        f = np.fft.fft(self.origNovFn)
        N = len(f)
        fidx = int(np.round((1.0/W)*N))
        f[fidx:-fidx] = 0
        self.novFn = np.abs(np.fft.ifft(f))

    ######################################################
    ##         Other Signal Processing Stuff            ##
    ######################################################

def makeMetronomeSound(tempos, Fs, NSeconds, filename):
    """
    Save a file which auralizes a number of tempos
    :param tempos: an array of tempos in beats per minute
    :param Fs: Sample rate, NSeconds: Number of seconds to go
    :param filename: Output filename (must be a .wav file)
    """
    blipsamples = int(np.round(0.02*Fs))
    blip = np.cos(2*np.pi*np.arange(blipsamples)*440.0/Fs)
    #blip = np.array(blip*np.max(np.abs(YAudio)), dtype=YAudio.dtype)
    X = np.zeros(int(round(NSeconds*Fs)))
    for tempo in tempos:
        #tempos are in beats per minute, convert to samples per beat
        T = 1.0/(tempo/(60.0*Fs))
        print("T = %.3g"%T)
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
    makeMetronomeSound([83.5], 44100, 10, "train2met.wav")

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
    s.loadAudio("Datasets/GTzan/hiphop/hiphop.00018.au")
    s.getMadmomTempo(do_plot=True)
    plt.show()

if __name__ == '__main__2':
    from DynProgOnsets import *
    s = BeatingSound()
    s.loadAudio("examples1/train4.wav")
    #s.loadAudio("sting.wav")
    s.getSuperfluxNoveltyFn()
    T = 85
    onsetsDeg = s.getDegaraOnsets(T)
    onsetsBock = s.getMadmomOnsets()
    print(onsetsBock)
    s.exportOnsetClicks("Erykah.wav", onsetsBock)
    slope = 2*np.pi*s.hopSize*T/(60*s.Fs)
    theta = np.arange(len(s.novFn))*slope
    (onsetsDP, score) = getOnsetsDP(theta, s, 6, alpha = 0.8)
    #s.exportOnsetClicks("StingNovFn.wav", onsetsDeg)
