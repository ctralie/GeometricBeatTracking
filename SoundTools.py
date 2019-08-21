"""
Purpose
-------
To provide an object oriented "BeatingSound" structure 
for loading audio and computing audio novelty functions and 
onset information using many different libraries

Dependencies
------------
* librosa (pip install librosa)
* madmom (pip install madmom)
* ffmpeg (if saving audio with onset clicks)
"""
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

class BeatingSound(object):
    """
    Attributes
    ----------
    XAudio: ndarray(float)
        Raw audio samples
    Fs: int
        Sample rate (default 44100)
    winSize: int
        Window size of spectrogram (default 1024)
    hopSize: int
        Hop size between spectrogram windows (default 512)
    S: ndarray(winSize, nwins)
        Spectrogram
    fmax: int
        Max frequency for MFCC coefficients (only relevant for getMFCCNoveltyFn)
    M: ndarray(nmfcc, winSize)
        Mel filterbank (only relevant for getMFCCNoveltyFn)
    X: ndarray(nmfcc, nwins)
        Mel frequency spectrum (only relevant for getMFCCNoveltyFn)
    novFn: ndarray(nwins)
        Novelty function
    origNovFn: ndarray(nwins)
        Used to store original novelty function before smoothing, etc
    """
    def __init__(self):
        self.filename = ""
        self.XAudio = np.array([])
        self.Fs = 44100
        self.winSize = 1024
        self.hopSize = 512
        self.fmax = 8000
        self.S = np.array([[]]) 
        self.M = np.array([[]]) 
        self.X = np.array([[]]) 
        self.novFn = np.array([])
        self.origNovFn = np.array([])

    def loadAudio(self, filename, Fs=44100):
        """
        Load in audio from a file
        """
        self.filename = filename
        self.XAudio, self.Fs = librosa.load(filename, sr=Fs)
        self.XAudio = librosa.core.to_mono(self.XAudio)

    ######################################################
    ##            My Novelty Functions                  ##
    ######################################################
    def getMFCCNoveltyFn(self, winSize, hopSize, pfmax):
        """
        Compute a simple novelty function based on signed
        differences between adjacent mel bins, and store
        the result in novFn
        Parameters
        ----------
        winSize: int
            Window size of spectrogram
        hopSize: int
            Hop size between spectrogram windows
        pfmax: int
            Maximum mel frequency
        """
        self.S = librosa.core.stft(self.XAudio, winSize, hopSize)
        self.M = librosa.filters.mel(self.Fs, winSize, fmax = pfmax)
        arg = self.M.dot(np.abs(self.S))
        self.X = librosa.core.logamplitude(arg)
        diff = self.X[:, 1:] - self.X[:, 0:-1]
        diff[diff < 0] = 0
        self.novFn = np.sum(diff, 0).flatten()
        self.origNovFn = np.array(self.novFn)

    def getWeightedPhaseNoveltyFn(self, winSize, hopSize):
        """
        Compute a phase-based audio novelty function, and store
        the result in novFn
        Parameters
        ----------
        winSize: int
            Window size of spectrogram
        hopSize: int
            Hop size between spectrogram windows
        """
        self.S = librosa.core.stft(self.XAudio, winSize, hopSize)
        (mag, phase) = librosa.magphase(self.S)
        insfreq = phase[:, 1:] - phase[:, 0:-1]
        insfreqp = insfreq[:, 1:] - insfreq[:, 0:-1]
        WP = mag[:, 0:insfreqp.shape[1]]*insfreqp
        self.novFn = np.mean(np.abs(WP), 0)
        self.novFn = self.novFn/np.max(self.novFn)
        self.origNovFn = np.array(self.novFn)

    ######################################################
    ##          External Novelty Functions              ##
    ######################################################

    def getLibrosaNoveltyFn(self, winSize, hopSize):
        """
        Wrap around librosa's audio novelty function generator.  Store
        the result in novFn
        Parameters
        ----------
        winSize: int
            Window size of spectrogram
        hopSize: int
            Hop size between spectrogram windows
        """
        [self.winSize, self.hopSize] = [winSize, hopSize]
        self.novFn = librosa.onset.onset_strength(y = self.XAudio, sr = self.Fs, hop_length = hopSize)
        self.origNovFn = np.array(self.novFn)

    def getSuperfluxNoveltyFn(self):
        """
        Wrap around Madmom's superflux novelty function
        http://madmom.readthedocs.io/en/latest/modules/features/onsets.html
        Store the result in novFn, and also change the hopSize to 441 (10ms hop @ 44100)
        and the winSize to 2048, since these are the parameters that Madmom uses
        """
        import madmom
        log_filt_spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(self.filename, num_bands = 24)
        self.novFn = madmom.features.onsets.superflux(log_filt_spec)
        #Madmom default hopsize/window size
        self.hopSize = 441
        self.winSize = 2048

    def getComplexfluxNoveltyFn(self):
        """
        Wrap around Madmom's complex flux novelty function, which may work
        better for bowed instruments
        http://madmom.readthedocs.io/en/latest/modules/features/onsets.html
        Store the result in novFn, and also change the hopSize to 441 (10ms hop @ 44100)
        and the winSize to 2048, since these are the parameters that Madmom uses
        """
        import madmom
        log_filt_spec = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(self.filename, num_bands = 24)
        self.novFn = madmom.features.onsets.complex_flux(log_filt_spec)
        #Madmom default hopsize/window size
        self.hopSize = 441
        self.winSize = 2048

    def getRNNNoveltyFn(self):
        """
        Wrap around Madmom's trained recurrent neural network
        audio novelty function
        http://madmom.readthedocs.io/en/latest/modules/features/onsets.html
        Store the result in novFn, and also change the hopSize to 441 (10ms hop @ 44100)
        """
        from madmom.features.beats import RNNBeatProcessor
        self.novFn = RNNBeatProcessor()(self.filename)
        self.hopSize = 441

    ######################################################
    ##           External  Onset Functions              ##
    ######################################################

    def getEllisLibrosaOnsets(self):
        """
        Call librosa to get the dynamic programming onsets
        Returns
        -------
        beats: ndarray(nbeats)
            An array of beat onsets times, in factors of hopSize samples
        """
        (tempo, beats) = librosa.beat.beat_track(self.XAudio, self.Fs, hop_length = self.hopSize)
        return beats

    def getMadmomOnsets(self):
        """
        Call madmom to get the onsets based on a dynamic bayes network on
        top of an audio novelty function from a recurrent neural network
        Returns
        -------
        beats: ndarray(nbeats)
            An array of beat onsets times, in factors of hopSize samples
        """
        from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
        proc = DBNBeatTrackingProcessor(fps=100)
        act = RNNBeatProcessor()(self.filename)
        b = proc(act)
        beats = np.array(np.round(b*self.Fs/self.hopSize), dtype=np.int64)
        return beats

    ######################################################
    ##           External  Tempo Functions              ##
    ######################################################

    def getMadmomTempo(self, do_plot=False):
        """
        Call madmom tempo estimation, which is generally quite accurate
        Parameters
        ----------
        do_plot: boolean
            Whether to plot a stem plot showing the locations
            of the estimated tempos, as well as their strengths
        Returns
        -------
        tempos: ndarray(float)
            Array of tempos in beats per minute, 
            sorted in decreasing order of strength
        """
        import time
        from madmom.features.beats import RNNBeatProcessor
        from madmom.features.tempo import TempoEstimationProcessor
        tic = time.time()
        act = RNNBeatProcessor()(self.filename)
        print("Elapsed Time novfn: %.3g"%(time.time()-tic))
        tic = time.time()
        proc = TempoEstimationProcessor(fps=100)
        tempos = proc(act)
        print("Elapsed time tempo: %.3g"%(time.time()-tic))
        if do_plot:
            plt.stem(tempos[:, 0], tempos[:, 1])
            plt.xlabel("bpm")
            plt.ylabel("confidence")
        return tempos


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

    def exportOnsetClicks(self, outname, onsets):
        """
        Save an audio file with a little 440hz blip on top of each onset
        Parameters
        ----------
        outname: string
            Path to which to save file
        onsets: ndarray(int)
            Locations in multiples of hopSize of the onsets
        """
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

if __name__ == '__main__':
    s = BeatingSound()
    s.loadAudio("Datasets/GTzan/hiphop/hiphop.00018.au")
    s.getMadmomTempo(do_plot=True)
    s.exportOnsetClicks("onsets.mp3", s.getMadmomOnsets())
    plt.show()