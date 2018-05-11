from scipy.signal import spectrogram
import numpy as np



# __all__ == ['sweep_spectrum']


def sweep_spectrum(frequency, signal, fs, **params):
    """
    Calculates the beat frequency spectrogram of the sweep reflectometry signal
    centered at 0 assuming a complex signal
    
    Parameters
    ----------
    frequency : array 
        Probing frequency vector
    signal : array
        Reflectometry signal, real or complex
    fs : float
        Sampling frequency of the sweep
    **params : params of the scipy.signal.spectrogram
        nperseg, noverlap, nfft
    

    Returns
    ----------
    probing_frequency : 1e array
    beat_frequency : 1e array
    spectrum : 2d array
    
    Example
    -------
    probing_frequency, beat_frequency, power = sweep_spectrum(frequency, signal, fs,
                                                          nperseg=256,noverlap=250,nfft=1024)

    """
    beat_frequency, t, power = spectrogram(signal, 
                                                    fs=fs,
                                                    return_onesided=False,
                                                    **params)
    power = np.fft.fftshift(power,axes=0)
    beat_frequency = np.fft.fftshift(beat_frequency)
    probing_frequency = frequency[(t*fs).astype('int')]
    return probing_frequency, beat_frequency, power


def sweep_delay(beat_frequency, dfdt):
    """
    Calculates the group delay from the beat_frequency and the sweeprate

    Parameters
    ----------
    beat_frequency : array
        Array of beat frequencies
    dfdt : float
        Sweep rate

    Returns
    -------
    group_delay : array
    """

    group_delay = beat_frequency/dfdt
    return group_delay


def sweep_normalize_spectrum(stft):
    """
    Normalizes a 2D matrix for each column
    """
    bins = stft.shape[1]
    stftnorm = np.zeros(stft.shape)
        #self.stftorig = self.stft;
    for i in range(0,bins):
        maxbin = np.max(stft[:,i])
        stftnorm[:,i] =stft[:,i]/maxbin
    return stftnorm




import numpy as np

class SWEEP_PERSISTENCE(object):
    """ 
    Contains the clas object that handles the STFT persistence algorithm
    The persistence is initialized with a given number of accumulations nracums.
    It internally stores an array of matrices that correspond to the size of the
    calculated STFT matrix.
    
    Parameters
    ----------
    persistence : int
        Number of sweeps to persist
    spectrum : 2d array
        2D spectrum to persist
        
    Returns
    -------
    persisted_spectrum : 2d array
        2D array of the persisted spectrum
    """
    
    def __init__(self, persistence=4, func=np.sum):
        self.persistence = persistence
        
        self.func = func
        
        self.persisted_spectrum = None
        self.__persist_iter = 0
        self.__persist_Nwindow = self.persistence
        
        self.__persist_mem_spectrum = None
    
    def __allocMemSTFT(self,stft_shape):
        # Creates aux matrices in memory for persistence window
        self.__persist_mem_spectrum = np.zeros(np.append([self.__persist_Nwindow],np.array([stft_shape])))

    def doPersist(self, func=None, memidx=None):
        """
        Calculates persistence over the given range with func
        """
        memidx = memidx or range(len(self.__persist_mem_spectrum))
        func = func or self.func
        persisted_spectrum = func(self.__persist_mem_spectrum[memidx], axis=0)
        return persisted_spectrum

    def reset(self, **kwargs):
        
        self.__init__(**kwargs)
    def __call__(self, spectrum, func=None,**kwargs):
        if self.__persist_mem_spectrum is None:
            self.__allocMemSTFT(spectrum.shape)
            self.__persist_iter = 0
        
        func = func or self.func

        # Adds current stft to memory of persistence stft
        self.__persist_mem_spectrum[self.__persist_iter] = spectrum
            
        self.persisted_spectrum = self.doPersist(func)

            
        self.__persist_iter = (self.__persist_iter +1)% self.__persist_Nwindow
        
        return self.persisted_spectrum
