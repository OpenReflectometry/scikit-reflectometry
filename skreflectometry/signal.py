from scipy.signal import spectrogram
from scipy import signal
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


def sweep_normalize_spectrum(spectrum):
    """
    Normalizes a 2D matrix for each column

    Parameters
    ----------
    spectrum : 2D array

    Returns
    -------
    spectrum : 2D array
        2D matrix normalized along the column

    """

    bins = spectrum.shape[1]
    norm = np.zeros(spectrum.shape)
        #self.stftorig = self.stft;
    for i in range(0,bins):
            maxbin = np.max(spectrum[:,i])
            norm[:,i] =spectrum[:,i]/maxbin
    return norm


def sweep_peaks(pf, fb, stft, minfb=None):
    """ Returns the peak positive value of fb for all pf 
    """
    
    if minfb is None:
        minfb=0
    zerorange = common.closest_index(fb,minfb)
    maxval = []
    maxfb = []
    # Limits search range to above DC level
    for i in range(0,len(pf)):
        # maxfb[i]=np.max(stft[:,i]);
        idx = np.argmax(stft[zerorange:,i])
        idx = idx + zerorange
        maxval =np.append(maxval, stft[idx,i])
        maxfb  =np.append(maxfb, fb[idx])
    maxval = np.divide(maxval,np.max(maxval))
    pk_fb = maxfb
    pk_amp = maxval
    #dp_dg = self.dt/self.df*self.dp_fb;
    pk_pf = pf
    
    return pk_pf, pk_fb, pk_amp



def filter_butterworth(sig, cutoff, fs=1, order=5,  type='low'):
    """ 
    Butterworth filter
    
    Parameters
    ---------
    sig : array
        Input signal
    cutoff : float
        Cutoff frequency of the filter
    fs : float
        Sampling frequency
    order : int
        Order of the filter
    type : string (default low)
        Type of the butterworth filter
        
    Returns
    -------
    sig : array
        Filtered signal
    """
    assert cutoff <= fs, "Cutoff must be lower than fs"
    assert order >= 3, "Filter order must be >= 3"
    nyq = np.float(fs)/2
    cutoffnorm = cutoff/nyq
    b, a = signal.butter(order, cutoffnorm, btype=type)
    sig = signal.filtfilt(b,a,sig)
    return sig


def filter_lowpass(sig, cutoff, fs=1, order=5):
    """ 
    Low pass butterworth filter
    
    Parameters
    ---------
    sig : array
        Input signal
    cutoff : float
        Cutoff frequency of the filter
    fs : float
        Sampling frequency
    order : int
        Order of the filter
    
    Returns
    -------
    sig : array
        Low pass filtered
    """
    return filter_butterworth(sig, cutoff, fs=fs, order=order, type='low')

def filter_highpass(sig, cutoff, fs=1, order=5):
    """ 
    Low pass butterworth filter
    
    Parameters
    ---------
    sig : array
        Input signal
    cutoff : float
        Cutoff frequency of the filter
    fs : float
        Sampling frequency
    order : int
        Order of the filter
    
    Returns
    -------
    sig : array
        Low pass filtered
    """
    return filter_butterworth(sig, cutoff, fs=fs, order=order, type='high')

def filter_bandpass(sig, cutoff, fs=1, order=5):
    """ 
    Low pass butterworth filter
    
    Parameters
    ---------
    sig : array
        Input signal
    cutoff : (float, float)
        Cutoff frequency of the filter
    fs : float
        Sampling frequency
    order : int
        Order of the filter
    
    Returns
    -------
    sig : array
        Low pass filtered
    """
    assert len(cutoff)==2,"Cutoff must be be (lowcut, highcut)"
    return filter_butterworth(sig, cutoff, fs=fs, order=order, type='band')


def frequency_instantaneous(sig, fs=1):
    """ Calculates the instantaneous frequency based on signal phase

    Parameters
    ----------
    sig : array
        Complex signal
    fs : float
        Sampling frequency
    
    Returns
    -------
    frequency : array
        Instantaneous frequency of the signal
    """
    assert np.iscomplex(sig).any(), "Signal must be complex"
    ts = 1.0/fs
    phase = np.unwrap(np.angle(sig))
    phasediff = np.append([0],np.diff(phase))
    freq = phasediff/(2*np.pi*ts)
    return freq



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
