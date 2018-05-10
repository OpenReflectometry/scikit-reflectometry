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
