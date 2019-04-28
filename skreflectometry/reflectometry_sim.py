from __future__ import print_function, division, absolute_import
from builtins import range
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.constants import speed_of_light
from scipy.signal import spectrogram

def phase_delay2(freq_probing, radius_arr, refractive_mat_sq, refract_epsilon=1e-9,
                antenna_side='lfs', reflect_at_wall=True, interp_pts=1e6):
    """
    TODO
    Parameters
    ----------
    freq_probing: ndarray
        Probing frequency of the band (or bands) used in the sweep
    radius_arr: ndarry
        Radius of the plasma in machien coordinates.
    refractive_mat: numpy matrix
        Matrix containing the refractive index with dimensions (radius_arr, freq_probing)    
    refract_epsilon: float
        Residual to be used in numerical integrations
    antenna_side: string
        Either 'hfs' for propagation left-right or 'lfs' for right-left propagation.
    reflect_at_wall : bool
        TODO
        Setting this to 'True' enables a reflection at the back-wall.
    
    Returns
    -------
    phase_diff: ndarray
        The total phase shift for the frequencies in freq_probing.
        
    Raises
    ------
    ValueError
        If the 'antenna_side' is other than 'lfs' or 'hfs'. The code converts all letters to lower-case, as a precautions
    """

    if antenna_side.lower() == 'hfs':
        vessel_side = 'hfs'
    elif antenna_side.lower() == 'lfs':
        vessel_side = 'lfs'
    else:
        raise ValueError('Unknown antenna_side option: ', str(antena_side))
    
    #Declare phase delay
    phase_delay = np.zeros_like(freq_probing)

    for ind in range(len(freq_probing)):
        if list(refractive_mat_sq[ind,:]>=0.0).count(True) == len(radius_arr): #Wave propagates through the plasma
            if reflect_at_wall:
                refract_int = simps(np.sqrt(refractive_mat_sq[ind,:]), x=radius_arr)
            else: #Propagates through the region
                refract_int = np.nan
        else: #There's a reflection layer
            #Find the non-propagating region
            msk = refractive_mat_sq[ind,:]<=0.0
            neg_rad = radius_arr[msk]
            #Finds the first negative index beyond the refraction index pass through zero
            if vessel_side == 'lfs':
                critical_index = np.argmax(radius_arr==np.max(neg_rad)) #Argmax returns the only True
                integ_rad = radius_arr[critical_index-1:]
                integ_N2 = refractive_mat_sq[ind,critical_index-1:]
                interp_rad = interp1d(integ_N2[0:3], integ_rad[0:3], kind='quadratic')
                zeroth_position = interp_rad(0.0)
                new_rad = np.linspace(zeroth_position, radius_arr[-1], num=interp_pts, endpoint=True)
                index_to_zero = 0
            else: #hfs
                critical_index = np.argmax(radius_arr==np.min(neg_rad))
                integ_rad = radius_arr[:critical_index+2]
                integ_N2 = refractive_mat_sq[ind,:critical_index+2]
                interp_rad = interp1d(integ_N2[-4:], integ_rad[-4:], kind='quadratic')
                zeroth_position = interp_rad(0.0)
                new_rad = np.linspace(radius_arr[0], zeroth_position, num=interp_pts, endpoint=True)
                index_to_zero = -1
                
            interp_N2 = interp1d(integ_rad, integ_N2, kind='quadratic', fill_value=0.0)
            new_N2 = interp_N2(new_rad)
            #Rude approximation
            new_N2[index_to_zero] = 0.0
            refract_int = simps(np.sqrt(new_N2), x=new_rad)
    
        phase_diff = 4.0 * np.pi * freq_probing[ind] / speed_of_light * refract_int
        phase_diff -= np.pi / 2.0
    
        phase_delay[ind] = phase_diff
        
    return phase_delay



def phase_delay(freq_probing, radius_arr, refractive_mat, refract_epsilon=1e-15,
                antenna_side='hfs', reflect_at_wall=True):
    """
    TODO
    Parameters
    ----------
    freq_probing: ndarray
        Probing frequency of the band (or bands) used in the sweep
    radius_arr: ndarry
        Radius of the plasma in machien coordinates.
    refractive_mat: numpy matrix
        Matrix containing the refractive index with dimensions (radius_arr, freq_probing)    
    refract_epsilon: float
        Residual to be used in numerical integrations
    antenna_side: string
        Either 'hfs' for propagation left-right or 'lfs' for right-left propagation.
    reflect_at_wall : bool
        TODO
        Setting this to 'True' enables a reflection at the back-wall.
    
    Returns
    -------
    phase_diff: ndarray
        The total phase shift for the frequencies in freq_probing.
        
    Raises
    ------
    ValueError
        If the 'antenna_side' is other than 'lfs' or 'hfs'. The code converts all letters to lower-case, as a precautions
    ValueError
        If the `method` selected does not exist.
    """

    if antenna_side.lower() == 'hfs':
        a=0
    elif antenna_side.lower() == 'lfs':
        refractive_mat = refractive_mat[:, ::-1]
    else:
        raise ValueError('Unknown antenna_side option: '+str(antena_side))

        
    #This method does not have enough precision
    # Returns the index of the first position where refraction < epsilon for
    #   every sampling frequency. If there is no such point, it returns 0.    
    reflect_pos_ind = np.argmax(refractive_mat <= refract_epsilon, axis=1)

    # Find if reflect_pos_ind == 0 are real reflections at entrance (pos = 0)
    #   or no reflection in the plasma.
    reflect_at_0_ind = (refractive_mat[:, 0] <= refract_epsilon)
    reflect_at_0_measured_ind = (reflect_pos_ind == 0)
    reflect_at_0_fake_ind = reflect_at_0_ind ^ reflect_at_0_measured_ind

    if reflect_at_wall:  # TODO
        reflect_pos_ind[reflect_at_0_fake_ind] = -1
    else:
        reflect_pos_ind[reflect_at_0_fake_ind] = 0

#     if method == 'trapz':
#         integral_func = np.trapz
#     elif method == 'simps':
#         integral_func = simps
#     else:
#         raise ValueError("Parameter 'method' must be 'trapz' or 'simps'")

    refract_mat_temp = np.copy(refractive_mat)
    for freq_ind in range(len(freq_probing)):
        refract_mat_temp[freq_ind, reflect_pos_ind[freq_ind]:-1] = 0

    refract_int = simps(refract_mat_temp, radius_arr, axis=1)

    phase_diff = 2 * 2 * np.pi * freq_probing / speed_of_light * refract_int
    phase_diff -= np.pi / 2

    return phase_diff


def group_delay(freq_probing, phase_delay_arr):
    """
    Calculates the group delay from the array of probing frequencies and the phase delays.
    Computes the beat frequency from the phase delays and sweep rate from probing frquencies.
    TODO: Assumes an acquisition rate of???
    Parameters
    ----------
    freq_probing: ndarray
        Probing frequency of the band (or bands) used in the sweep
    phase_delay_arr: ndarray
        Calculated phase delay of a frequency sweep of 'freq_probing'
    Returns
    -------
    group_delay: ndarray
        Ratio between the beat frequency and the Sweeping rate
    """

    phase_diff = np.gradient(phase_delay_arr)
    omega_diff = np.gradient(2 * np.pi * freq_probing)
    group_delay = phase_diff / omega_diff
    
    return group_delay


# def beat_spectrogram(beat, fs=1.0,
#                      nperseg=136, nfft=2048, noverlap=128):
#     """
#     TODO
#     Parameters
#     ----------
#     freq_samp
#     time_delay_arr
#     fs
#     nperseg
#     nfft
#     noverlap
#
#     Returns
#     -------
#
#     """
#
#     freqs, times, spectrum = \
#         spectrogram(beat, fs=fs,
#                     nperseg=nperseg, nfft=nfft, noverlap=noverlap)
#
#     return freqs, times, spectrum


def beat_signal(freq_probing, time_delay_arr):
    """
    TODO
    Parameters
    ----------    
    freq_probing: ndarray
        Probing frequency of the band (or bands) used in the sweep
    time_delay_arr: ndarray
        

    Returns
    -------
    A simulated signal of unitary amplitude according to the definition cos(phi) where 'phi' is
    the total phase delay for the instantaneous frequency.
    """
    return np.cos(np.cumsum(time_delay_arr * np.gradient(2 * np.pi * freq_probing)))


def beat_maximums(freqs, spectrum):
    """
    TODO
    Parameters
    ----------
    freqs
    spectrum

    Returns
    -------

    """

    return freqs[np.argmax(spectrum, axis=0)]
