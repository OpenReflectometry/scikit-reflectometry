from __future__ import print_function, division, absolute_import
# from builtins import range
import numpy as np
from scipy.constants import speed_of_light
from scipy.integrate import simps
from scipy.signal import spectrogram
from skreflectometry.physics import (
    plasma_frequency, distance_vacuum, plasma_density, time_delay_vacuum
)
from skreflectometry.reflectometry_sim import (
    group_delay, phase_delay, beat_signal, beat_maximums
)


def cutoff_freq_O(density):
    """
    Calculate the cut-off frequency of O mode.

    The cut-off frequency for O mode is the plasma frequency.

    Parameters
    ----------
    density : number or ndarray
        Density/ies of the medium.

    Returns
    -------
    number or ndarray
        Cut-off frequency/ies.

    """

    return plasma_frequency(density)


def refraction_index_O(wave_freq, density, squared=False):
    """
    Calculate the refraction index of an O mode wave in a medium.

    Parameters
    ----------
    wave_freq : number or ndarray
        Frequency of the wave entering the medium.
    density : number or ndarray
        Density of the medium.
    squared : bool, optional
        If squared is True, return the square of the refraction index. Otherwise
        return the real part of the refractive index. Default is False.

    Returns
    -------
    number
        The refraction index.

    """

    n_squared = 1 - np.power(plasma_frequency(density) / wave_freq, 2.)

    if squared:
        return n_squared
    else:
        # don't allow negative refractive indexes
        return np.sqrt(np.maximum(n_squared, 0))


def refractive_matrix_O(dens_prof, freq_samp, squared=False):
    """
    TODO
    Parameters
    ----------
    dens_prof
    freq_samp

    Returns
    -------

    """
    dens_mat, freq_mat = np.meshgrid(dens_prof, freq_samp)

    return refraction_index_O(freq_mat, dens_mat, squared)


def group_delay_from_spectrogram_O(f_spectrum, t_spectrum, spectrum,
                                   sweep_rate, f_probe_limits):
    """
    TODO
    Parameters
    ----------
    f_spectrum
    t_spectrum
    spectrum
    sweep_rate
    f_probe_limits

    Returns
    -------

    """

    beat_max = beat_maximums(f_spectrum, spectrum)

    f_probe_spect = t_spectrum * sweep_rate + f_probe_limits[0]
    tau_g_spect = beat_max / sweep_rate

    return f_probe_spect, tau_g_spect


def initialize_group_delay(f_probe_spect, tau_g_spect,
                           vacuum_distance=0.1, num_points_delay=16):
    """
    TODO
    Parameters
    ----------
    f_probe_spect
    tau_g_spect
    vacuum_distance
    num_points_delay

    Returns
    -------

    """

    if num_points_delay > 0:
        f_probe_init = np.linspace(1, f_probe_spect[0], num_points_delay,
                                   endpoint=False)
        tau_0 = time_delay_vacuum(vacuum_distance)
        tau_g_init = np.linspace(tau_0, tau_g_spect[0], num_points_delay,
                                 endpoint=False)

        f_probe_spect = np.hstack([f_probe_init, f_probe_spect])
        tau_g_spect = np.hstack([tau_g_init, tau_g_spect])

    return f_probe_spect, tau_g_spect


def abel_inversion_single(freq_samp, time_delay, current_ind, pos_antenna=1.15,
                          other_method=False):
    """
    TODO

    .. math:: R_c(\omega_{pe}) = R_{ant} - c/\omega \int_0 ^{\omega_{pe}}
        t_g \quad 1 / \sqrt{\omega_{pe}^2 - \omega^2} \\text{d} \omega


    Parameters
    ----------
    freq_samp
    time_delay
    current_ind
    pos_antenna
    other_method

    Returns
    -------

    """
    fpe = freq_samp[current_ind]
    density = plasma_density(fpe)

    if current_ind < 1:
        return pos_antenna, density

    if other_method:
        # Method: TODO

        freqs_used = freq_samp[:current_ind + 1]
        times_used = time_delay[1:current_ind + 1]

        temp_arr = np.arcsin(freqs_used[1:] / fpe) - np.arcsin(
            freqs_used[:-1] / fpe)

        integral = np.dot(temp_arr, times_used)
        # r_vacuum = pos_antenna
    else:
        # Naive Abel Inversion

        freqs_used = freq_samp[:current_ind]
        times_used = time_delay[:current_ind]

        temp_arr = 1 / np.sqrt(np.power(fpe, 2) - np.power(freqs_used, 2))
        integral = simps(times_used * temp_arr, x=freqs_used)
        # r_vacuum = distance_vacuum(time_delay[0]) / 2 + pos_antenna

    radius = speed_of_light * integral / np.pi + pos_antenna

    return radius, density


def abel_inversion(freq_samp, time_delay, pos_antenna=1.15, other_method=False):
    """
    TODO
    Parameters
    ----------
    freq_samp
    time_delay
    pos_antenna
    other_method

    Returns
    -------

    """

    radius_arr = np.zeros_like(freq_samp)
    dens_arr = np.zeros_like(freq_samp)

    for freq_index in range(len(freq_samp)):
        radius_arr[freq_index], dens_arr[freq_index] = abel_inversion_single(
            freq_samp, time_delay, freq_index,
            pos_antenna=pos_antenna, other_method=other_method
        )

    return radius_arr, dens_arr


def full_analysis(radius_arr, dens_prof, f_sampling=125e6, sweep_time=25e-6,
                  f_probe_limits=(1, 1e11), full_output=False,
                  antenna_side='hfs', reflect_at_wall=True, pos_antenna=1.15,
                  vacuum_distance=0.1, num_points_delay=16):
    """
    TODO
    Parameters
    ----------
    radius_arr
    dens_prof
    f_sampling
    sweep_time
    f_probe_limits
    full_output
    antenna_side
    reflect_at_wall
    vacuum_distance
    num_points_delay

    Returns
    -------

    """

    n_points_fs = int(sweep_time * f_sampling)

    sweep_rate = (f_probe_limits[1] - f_probe_limits[0]) / sweep_time
    f_probe = np.linspace(f_probe_limits[0], f_probe_limits[1], n_points_fs)

    refract_index = refractive_matrix_O(dens_prof, f_probe)

    phi = phase_delay(f_probe, radius_arr, refract_index,
                      antenna_side=antenna_side,
                      reflect_at_wall=reflect_at_wall)
    tau_g = group_delay(f_probe, phi)

    beat_sig = beat_signal(f_probe, tau_g)

    f_spectrum, t_spectrum, spectrum = spectrogram(
        beat_sig, fs=f_sampling, nperseg=136, nfft=2048, noverlap=128
    )

    f_probe_spect, tau_g_spect = group_delay_from_spectrogram_O(
        f_spectrum, t_spectrum, spectrum, sweep_rate, f_probe_limits
    )

    f_probe, tau_g = initialize_group_delay(
        f_probe, tau_g,
        vacuum_distance=vacuum_distance, num_points_delay=num_points_delay
    )

    f_probe_spect, tau_g_spect = initialize_group_delay(
        f_probe_spect, tau_g_spect,
        vacuum_distance=vacuum_distance, num_points_delay=num_points_delay
    )

    radius_original, dens_original = abel_inversion(
        f_probe, tau_g, pos_antenna=pos_antenna, other_method=True
    )
    radius_spect, dens_spect = abel_inversion(
        f_probe_spect, tau_g_spect, pos_antenna=pos_antenna, other_method=True
    )

    if full_output:
        return {
            'f_sampling': f_sampling,
            'sweep_rate': sweep_rate,
            'f_probe': f_probe,
            'refract_index': refract_index,
            'phi': phi,
            'tau_g': tau_g,
            'beat_sig': beat_sig,
            'radius_calc': radius_original,
            'dens_calc': dens_original,
            'spectrum_data': {
                'f': f_spectrum, 't': t_spectrum, 'signal': spectrum,
                'tau_g': tau_g_spect, 'f_probe': f_probe_spect,
                'radius_calc': radius_spect, 'dens_calc': dens_spect
            }
        }
    else:
        return f_sampling, f_probe, beat_sig, sweep_rate, \
               radius_original, dens_original, radius_spect, dens_spect


def CalcInvPerfO(fpro, gdel, vacd=0.0, initpts=32):
    """Inverts an O-mode groupd delay into a density profile.
    Group delay is not initialized

    Parameters
    ------------
    fpro: array
          Probing frequencies
    gdel: array
          Group delays. Must be the same size as "fpro"
    vacd: float
          Vaccuum distance to the antenna: where the plasma is believed to start.
    initpts: int
             How many points to be used in the initialization

    Returns
    ------------
    rad: array
         Radius
    dens: array
          Density
    fpro: array
          The actual probing frequency used to calculate the density profile
    gdel: array
          The actual group delay used to calculate the density profile
    """

    k3 = speed_of_light / np.pi

    # Initialization points
    init0_f = 0.0
    vac_gdel = 2.0 * vacd / speed_of_light
    init0_g = vac_gdel

    init1_f = fpro[0]
    init1_g = gdel[0]

    # Automatically initializes GD, if Fprobe starts at zero this does nothing
    initf = np.linspace(init0_f, init1_f, num=initpts, endpoint=False)
    initg = np.linspace(init0_g, init1_g, num=initpts, endpoint=False)

    fpro = np.concatenate((initf, fpro))
    gdel = np.concatenate((initg, gdel))

    # Converts the density
    dens = plasma_density(fpro)

    # Gets the integral done
    summa = []
    # for i in range(np.size(dens)):
    #    inte = []
    #    for j in range(i):
    #        inte.append( gdel[j] * 1.0/(np.sqrt(fpro[i]*fpro[i]-fpro[j]*fpro[j])) )
    #    summa.append(np.trapz(inte, x=fpro[0:i]))
    # rad = konst.c/np.pi * np.array(summa)
    II = np.zeros_like(fpro)
    for j in range(1, len(gdel)):
        for i in range(1, j + 1):
            II[j] = II[j] + gdel[i] * (
                np.arcsin(fpro[i] / fpro[j]) - np.arcsin(fpro[i - 1] / fpro[j]))

    rad = k3 * II
    return rad, dens, fpro, gdel
