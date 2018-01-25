from __future__ import print_function, division, absolute_import
# from builtins import range
import numpy as np
from skreflectometry.physics import plasma_frequency, cyclotron_frequency


def cutoff_freq_X(density, magnetic_field):
    """
    Calculate the cut-off frequencies of X mode.

    The cut-off frequencies for X mode are given by
    :math:`f_{L,R}=\\sqrt{f_{pe}^2 + f_{ce}^2/4}\\ \mp f_{ce} / 2` .

    Parameters
    ----------
    density : number or ndarray
        Density/ies of the medium.
    magnetic_field : number or ndarray
        Magnetic field(s) present.

    Returns
    -------
    number or ndarray
        Left cut-off frequency/ies.
    number or ndarray
        Right cut-off frequency/ies.

    """

    fpe = plasma_frequency(density)
    fce = cyclotron_frequency(magnetic_field)

    f_temp = np.sqrt(np.power(fpe, 2.) + np.power(fce, 2.) / 4)

    cutoff_left = f_temp - fce / 2
    cutoff_right = f_temp + fce / 2

    return cutoff_left, cutoff_right


def refraction_index_X(wave_freq, density, magnetic_field, squared=False):
    """
    Calculate the refraction index of an O mode wave in a medium.

    Parameters
    ----------
    wave_freq : number or ndarray
        Frequency of the wave entering the medium.
    density : number or ndarray
        Density of the medium.
    magnetic_field : number or ndarray
        Magnetic field present in the medium.
    squared : bool, optional
        If squared is True, return the square of the refraction index. Otherwise
        return the real part of the refractive index. Default is False.

    Returns
    -------
    number
        The refraction index.

    """

    fce2 = np.power(cyclotron_frequency(magnetic_field), 2.)
    fpe2 = np.power(plasma_frequency(density), 2.)
    fwav2 = np.power(wave_freq, 2.)

    n_squared = 1 - (fpe2 / fwav2) * (fwav2 - fpe2) / (fwav2 - fpe2 - fce2)

    if squared:
        return n_squared
    else:
        # don't allow negative refractive indexes
        return np.sqrt(np.maximum(n_squared, 0))


def refractive_matrix_X(dens_prof, freq_samp, mag_field, squared=False):
    """
    TODO
    Parameters
    ----------
    dens_prof : ndarray
        Density of the medium.
    freq_samp : ndarray
        Frequency of the wave entering the medium.
    mag_field : ndarray
        Magnetic field present in the medium.
    squared : bool, optional
        If squared is True, return the square of the refraction index. Otherwise
        return the real part of the refractive index. Default is False.

    Returns
    -------

    """
    dens_mat, freq_mat = np.meshgrid(dens_prof, freq_samp)
    mag_mat = np.tile(mag_field, (len(freq_samp), 1))

    return refraction_index_X(freq_mat, dens_mat, mag_mat, squared)
