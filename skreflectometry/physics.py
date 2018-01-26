from __future__ import print_function, division, absolute_import
# from builtins import range
import numpy as np
from scipy.constants import elementary_charge, electron_mass, epsilon_0, \
    speed_of_light


def cyclotron_frequency(magnetic_field):
    """Calculate the cyclotron frequency (:math:`f_{ce}` ) for a given
        magnetic field."""
    return elementary_charge * magnetic_field / (2 * np.pi * electron_mass)


def cyclotron_field(cyclotron_freq):
    """Calculate the magnetic field for a given cyclotron frequency."""
    return 2 * np.pi * electron_mass * cyclotron_freq / elementary_charge


def plasma_frequency(density):
    """Calculate the plasma frequency (:math:`f_{pe}` ) for a given density."""
    return elementary_charge / (2 * np.pi) * np.sqrt(
        density / (electron_mass * epsilon_0))


def plasma_density(frequency):
    """Calculate the plasma density for a given frequency."""
    return electron_mass * epsilon_0 * np.power(2 * np.pi * frequency, 2) / \
        np.power(elementary_charge, 2)


def upper_hybrid_frequency(density, magnetic_field):
    """Calculate the upper hybrid frequency (:math:`f_H` ) for given density
        and magnetic fields."""
    return np.sqrt(np.power(plasma_frequency(density), 2.) +
                   np.power(cyclotron_frequency(magnetic_field), 2.))


def time_delay_medium(distance, refractive_index=1):
    """
    Calculate the round trip delay for a certain distance and medium.

    Parameters
    ----------
    distance : number
        One way distance in a medium.
    refractive_index : number, optional
        Refractive index of the medium. Default value is 1 (vacuum).

    Returns
    -------
    number
        The round trip delay.
    """

    return 2 * distance * refractive_index / speed_of_light


def time_delay_vacuum(distance):
    """
    Calculate the round trip delay for a certain distance in vacuum.

    Parameters
    ----------
    distance : number
        One way distance in vacuum.

    Returns
    -------
    number
        The round trip delay.
    """

    return time_delay_medium(distance, refractive_index=1)


def distance_vacuum(time_delay):
    """
    TODO
    Parameters
    ----------
    time_delay

    Returns
    -------

    """

    return time_delay * speed_of_light
