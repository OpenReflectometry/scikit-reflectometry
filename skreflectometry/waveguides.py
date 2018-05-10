from __future__ import print_function, division, absolute_import


import numpy as np
from scipy import constants as konst


#    TODO: add all waveguide standard dimensions http://www.miwv.com/millimeter-wave-resources/wiki/waveguide-dimensions/
#    TODO: Add single waveguide as class or structure or dict

#from waveguide_standards import waveguide_standards
waveguide_standards = {'WR-10': {'band': 'W',
  'a': 0.00254,
  'b': 0.00127,
  'f1': 75000000000.0,
  'f2': 110000000000.0},
 'WR-12': {'band': 'E',
  'a': 0.0030988,
  'b': 0.0015494,
  'f1': 60000000000.0,
  'f2': 90000000000.0},
 'WR-15': {'band': 'V',
  'a': 0.0037592,
  'b': 0.0018796,
  'f1': 50000000000.0,
  'f2': 75000000000.0},
 'WR-19': {'band': 'U',
  'a': 0.0047752,
  'b': 0.0023876,
  'f1': 40000000000.0,
  'f2': 60000000000.0},
 'WR-22': {'band': 'Q',
  'a': 0.0056896,
  'b': 0.0028448,
  'f1': 33000000000.0,
  'f2': 50000000000.0},
 'WR-28': {'band': 'Ka',
  'a': 0.007112,
  'b': 0.003556,
  'f1': 26500000000.0,
  'f2': 40000000000.0},
 'WR-42': {'band': 'K',
  'a': 0.010667999999999999,
  'b': 0.004318000000000001,
  'f1': 18000000000.0,
  'f2': 26500000000.0},
 'WR-5': {'band': 'G',
  'a': 0.0012954000000000002,
  'b': 0.0006477000000000001,
  'f1': 140000000000.0,
  'f2': 220000000000.0},
 'WR-51': {'band': 'K',
  'a': 0.012954,
  'b': 0.006477,
  'f1': 15000000000.0,
  'f2': 22000000000.0},
 'WR-6': {'band': 'D',
  'a': 0.0016510000000000001,
  'b': 0.0008255000000000001,
  'f1': 110000000000.0,
  'f2': 170000000000.0},
 'WR-62': {'band': 'Ku',
  'a': 0.0157988,
  'b': 0.0078994,
  'f1': 12400000000.0,
  'f2': 18000000000.0},
 'WR-8': {'band': 'F',
  'a': 0.002032,
  'b': 0.001016,
  'f1': 90000000000.0,
  'f2': 140000000000.0},
 'WR-90': {'band': 'X',
  'a': 0.02286,
  'b': 0.01016,
  'f1': 8199999999.999999,
  'f2': 12400000000.0}}


def group_velocity(f, fc=31.3905e9, v=konst.c):
    """
    Calculates group velocity for a given waveguide with cut off frequency fc
    Default parameters for U-band
    Parameters
    ----------
    f : array float
        Frequency for which to get group velocities
    fc : float
        Waveguide cut off frequency
    v : float (default: c)
        Velocity of wave in unbounded medium

    Returns
    -------
    vg : array float
        Group velocity for each of the frequencies
    """
    vg = v*np.sqrt(1-(np.power(fc/f,2)))
    return vg

def cutoff(a, v=konst.c):
    """
    Calculates the waveguide cutoff from the long edge term

    Parameters
    ----------
    a : float
        Long edge of the waveguide cross section [meter]
    v : float (default: c)
        Velocity of wave in unbounded medium

    Returns
    -------
    fc : float
        Geometry waveguide cutoff frequency [Hz]
    """

    fc = v/(2*a)
    return fc

def get_dimensions(standard):
    """
    Returns the rectangular waveguide cross section dimensions a,b from standard

    Parameters
    ----------
    standard : string
        Name of waveguide standard

    Returns
    ----------
    a,b : float
        Long and short edges of the waveguide dimension


    """

    a = waveguide_standards[standard]["a"]
    b = waveguide_standards[standard]["b"]
    return a,b

def get_frequencies(standard, size=2):
    """
    Returns the rectangular waveguide frequency range from standard

    Parameters
    ----------
    standard : string
        Name of waveguide standard
    size : int (default=2)

    Returns
    ----------
    f : array float
        Array of frequency range between the limits with a given size


    """



    f1 = waveguide_standards[standard]["f1"]
    f2 = waveguide_standards[standard]["f2"]

    f = np.linspace(f1,f2,size)
    return f

def get_band(standard):
    """
    Returns the band name for the given waveguide standard

    Parameters
    ----------
    standard : string
        Name of waveguide standard

    Returns
    ----------
    band : string
    """

    return waveguide_standards[standard]["band"]

def propagation_delay(vg, length):
    """
    Calculates waveguide delay for a given length of waveguide

    Parameters
    ----------
    vg : array
        Group velocity

    length : float
        Length of waveguides

    Returns
    -------
    delay : array float
        Delays given for each frequency
    """

    delay_waveguide = length/vg
    return delay_waveguide

def obsolete_propagation_delay(f, waveguide, length, size):
    """
    Calculates waveguide delay for a given length of waveguide

    Parameters
    ----------
    f : array
        Frequency range
    waveguide : string
        Waveguide standard name
    length : float
        Length of waveguides

    Returns
    -------
    delay : array float
        Delays given for each frequency
    """

    a,b = waveguides.get_dimensions(waveguide)

    fc = waveguides.cutoff(a)
    vg = waveguides.group_velocity(f,fc=fc)

    delay_waveguide = length/vg
    return delay_waveguide
