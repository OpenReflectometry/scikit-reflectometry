from __future__ import print_function, division, absolute_import


import numpy as np
from scipy import constants as konst


#    TODO: add all waveguide standard dimensions http://www.miwv.com/millimeter-wave-resources/wiki/waveguide-dimensions/
#    TODO: Add single waveguide as class or structure or dict   

waveguide_standards = {
        "WR-15": {"a": 3.76e-3, "b": 1.88e-3, "f1":50e9, "f2":75e9, "band":"V"}
    }

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
    