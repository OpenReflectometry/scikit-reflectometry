from __future__ import print_function, division, absolute_import
# from builtins import range
import numpy as np

import scipy.io 


def load_data(filename)
    """
    Loads the data in the matlab file given by filename 
    Parameters
    ----------
    filename : string

    Returns
    ----------
    data : dict
        Dictionary containing loaded data

    """

    data = scipy.io.loadmat(filename)
    return data



def save_data(filename, data, **params)
    """
    Saves the data dictionary into a filename given by filename
    Parameters
    ----------
    filename : string
    data : dict

    Returns
    ----------
    
    """

    return scipy.io.save(filename, data, **params)




def raw_xmode():
    """
    Loads the raw X-mode data

    Returns
    ----------
    data : dict
    """
    return load_mat('raw_xmode.mat')


def raw_xmode_both_cutoff():
    """
    Loads the raw X-mode both cutoff

    Returns
    ----------
    data : dict
    """
    return load_mat('raw_xmode_lower_cutoff.mat')



