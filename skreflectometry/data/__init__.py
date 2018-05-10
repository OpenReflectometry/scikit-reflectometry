from __future__ import print_function, division, absolute_import
# from builtins import range
import numpy as np

import scipy.io 

import os as _os
from .. import data_dir

__all__ = ['raw_xmode',
            'raw_xmode_both_cutoff']

def load_data(filename):
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

    data = scipy.io.loadmat(_os.path.join(data_dir, filename))
    return data



def save_data(filename, data, **params):
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

def parse_1d_array_mat(data):
    """
    Loads the raw X-mode data

    Parameters
    ----------
    data : dict
        Assumes the dictionary data is stored with 2-dim arrays and clears it

    Returns
    ----------
    data : dict
    """

    for key in data.keys():
        if type(data[key]) ==  np.ndarray:
            if (data[key].shape == (1,1)):
                data[key] = data[key][0,0]
            elif (data[key].ndim >1) and (len(data[key]) == 1):
                data[key] = data[key][0]
    return data
        

def raw_xmode():
    """
    Loads the raw X-mode data

    Returns
    ----------
    data : dict
    """
    return parse_1d_array_mat(load_data('raw_xmode.mat'))


def raw_xmode_both_cutoff():
    """
    Loads the raw X-mode both cutoff

    Returns
    ----------
    data : dict
    """
    return parse_1d_array_mat(load_data('raw_xmode_lower_cutoff.mat'))



