#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 17:10:46 2024

@author: crrowell
"""

import scipy.io as spio
import numpy as np
from typing import Union


def matfile_struct_to_dict(matfile: str, var_name: str) -> dict:
    """
    

    Parameters
    ----------
    matfile : str
        Path to .mat file.
    var_name : TYPE
        Name of struct variable in .mat file.

    Returns
    -------
    python_dict : dict
        Dictionary containg keys and values which correspond
        to the original matlab struct.
    
    Notes
    _______
    Watch for variable types as they may not convert cleanly.
    Matlab doubles may be converted to integers, for example,
    or some objects may be nested in arrays, etc.

    """
    # Load the .mat file
    mat_data = spio.loadmat(matfile)
    # Extract the structure array from the loaded data
    matlab_struct = mat_data[var_name]
    python_dict = matlab_struct_to_dict(matlab_struct)
    
    return python_dict

# Function to convert MATLAB structure to Python dictionary
def matlab_struct_to_dict(matlab_struct: dict) -> dict:
    """
    Parameters
    ----------
    matlab_struct : array
        Array representing matlab structure data as 
        returned by scipy.io.loadmat.

    Returns
    -------
    python_dict : dict
        Dictionary containg keys and values which correspond
        to the original matlab struct.

    """
    python_dict = {}
    for field in matlab_struct.dtype.names:
        python_dict[field] = matlab_struct[field][0, 0].squeeze()
        # if not np.shape(python_dict[field]):
        #     python_dict[field] =  python_dict[field][0]
    return python_dict

def extrapVentRadius(Q: Union[float,np.ndarray[float]]) -> Union[float,np.ndarray[float]]:
    # Conduit radius extrapolation for control scenarios    
    # Empirical curve fit for conduit area
    C1 = 0.009913;
    C2 = 0.7267;
    C3 = 58.04;
    afun = lambda x : C1 * (x)**C2 + C3;

    r = (afun(Q) / np.pi)**(1/2);
    
    return r