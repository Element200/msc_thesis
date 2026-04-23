# -*- coding: utf-8 -*-
"""
distance_functions.py.

Created on Tue Jun 10 11:35:46 2025
v1.0

@author: sane

Distance functions between PDFs. All distance functions must satisfy the properties:
    1. d[p(x,t),π(x;T)] > 0 for all p;
    2. d[p(x,t_2),π(x;T)] <= d[p(x,t_1),π(x;T)] if t_2 > t_1 (monotonic nonincreasing-ness with relaxation);
    3. d[π(x;T_h),π(x;T_c)] > d[π(x;T_w),π(x;T_c)] if T_h > T_w (monotonic increasingness with temperature)
    
These are optimised to take in vectors of histograms and operate on the bins axis.
"""

import numpy as np
import scipy

def L1(hist_1, hist_2, dx, axis=None):
    """
    Find sum |hist_2-hist_1|_i. Does NOT calculate the integral; inputs must be converted to PMFs before function is called.

    Parameters
    ----------
    hist_1 : 2D vector of numerics
        Some probability density vector (this will not fail if hist_1 is not a probability mass but the intended use case is for PMFs).
    hist_2 : 2D vector of numerics of same shape as hist_1
        Second probability density vector.
    dx: float
        Distance between bins
    axis : int, optional
        If hist_1 is multi-dimensional, the axis to sum along. The default is None.

    Returns
    -------
    float
        sum |hist_2-hist_1|_i. If this is less than zero or greater than two, hist_1 and hist_2 are not correctly normalised.

    """
    # hist_1 and hist_2 must be at least two-dimensional arrays
    assert hist_1.shape[axis]==hist_2.shape[axis], f"Mismatch between hist_1 {np.array(hist_1).shape} and hist_2 {np.array(hist_2).shape}"
    return np.sum(np.abs(hist_2-hist_1), axis=axis)*dx

def Lp_constructor(p):
    """Returns a p-norm."""
    def Lp(hist_1, hist_2, dx, axis=None):
        """
        Find sum |hist_2-hist_1|^p_i. Does NOT calculate the integral; inputs must be converted to PMFs before function is called.
    
        Parameters
        ----------
        hist_1 : 2D vector of numerics
            Some probability density vector (this will not fail if hist_1 is not a probability mass but the intended use case is for PMFs).
        hist_2 : 2D vector of numerics of same shape as hist_1
            Second probability density vector.
        dx: float
            Distance between bins
        axis : int, optional
            If hist_1 is multi-dimensional, the axis to sum along. The default is None.
        p : int, optional
            The p in the p-norm. The default is 2.
    
        Returns
        -------
        float
            sum |hist_2-hist_1|_i. If this is less than zero or greater than two, hist_1 and hist_2 are not correctly normalised.
    
        """
        # hist_1 and hist_2 must be at least two-dimensional arrays
        assert hist_1.shape[axis]==hist_2.shape[axis], f"Mismatch between hist_1 {np.array(hist_1).shape} and hist_2 {np.array(hist_2).shape}"
        return dx*(np.sum(np.abs(hist_2-hist_1)**p, axis=axis)**(1/p))
    return Lp

def kullback_leibler(hist_1, hist_2, dx, axis=None):
    """
    Take the entropic distance between hist_1 and hist_2. Not implemented to fullest potential yet.

    Parameters
    ----------
    hist_1 : 2D vector of numerics
        Some probability mass vector (this will not fail if hist_1 is not a probability mass but the intended use case is for PMFs).
    hist_2 : 2D vector of numerics of same shape as hist_1
        Second probability mass vector.
    dx: float
        Distance between bins
    axis : int, optional
        If hist_1 is multi-dimensional, the axis to sum along. The default is None.

    Returns
    -------
    vector of floats
        KL distance between hist_1 and hist_2.

    """
    def _helper(element):
        # Avoids issues with taking the log of 0.
        if element == 0:
            return 0
        else:
            return np.log(element) # Can't use branchless programming unfortunately because 0*inf = nan
    log_ish = np.vectorize(_helper)
    assert len(hist_1) == len(hist_2)
    return np.sum(hist_1*log_ish(hist_1) - hist_1*log_ish(hist_2), axis=axis)*dx # don't use hist_1/hist_2 in the log in case of zero bins

def kolmogorov_smirnov(hist_1, hist_2, dx, axis=None):
    """
    Calculate the Kolmogorov-Smirnov statistic for histograms hist_1 and hist_2.

    Parameters
    ----------
    hist_1 : vector of numerics
        Probability density vector.
    hist_2 : vector of numerics of same shape as hist_1
        Another probability density vector.
    dx: float
        Distance between bins
    axis : int, optional
        axis to cumsum along. The default is None.

    Returns
    -------
    vector of numerics
        Kolmogorov-Smirnov statistic between both histograms.

    """
    CDF1, CDF2 = np.cumsum(hist_1*dx, axis=axis), np.cumsum(hist_2*dx, axis=axis) # Compute the CDFs
    return np.max(np.abs(CDF2-CDF1), axis=axis)

def KS_empirical(data, CDF, x):
    """
    Use the empirical CDF generated from data to compare it to a proper CDF. Uses scipy's methods. CDF is an array --- we generate an interpolating function to pass to scipy.

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    CDF : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    CDF_interpolated = lambda t: np.interp(t, x, CDF)
    return CDF_interpolated