from __future__ import division
import numpy as np
import nanfunctions as nf

__all__ = ['tstatistic',
           'nantstatistic']

def tstatistic(a,b,axis=0,equal_var=True):
    """Computes a two-sample t-statistic on a and b along the specific axis
    Code is lifted from scipy.stats.ttest_ind except that there is no
    calculation of the associated p-value

    Parameters
    ----------
    a,b : ndarray with shapes equal along all dims except axis
        Input data for the calculation.
    axis : int
        Specify the axis along which the statistic will be computed.
    equal_var : bool
        Specify if the statistic will use a pooled estimate of the variance (True)
        or if it will make no assumption about equal variance (False).

    Returns
    -------
    t : ndarray with one less dimension
    """

    v1 = np.var(a, axis, ddof=1)
    v2 = np.var(b, axis, ddof=1)
    n1 = a.shape[axis]
    n2 = b.shape[axis]

    if equal_var:
        df = n1 + n2 - 2
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(df)
        denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        df = ((vn1 + vn2)**2) / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

        # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
        # Hence it doesn't matter what df is as long as it's not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = np.sqrt(vn1 + vn2)

    d = np.mean(a, axis) - np.mean(b, axis)
    t = np.divide(d, denom)
    return t

def nantstatistic(a,b,axis=0,equal_var=True):
    """Computes a two-sample t-statistic on a and b along the specific axis
    Uses nan* functions which can be slightly slower.
    Code is lifted from scipy.stats.ttest_ind except that there is no
    calculation of the associated p-value

    Parameters
    ----------
    a,b : ndarray with shapes equal along all dims except axis
        Input data for the calculation.
    axis : int
        Specify the axis along which the statistic will be computed.
    equal_var : bool
        Specify if the statistic will use a pooled estimate of the variance (True)
        or if it will make no assumption about equal variance (False).

    Returns
    -------
    t : ndarray with one less dimension
    """

    v1 = nf.nanvar(a, axis, ddof=1)
    v2 = nf.nanvar(b, axis, ddof=1)
    n1 = a.shape[axis]
    n2 = b.shape[axis]

    if equal_var:
        df = n1 + n2 - 2
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(df)
        denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        df = ((vn1 + vn2)**2) / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

        # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
        # Hence it doesn't matter what df is as long as it's not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = np.sqrt(vn1 + vn2)

    d = nf.nanmean(a, axis) - nf.nanmean(b, axis)
    t = np.divide(d, denom)
    return t