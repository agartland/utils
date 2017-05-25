
import numpy as np

__all__ = ['tstatistic',
           'nantstatistic',
           'diffmean']

def diffmean(a, b, axis=0):
    """Difference of means statistic.

    Parameters
    ----------
    a,b : ndarray with shapes equal along all dims except axis
        Input data for the calculation.
    axis : int
        Specify the axis along which the statistic will be computed.

    Returns
    -------
    dm : ndarray with one less dimension"""
    return a.mean(axis=axis) - b.mean(axis=axis)

def tstatistic(a, b, axis=0, equal_var=True):
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
    t : ndarray with one less dimension"""
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

def nantstatistic(a, b, axis = 0, equal_var = True):
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

    v1 = np.nanvar(a, axis, ddof=1)
    v2 = np.nanvar(b, axis, ddof=1)
    n1 = a.shape[axis]
    n2 = b.shape[axis]

    if equal_var:
        df = n1 + n2 - 2
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / np.float(df)
        denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        df = ((vn1 + vn2)**2) / ((vn1**2) / (n1 - 1) + (vn2**2) / (n2 - 1))

        # If df is undefined, variances are zero (assumes n1 > 0 & n2 > 0).
        # Hence it doesn't matter what df is as long as it's not NaN.
        df = np.where(np.isnan(df), 1, df)
        denom = np.sqrt(vn1 + vn2)

    d = np.nanmean(a, axis) - np.nanmean(b, axis)
    t = np.divide(d, denom)
    return t

'''TODO: implement in numba so they can be used in a numbized permutation test
"""Attempt to import numba and define numba compiled versions of these functions"""
import os
import sys
try:
    import numba as nb
    print 'mytstats: Successfully imported numba version %s' % (nb.__version__)
    NB_SUCCESS = True
except OSError:
    try:
        """On Windows it is neccessary to be on the same drive as the LLVM DLL
        in order to import numba without generating a "Windows Error 161: The specified path is invalid."""
        curDir = os.getcwd()
        targetDir = os.path.splitdrive(sys.executable)[0]
        os.chdir(targetDir)
        import numba as nb
        print 'mytstats: Successfully imported numba version %s' % (nb.__version__)
        NB_SUCCESS = True
    except OSError:
        NB_SUCCESS = False
        print 'mytstats: Could not load numba\n(may be a path issue try starting python in C:\\)'
    finally:
        os.chdir(curDir)
except ImportError:
    NB_SUCCESS = False
    print 'mytstats: Could not load numba'

"""TODO: (1) Test numba functions (this code is just copied from above)
             if the function can just be decorated then do that,
             but i think it may need to be modified

         (2) Add numba permutation function that utlizes these statistics
             (this is where the speed-up will happen)"""

if NB_SUCCESS and False:
    __all__.extend(['nb_tstatistic', 'nb_nantstatistic'])

    @nb.jit(nb.float64[:](nb.float64[:],nb.float64[:], nb.int32, nb.boolean), nopython = True)
    def nb_tstatistic(a, b, axis, equal_var):
        v1 = np.var(a, axis, ddof=1)
        v2 = np.var(b, axis, ddof=1)
        n1 = a.shape[axis]
        n2 = b.shape[axis]

        if equal_var:
            df = n1 + n2 - 2
            svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / np.float(df)
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

    @nb.jit(nb.float64[:](nb.float64[:],nb.float64[:], nb.int, nb.boolean), nopython = True)
    def nb_nantstatistic(a, b, axis, equal_var):
        v1 = np.nanvar(a, axis, ddof=1)
        v2 = np.nanvar(b, axis, ddof=1)
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

        d = np.nanmean(a, axis) - np.nanmean(b, axis)
        t = np.divide(d, denom)
        return t

'''