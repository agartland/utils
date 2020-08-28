import numpy as np
from scipy import stats

__all__ = ['unique_rows',
            'argrank',
            'mnmx',
            'mnmxi',
            'argsort_rows',
            'untangle',
            'first_nonzero']

def unique_rows(a, return_index=False, return_inverse=False, return_counts=False):
    """Performs np.unique on whole rows of matrix a using a "view".
    See http://stackoverflow.com/a/16971324/74616"""
    try:
        dummy, uniqi, inv_uniqi, counts = np.unique(a.view(a.dtype.descr * a.shape[1]), return_index = True, return_inverse = True, return_counts = True)
        out = [a[uniqi,:]]
        if return_index:
            out.append(uniqi)
        if return_inverse:
            out.append(inv_uniqi)
        if return_counts:
            out.append(counts)
    except ValueError:
        """This doesn't work with all types of data, so fall back on slow-simple algo"""
        s = set()
        for i in range(a.shape[0]):
            s.add(tuple(a[i,:].tolist()))
        out = [np.array([row for row in s])]
    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

def untangle(y_orig, y_tangled, rtol=0.001):
    """Assuming two vectors contain the same elements (to a
    relative tolerance), but in different order, find the
    indices into y_tangled that "untangle" it so that it matches
    the original data.

    Parameters
    ----------
    y_orig : np.ndarray
    y_tangled : np.ndarray

    Returns
    -------
    untanglei : np.ndarray
        Indices into y_tangled such that: y_orig == y_tangled[untanglei]"""

    yo = np.array(y_orig)
    y = np.array(y_tangled)
    assert len(yo) == len(y)
    
    yo_sorti = np.argsort(yo)
    y_sorti = np.argsort(y)
    
    yo_sorted = yo[yo_sorti]
    y_sorted = y[y_sorti]
    assert np.allclose(yo_sorted, y_sorted, rtol=rtol)

    yo_unsorti = np.argsort(yo_sorti)
    
    untanglei = y_sorti[yo_unsorti]
    newy = y[untanglei]
    # assert newy == yo
    return untanglei

def argrank(vec, method='average'):
    """Return the rank (0 based) of the elements in vec"""
    return stats.rankdata(vec, method=method)
    '''sorti = np.argsort(vec)
    ranks = np.empty(len(vec), int)
    try:
        ranks[sorti] = np.arange(len(vec))
    except IndexError:
        ranks[sorti.values] = np.arange(len(vec))
    return ranks'''
def mnmx(arr):
    """Shortcut to return both the min and the max of arr"""
    return (np.min(arr), np.max(arr))
def mnmxi(arr):
    """Shortcut to return both the argmax and argmin of arr"""
    return (np.argmin(arr), np.argmax(arr))
def argsort_rows(a):
    """Performs argsort on whole rows of matrix a.
    See unique_rows and http://stackoverflow.com/a/16971324/74616"""
    sorti = np.argsort(a.view(a.dtype.descr * a.shape[1]), axis=0).flatten()
    return sorti
    

def first_nonzero(mask, axis, invalid_val=-1):
    """SO: https://stackoverflow.com/questions/47269390/numpy-how-to-find-first-non-zero-value-in-every-column-of-a-numpy-array"""
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)