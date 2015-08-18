import numpy as np

__all__ = ['unique_rows',
            'argrank',
            'mnmx',
            'mnmxi',
            'argsort_rows']

def unique_rows(a, return_index = False, return_inverse = False, return_counts = False):
    """Performs np.unique on whole rows of matrix a using a "view".
    See http://stackoverflow.com/a/16971324/74616"""
    try:
        dummy,uniqi,inv_uniqi,counts = np.unique(a.view(a.dtype.descr * a.shape[1]), return_index = True, return_inverse = True, return_counts = True)
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
    return tuple(out)

def argrank(vec):
    """Return the rank (0 based) of the elements in vec"""
    sorti = np.argsort(vec)
    ranks = np.empty(len(vec), int)
    try:
        ranks[sorti] = np.arange(len(vec))
    except IndexError:
        ranks[sorti.values] = np.arange(len(vec))
    return ranks
def mnmx(arr):
    """Shortcut to return both the min and the max of arr"""
    return (np.min(arr),np.max(arr))
def mnmxi(arr):
    """Shortcut to return both the argmax and argmin of arr"""
    return (np.argmin(arr),np.argmax(arr))
def argsort_rows(a):
    """Performs argsort on whole rows of matrix a.
    See unique_rows and http://stackoverflow.com/a/16971324/74616"""
    sorti = np.argsort(a.view(a.dtype.descr * a.shape[1]),axis=0).flatten()
    return sorti