import numpy as np
from scipy import special

__all__ = ['jensen_shannon_divergence']

def jensen_shannon_divergence(a, b):
    """Compute Jensen-Shannon divergence between two categorical probability distributions.

    Lifted from github/scipy:
    https://github.com/luispedro/scipy/blob/ae9ad67bfc2a89aeda8b28ebc2051fff19f1ba4a/scipy/stats/stats.py

    Parameters
    ----------
    a : array-like
        possibly unnormalized distribution
        
    b : array-like
        possibly unnormalized distribution. Must be of same size as ``a``.
    
    Returns
    -------
    j : float
    
    See Also
    --------
    jsd_matrix : function
        Computes all pair-wise distances for a set of measurements"""
    
    a = np.asanyarray(a, dtype=float)
    b = np.asanyarray(b, dtype=float)
    a = a / a.sum()
    b = b / b.sum()
    m = (a + b)
    m /= 2.
    m = np.where(m, m, 1.)
    return 0.5 * np.sum(special.xlogy(a, a/m) + special.xlogy(b, b/m))
