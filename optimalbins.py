import numpy as np

__all__ = ['optimalBins', 'optimalBinSize']

def optimalBinSize(x):
    """Returns the optimal bin size for data in x"""
    interquartile = np.diff(np.prctile(x, [25, 75]))
    return 2. * interquartile * len(x)**(-1./3)

def optimalBins(x,factor=1):
    """Returns the edges of bins for an optimal histogram of the data X.
    factor widens the bins"""
    sz = optimalBinSize(x) * factor
    return np.arange(x.min(), x.max(), sz)