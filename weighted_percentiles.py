import numpy as np

__all__ = ['weighted_percentiles']

def weighted_percentiles(a, percentiles, weights=None):
    """Compute weighted percentiles by using interpolation of the weighted ECDF.
    
    Parameters
    ----------
    a : np.ndarray
        Vector of data for computing quantiles
    percentiles : np.ndarray
        Vector of percentiles in [0, 100]
    weights : np.ndarray
        Vector of non-negative weights. Not required to sum to one.

    Returns
    -------
    percentiles : np.ndarray"""
    
    a = np.array(a)
    percentiles = np.array(percentiles)
    quantiles = percentiles / 100.
    
    if weights is None:
        weights = np.ones(len(a))
    else:
        weights = np.array(weights)
    
    assert np.all(weights > 0), 'Weights must be > 0'
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'Percentiles must be in [0, 100]'

    sorti = np.argsort(a)
    a = a[sorti]
    weights = weights[sorti]

    """Two definitions for the weighted eCDF. See _plotSolutions() below for a comparison.
    Note that there are also several options in R for copmuting a weighted quantile,
    but I did not fully understand the motivation for each. The chosen option here was intuitive to me
    and agreed well with the empirical solution below.
    https://github.com/harrelfe/Hmisc/R/wtd.stats.s"""
    
    # ecdf = np.cumsum(weights) / weights.sum()
    ecdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)

    return np.interp(quantiles, ecdf, a)

def _empirical_weighted_percentiles(a, percentiles, weights=None, N=1000):
    a = np.array(a)
    percentiles = np.array(percentiles)
    quantiles = percentiles / 100.
    
    if weights is None:
        weights = np.ones(len(a))
    else:
        weights = np.array(weights)
    
    assert np.all(weights > 0), 'Weights must be > 0'
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'Percentiles must be in [0, 100]'

    sorti = np.argsort(a)
    a_sort = a[sorti]
    w_sort = w[sorti]

    apop = np.zeros(int(N*1.2))
    starti = 0
    for ai, wi in zip(a_sort, w_sort/w_sort.sum()):
        n = int(np.round(wi * N))
        apop[starti: starti + n] = ai
        starti += n
    apop = apop[:starti]
    return np.percentile(apop, percentiles)

def _plotSolutions():
    import matplotlib.pyplot as plt
    #a = np.random.randn(10)
    #w = np.abs(np.random.randn(10))

    a = np.array([-1.06151426,  0.55011175,  0.22815913,  0.62298578, -0.606928  ,
            0.67393622,  0.24912888, -1.19431307,  0.11873281,  0.32038022])
    w = np.array([ 0.6587839 ,  0.28195309,  0.20423927,  0.73463671,  0.72642352,
            0.29409455,  0.60123757,  3.03307223,  0.92969147,  0.46556024])

    quantiles = np.linspace(0, 1, 101)
    percentiles = quantiles * 100

    res1 = weighted_percentiles(a, percentiles, weights=w)
    res2 = _empirical_weighted_percentiles(a, percentiles, weights=w, N=10000)

    sorti = np.argsort(a)
    a_sort = a[sorti]
    w_sort = w[sorti]

    Rres = np.array([-1.1943131, -1.1943131, -1.1943131, -1.1479638, -0.7409240, -0.2696073,  0.1393113,  0.2296718,  0.4524151, 0.6350469,  0.6703540])

    ecdf1 = np.cumsum(w_sort)/np.sum(w_sort)
    ecdf2 = (np.cumsum(w_sort) - 0.5 * w_sort)/np.sum(w_sort)

    plt.figure(50)
    plt.clf()
    plt.plot(a_sort, ecdf1, '-ok', label='ECDF simple')
    plt.plot(a_sort, ecdf2, '-sk', label='ECDF complex')
    plt.plot(Rres, np.linspace(0, 1, 11), '-sr', label='R')
    for res,l in zip([res1, res2], ['interp ecdf', 'emp']):
        plt.plot(res, percentiles/100, '--.', label=l)
    plt.legend()

