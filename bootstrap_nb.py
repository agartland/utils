import numpy as np
import pandas as pd
from scipy import stats
from numba import jit, prange

__all__ = ['bootci_nb',
           'permtest_nb']

@jit(nopython=True, parallel=True, error_model='numpy')
def _bootstrap_jit(dat, statfunction, nstraps, nstats):
    n = dat.shape[0]
    res = np.zeros((nstraps, nstats))
    for booti in range(nstraps):
        rind = np.random.choice(np.arange(n), n)
        res[booti, :] = statfunction(dat[rind, :])
    """Sort each stat independently"""
    for stati in range(nstats):
        res[:, stati].sort()
    return res
'''
@jit(nopython=True, parallel=True, error_model='numpy')
def _jackknife_jit(dat, statfunction, nstats):
    n = dat.shape[0]
    jstats = np.zeros((n, nstats))
    #jind = np.ones(n, dtype=np.bool_)
    for i in prange(n):
        jind = np.ones(n, dtype=np.bool_)
        jind[i] = False
        jstats[i, :] = statfunction(dat[jind, :])
        #jind[i] = True
    
    bca_accel = np.zeros(nstats)
    for coli in range(nstats):
        jmean = np.nanmean(jstats[:, coli])
        bca_accel[coli] = np.nansum((jmean - jstats[:, coli])**3) / (6.0 * np.nansum((jmean - jstats[:, coli])**2)**1.5)
    return bca_accel
'''
@jit(nopython=True, parallel=True, error_model='numpy')
def _jackknife_jit(dat, statfunction, nstats):
    n = dat.shape[0]
    jstats = np.zeros((n, nstats))
    jind = np.ones(n, dtype=np.bool_)
    for i in range(n):
        jind[i] = False
        jstats[i, :] = statfunction(dat[jind, :])
        jind[i] = True
    
    bca_accel = np.zeros(nstats)
    for coli in range(nstats):
        jmean = np.nanmean(jstats[:, coli])
        bca_accel[coli] = np.nansum((jmean - jstats[:, coli])**3) / (6.0 * np.nansum((jmean - jstats[:, coli])**2)**1.5)
    return bca_accel



def bootci_nb(dat, statfunction, alpha=0.05, n_samples=10000, method='bca'):
    """Estimate bootstrap CIs for a statfunction that operates along the rows of
    a np.ndarray matrix and returns a np.ndarray vector of results.

    Parameters
    ----------
    dat : np.ndarray
        Data that will be passed to statfunction as a single parameter.
    statfunction : function
        Function that should operate along the rows of dat and return a vector
    alpha : float [0, 1]
        Specify CI: [alpha/2, 1-alpha/2]
    n_samples : int
        Number of bootstrap samples.
    method : str
        Specify bias-corrected and accelerated ("bca") or percentile ("pi")
        bootstrap.

    Returns
    -------
    cis : np.ndarray [est, lcl, ucl] x [nstats]
        Point-estimate and CI of statfunction of dat"""

    ostat = statfunction(dat)
    nstats = len(ostat)

    alphas = np.array([alpha/2, 1-alpha/2])
    
    """boot_res.shape --> (n_samples, nstats)"""
    boot_res = _bootstrap_jit(dat, statfunction, nstraps=n_samples, nstats=nstats)
    
    if method == 'pi':
        """Percentile Interval Method
        avals.shape --> (2, nstats)"""
        avals = np.tile(alphas, (boot_res.shape[1], 1)).T
    elif method == 'bca':
        """Bias-Corrected Accelerated Method
        bca_accel.shape --> (nstats, )"""
        bca_accel = _jackknife_jit(dat, statfunction, nstats)

        z0 = stats.distributions.norm.ppf( (np.sum(boot_res < ostat[None, :], axis=0)) / np.sum(~np.isnan(boot_res), axis=0) )
        zs = z0[None, :] + stats.distributions.norm.ppf(alphas).reshape(alphas.shape + (1,) * z0.ndim)
        avals = stats.distributions.norm.cdf(z0[None, :] + zs / (1 - bca_accel[None, :] * zs))

    non_nan_ind = ~np.isnan(boot_res)
    nvals = np.round((np.sum(non_nan_ind, axis=0) - 1) * avals).astype(int)
    
    """cis.shape --> (nstats, 3)"""
    cis = np.zeros((boot_res.shape[1], len(avals) + 1))
    for i in range(boot_res.shape[1]):
        cis[i, 0] = ostat[i]
        if np.all(np.isnan(avals[:, i])):
            print('No bootstrap variation in stat %d: LCL = UCL = observed stat' % (i))
            cis[i, 1:1+len(alphas)] = ostat[i] * np.ones(len(alphas))
        else:
            cis[i, 1:1+len(alphas)] = boot_res[nvals[:, i], i]
            if np.any(nvals[:, i] < 10) or np.any(nvals[:, i] > n_samples-10):
                print('Extreme samples used for stat %d: [%d, %d]. Results unstable.' % (i, nvals[0,i], nvals[1,i]))
    return cis


@jit(nopython=True, parallel=True, error_model='numpy')
def _perm_jit(d, sf, pcs, n):
    res = sf(d)
    samples = np.zeros((len(res), n))
    """Using prange here means we have to make a copy of d inside each loop
    Cost is memory, but this should be fine with reasonably sized matrices.
    Speed up is about 10x"""
    for sampi in prange(n):
        d_copy = d.copy()
        rind = np.random.permutation(d_copy.shape[0])
        for coli in pcs:
            d_copy[:, coli] = d_copy[rind, coli]
        samples[:, sampi] = sf(d_copy)
    return samples

def permtest_nb(dat, statfunction, perm_cols, n_samples=9999, alternative='two-sided'):
    """Estimate p-values for the results of statfunction against the permutation null.

    Parameters
    ----------
    dat : np.ndarray matrix
        Observed data required as sole input for statfunction.
    statfunction : function
        Operates on dat and returns a vector of statistics.
    perm_cols : array of indices
        Columns that need to be permuted in dat to generate a null dataset
    n_samples : int
        Number of permutations to test
    alternative : str
        Specify a "two-sided" test or one that tests that the observed data is "less" than
        or "greater" than the null statistics.

    Returns
    -------
    pvalue : float"""
    
    samples = _perm_jit(dat.copy(), statfunction, np.array(perm_cols, dtype=np.int), int(n_samples))    
    if alternative == 'two-sided':
        #pvalues = ((np.abs(samples) > np.abs(statfunction(dat)[None, :])).sum(axis=1) + 1) / (n_samples + 1)
        pvalues = ((np.abs(samples) > np.abs(statfunction(dat)[:, None])).sum(axis=1) + 1) / (n_samples + 1)
    elif alternative == 'greater':
        pvalues = ((samples > statfunction(dat)[None, :]).sum(axis=1) + 1) / (n_samples + 1)
    elif alternative == 'less':
        pvalues = ((samples < statfunction(dat)[None, :]).sum(axis=1) + 1) / (n_samples + 1)
    return pvalues

def _test_permtest(effect=0.5, n_samples=9999):
    from scipy import stats
    import time

    dat = np.random.randn(1000, 5)
    dat[:, 0] = np.random.randint(2, size=dat.shape[0])

    dat[dat[:, 0] == 0, 1] = dat[dat[:, 0] == 0, 1] + effect

    @jit(nopython=True)
    def func(d):
        return np.array([np.mean(d[d[:, 0] == 0, 1]) - np.mean(d[d[:, 0] == 1, 1])])

    res = func(dat)
    st = time.time()
    res = permtest_nb(dat, func, perm_cols=[0], n_samples=n_samples)
    et = (time.time() - st)
    print(res)
    print('Time: %1.2f sec' % et)

    print(stats.ttest_ind(dat[dat[:, 0] == 0, 1], dat[dat[:, 0] == 1, 1]))


def _test_bootci(n_samples=10000, method='bca'):
    import scikits.bootstrap as boot
    import time

    np.random.seed(110820)
    dat = np.random.randn(1000, 5)
    
    @jit(nopython=True)
    def func(d):
        return np.array([np.mean(d[:, 0]), np.median(d[:, 1]), np.max(d[:, 2])])

    st = time.time()
    res = bootci_nb(dat, func, alpha=0.05, n_samples=n_samples, method=method)
    et = (time.time() - st)
    print(res)
    print('Time: %1.2f sec' % et)

    st = time.time()
    a = boot.ci(dat[:, 0], statfunction=np.mean, n_samples=n_samples, method=method)
    b = boot.ci(dat[:, 1], statfunction=np.median, n_samples=n_samples, method=method)
    c = boot.ci(dat[:, 2], statfunction=np.max, n_samples=n_samples, method=method)
    et = (time.time() - st)

    print('Mean_0', a)
    print('Median_1', b)
    print('Median_2', c)
    print('Time: %1.2f sec' % et)