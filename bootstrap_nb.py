import numpy as np
import pandas as pd
from scipy import stats
from numba import jit

__all__ = ['bootci_nb',
           'permtest_nb']

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

    ostat = statfunction(dat)

    alphas = np.array([alpha/2, 1-alpha/2])
    boot_res = _bootstrap_jit(dat, statfunction, nstraps=n_samples, nstats=len(ostat))

    # Percentile Interval Method
    if method == 'pi':
        avals = np.tile(alphas, (boot_res.shape[1], 1)).T
    # Bias-Corrected Accelerated Method
    elif method == 'bca':
        # The value of the statistic function applied just to the actual data.
        ostat = statfunction(dat)
        bca_accel = _jackknife_jit(dat, statfunction, len(ostat))

        """The bias correction value"""
        z0 = stats.distributions.norm.ppf( (np.sum(boot_res < ostat[None, :], axis=0)) / np.sum(~np.isnan(boot_res), axis=0) )
        zs = z0 + stats.distributions.norm.ppf(alphas).reshape(alphas.shape + (1,) * z0.ndim)
        avals = stats.distributions.norm.cdf(z0 + zs / (1 - bca_accel * zs))

    non_nan_ind = ~np.isnan(boot_res)
    nvals = np.round((np.sum(non_nan_ind, axis=0) - 1) * avals).astype(int)

    if np.any(np.isnan(nvals)):
        print('Nan values for some stats suggest there is no bootstrap variation.')
        print(ostat[:10, :])
    
    cis = np.zeros((boot_res.shape[1], len(avals) + 1))
    for i in range(boot_res.shape[1]):
        cis[i, 0] = ostat[i]
        cis[i, 1:1+len(alphas)] = boot_res[nvals[i], i]

    if np.any(nvals < 10) or np.any(nvals > n_samples-10):
        print('Extreme samples used: results unstable')
        print(nvals)

    return ostat, cis


def permtest_nb(dat, statfunction, perm_cols, n_samples=9999, alternative='two-sided'):
    """Estimate a p-value for the statfunction against the permutation null.

    Parameters
    ----------
    dat : np.ndarray matrix
        Observed data required as sole input for statfunction.
    statfunction : function
        Operates on dat and returns a scalar statistic.
    perm_cols : list of str
        Columns that need to be permuted in dat to generate a null dataset
    n_samples : int
        Number of permutations to test
    alternative : str
        Specify a "two-sided" test or one that tests that the observed data is "less" than
        or "greater" than the null statistics.

    Returns
    -------
    pvalue : float"""

    @jit(nopython=True, parallel=True)
    def _perm_jit(d_copy, sf, pcs, n):
        samples = np.zeros(n)
        for sampi in range(n):
            rind = np.random.permutation(dat.shape[0])
            for coli in pcs:
                d_copy[:, coli] = d_copy[rind, coli]
            samples[sampi] = sf(d_copy)
        return samples
    
    samples = _perm_jit(dat.copy(), statfunction, np.array(perm_cols, dtype=int), int(n_samples))    
    if alternative == 'two-sided':
        pvalue = ((np.abs(samples) > np.abs(statfunction(dat))).sum() + 1) / (n_samples + 1)
    elif alternative == 'greater':
        pvalue = ((samples > statfunction(dat)).sum() + 1) / (n_samples + 1)
    elif alternative == 'less':
        pvalue = ((samples < statfunction(dat)).sum() + 1) / (n_samples + 1)
    return pvalue

def _test_permtest(effect=0.5, n_samples=9999):
    from scipy import stats
    import time

    dat = np.random.randn(100, 5)
    dat[:, 0] = np.random.randint(2, size=dat.shape[0])

    dat[dat[:, 0] == 0, 1] = dat[dat[:, 0] == 0, 1] + effect

    @jit(nopython=True)
    def func(d):
        return np.mean(d[d[:, 0] == 0, 1]) - np.mean(d[d[:, 0] == 1, 1])

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
    dat = np.random.randn(100, 5)
    
    @jit(nopython=True)
    def func(d):
        return np.array([np.mean(d[:, 0]), np.median(d[:, 1])])

    st = time.time()
    res = bootci_nb(dat, func, alpha=0.05, n_samples=n_samples, method=method)
    et = (time.time() - st)
    print(res)
    print('Time: %1.2f sec' % et)

    st = time.time()
    a = boot.ci(dat[:, 0], statfunction=np.mean, n_samples=n_samples, method=method)
    b = boot.ci(dat[:, 1], statfunction=np.median, n_samples=n_samples, method=method)
    et = (time.time() - st)

    print('MeanA', a)
    print('MedianB', b)
    print('Time: %1.2f sec' % et)