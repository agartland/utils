import numpy as np
import pandas as pd
from scipy import stats

__all__ = ['bootci_pd',
           'permtest_pd']

def bootci_pd(df, statfunction, alpha=0.05, n_samples=10000, method='bca'):
    """Estimate bootstrap CIs for a statfunction that operates along the rows of
    a pandas.DataFrame and return a dict of results

    This is about 10x slower than using scikits.bootstrap.ci for a statistic
    doesn't require resampling the whole DataFrame. However, if the statistic
    requires the whole DataFrame or you are computing many statistics on the
    same DataFrame that all require CIs, then this function may be efficient.

    Parameters
    ----------
    df : pd.DataFrame
        Data that will be passed to statfunction as a single parameter.
    statfunction : function
        Function that should operate along the rows of df and return a dict
    alpha : float [0, 1]
        Specify CI: [alpha/2, 1-alpha/2]
    n_samples : int
        Number of bootstrap samples.
    method : str
        Specify bias-corrected and accelerated ("bca") or percentile ("pi")
        bootstrap.

    Returns
    -------
    cis : pd.Series [est, lcl, ucl]
        Point-estimate and CI of statfunction of df"""
    alphas = np.array([alpha/2, 1-alpha/2])

    # The value of the statistic function applied just to the actual data.
    res = pd.Series(statfunction(df))
    
    #st = time.time()
    boot_res = []
    for i in range(n_samples):
        # rind = np.random.randint(df.shape[0], size=df.shape[0])
        # boot_res.append(statfunction(df.iloc[rind]))
        boot_res.append(statfunction(df.sample(frac=1, replace=True)))
    boot_res = pd.DataFrame(boot_res)
    #print(time.time() - st)

    # Percentile Interval Method
    if method == 'pi':
        avals = np.tile(alphas, (boot_res.shape[1], 1)).T
    # Bias-Corrected Accelerated Method
    elif method == 'bca':
        ind = np.ones(df.shape[0], dtype=bool)
        jack_res = []
        for i in range(df.shape[0]):
            ind[i] = False
            jack_res.append(statfunction(df.loc[ind]))
            ind[i] = True

        jack_res = pd.DataFrame(jack_res)
        jmean = np.nanmean(jack_res, keepdims=True, axis=0)
        bca_accel = np.nansum((jmean - jack_res.values)**3, axis=0) / (6.0 * np.nansum((jmean - jack_res.values)**2, axis=0)**1.5)

        """The bias correction value"""
        z0 = stats.distributions.norm.ppf( (np.sum(boot_res.values < res.values[None, :], axis=0)) / np.sum(~np.isnan(boot_res.values), axis=0) )
        zs = z0 + stats.distributions.norm.ppf(alphas).reshape(alphas.shape + (1,) * z0.ndim)
        avals = stats.distributions.norm.cdf(z0 + zs / (1 - bca_accel * zs))

    non_nan_ind = ~np.isnan(boot_res)
    nvals = np.round((np.sum(non_nan_ind.values, axis=0) - 1) * avals).astype(int)
    nvals = pd.DataFrame(nvals.T, columns=['%1.3f' % a for a in alphas], index=boot_res.columns)

    if np.any(np.isnan(nvals)):
        print('Nan values for some stats suggest there is no bootstrap variation.')
        print(res.head(10))
    
    cis = pd.DataFrame(np.zeros((len(boot_res.columns), len(avals) + 1)), index=boot_res.columns, columns=['est'] + ['%1.3f' % a for a in alphas])
    for i,col in enumerate(boot_res.columns):
        boot_res.values[:, i].sort()
        cis.loc[col, 'est'] = res[col]
        cis.loc[col, ['%1.3f' % a for a in alphas]] = boot_res[col].values[nvals.loc[col]]

    if np.any(nvals < 10) or np.any(nvals > n_samples-10):
        print('Extreme samples used: results unstable')
        print(nvals)

    return cis

def permtest_pd(df, statfunction, perm_cols, n_samples=9999, alternative='two-sided'):
    """Estimate a p-value for the statfunction against the permutation null.

    Parameters
    ----------
    df : pd.DataFrame
        Observed data required as sole input for statfunction.
    statfunction : function
        Operates on df and returns a scalar statistic.
    perm_cols : list of str
        Columns that need to be permuted in df to generate a null dataset
    n_samples : int
        Number of permutations to test
    alternative : str
        Specify a "two-sided" test or one that tests that the observed data is "less" than
        or "greater" than the null statistics.

    Returns
    -------
    pvalue : float"""

    n_samples = int(n_samples)
    
    tmp = df.copy()
    samples = np.zeros(n_samples)
    for sampi in range(n_samples):
        rind = np.random.permutation(df.shape[0])
        tmp.loc[:, perm_cols] = tmp.loc[:, perm_cols].values[rind, :]
        samples[sampi] = statfunction(tmp)
    
    if alternative == 'two-sided':
        pvalue = ((np.abs(samples) > np.abs(statfunction(df))).sum() + 1) / (n_samples + 1)
    elif alternative == 'greater':
        pvalue = ((samples > statfunction(df)).sum() + 1) / (n_samples + 1)
    elif alternative == 'less':
        pvalue = ((samples < statfunction(df)).sum() + 1) / (n_samples + 1)

    return pvalue


def _test_permtest_pd(effect=0.5, n_samples=9999):
    from scipy import stats
    import time

    df = pd.DataFrame(np.random.randn(100, 5))
    df.loc[:, 0] = np.random.randint(2, size=df.shape[0])

    df.loc[df[0] == 0, 1] = df.loc[df[0] == 0, 1] + effect

    def func(d):
        return np.mean(d.loc[d[0] == 0, 1]) - np.mean(d.loc[d[0] == 1, 1])

    st = time.time()
    res = permtest_pd(df, func, perm_cols=[0], n_samples=n_samples)
    et = (time.time() - st)
    print(res)
    print('Time: %1.2f sec' % et)

    print(stats.ttest_ind(df.loc[df[0] == 0, 1], df.loc[df[0] == 1, 1]))


def _test_bootci_pd(n_samples=10000, method='bca'):
    import scikits.bootstrap as boot
    import time

    df = pd.DataFrame(np.random.randn(100, 5))
    def func(d):
        return {'MeanA':d[0].mean(), 'MedianB':np.median(d[1])}
    def func2(d):
        return d.mean()
    st = time.time()
    res = bootci_pd(df, func, alpha=0.05, n_samples=n_samples, method=method)
    et = (time.time() - st)
    print(res)
    print('Time: %1.2f sec' % et)

    st = time.time()
    a = boot.ci(df[0].values, statfunction=np.mean, n_samples=n_samples, method=method)
    b = boot.ci(df[1].values, statfunction=np.median, n_samples=n_samples, method=method)
    et = (time.time() - st)

    print('MeanA', a)
    print('MedianB', b)
    print('Time: %1.2f sec' % et)

