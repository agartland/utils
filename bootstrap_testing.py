"""
Set of functions that help integrate bootstrap statistical tests into my workflow.

The code is based on chapters from the seminal text:
    Efron B, Tibshirani R. 1993. An introduction to the bootstrapMonographs on statistics and applied probability. Chapman & Hall, New York.

The confidence interval computations is simply a wrapper around the scikits_bootstrap package in pypi:
    scikits-bootstrap (Constantine Evans)
    https://pypi.python.org/pypi/scikits.bootstrap
"""


import numpy as np
from scikits_bootstrap import ci
from numpy.random import randint
from functools import partial

__all__ = ['permTwoSampTest',
           'bootstrapTwoSampTest',
           'bootstrapOneSampTest',
           'bootstrapSE',
           'bootstrapCI',
           'bootstrapPvalue',
           'bootstrapGeneral']

def permTwoSampTest(vec1,vec2,statFunc=None,nPerms=10000):
    """Uses group permutations to calculate a p-value for a
    two sample test for the difference in the mean.
    Optionally specify statFunc for other comparisons."""
    L1=len(vec1)
    L2=len(vec2)
    data=np.concatenate((array(vec1),array(vec2)))
    L=len(data)
    assert L==(L1+L2)

    if statFunc is None:
        statFunc = lambda v1,v2: mean(v1)-mean(v2)

    samples = np.zeros(nPerms)
    for sampi in np.arange(nPerms):
        inds = permutation(L)
        samples[sampi] = statFunc(data[inds[:L1]],data[inds[L1:]])
    return (abs(samples)>abs(statFunc(vec1,vec2))).sum()/nPerms

def bootstrapTwoSampTest(vec1,vec2,statFunc=None,nPerms=10000):
    """Uses a bootstrap to calculate a p-value for a
    two sample test for the difference in the mean.
    Optionally specify statFunc for other comparisons."""
    L1=len(vec1)
    L2=len(vec2)
    data=list(vec1)+list(vec2)
    L=len(data)
    assert L==(L1+L2)

    if statFunc is None:
        """Use studentized statistic with pooled variance instead for more accuracy
        (but assumes equal variances)"""
        statFunc = lambda v1,v2: (mean(v1)-mean(v2)) / (sqrt((sum((v1-mean(v1))**2) + sum((v2-mean(v2))**2))/(L1+L2-2)) * sqrt(1/L1+1/L2))
        #statFunc = lambda v1,v2: mean(v1)-mean(v2)

    samples = np.zeros(nPerms)
    for sampi in np.arange(nPerms):
        inds = randint(L,size=L)
        samples[sampi] = statFunc([data[i] for i in inds[:L1]],[data[i] for i in inds[L1:]])
    return (abs(samples)>abs(statFunc(vec1,vec2))).sum()/nPerms

def bootstrapPairedTwoSampTest(vec1,vec2,nPerms=10000):
    """Uses a bootstrap to calculate a p-value for a
    two sample paired test of H0: mean(vec1-vec2) != 0"""
    return bootstrapOneSampTest(vec1-vec2,nv=0,nPerms=nPerms)

def bootstrapOneSampTest(data,nv=0,nullTranslation=None,statFunc=None,nPerms=10000):
    """Uses a bootstrap to calculate a p-value for a
    one sample test of H0: mean(data) != nv

    Uses a t-statistic as is used in Efron and Tibshirani"""
    L=len(data)
    if statFunc is None:
        statFunc = lambda data: (np.mean(data)-nv)/(np.std(data)/np.sqrt(len(data)))
        """Could also default to use mean instead of tstat"""
        #statFunc = lambda data: mean(data)-nv
    if nullTranslation is None:
        nullTranslation = lambda data: data - np.mean(data) + nv
    
    nullDist = nullTranslation(data)
    samples = np.zeros(nPerms)
    for sampi in np.arange(nPerms):
        inds = randint(L,size=L)
        samples[sampi] = statFunc([nullDist[i] for i in inds])
    return (np.abs(samples)>np.abs(statFunc(data))).sum()/nPerms
def bootstrapPvalue(data,statFunc,alpha=0.05,nPerms=10000,returnNull=False):
    """Uses a bootstrap to compute a pvalue based on the pvalues of a statistic on the data.
    Neccessary for statistics for which it is not easy to specify a null value or translate the
    observed distribution to get a null distribution.
    Good for correlations or paired Wilcoxon tests etc.
    statFunc should return a p-value (uniform distribution for the null)
    Returns the fraction of bootstrap samples for which the pvalue < alpha
    (e.g. rejecting 95% of the bootstrap samples is a global pvalue = 0.05)"""
    L=len(data)
    
    pvalues = np.zeros(nPerms)
    for sampi in np.arange(nPerms):
        inds = randint(L,size=L)
        pvalues[sampi] = statFunc([data[i] for i in inds])
    if returnNull:
        return (pvalues>alpha).sum()/nPerms,pvalues
    else:
        return (pvalues>alpha).sum()/nPerms

def bootstrapGeneral(data,nv=0,statFunc=np.mean,nPerms=10000,returnNull=False):
    """Uses a bootstrap to compute a pvalue for a statistic on the data.
    Similar to bootstrapOneSampTest() which is good when the statistic is a mean or tstat.
    This function is good for correlations when nv can be 0
    NOTE: signs might get messed up if nv > obs"""
    L=len(data)
    
    samples = np.zeros(nPerms)
    for sampi in np.arange(nPerms):
        inds = randint(L,size=L)
        samples[sampi] = statFunc([data[i] for i in inds])
    if returnNull:
        return 2*(samples<nv).sum()/nPerms,samples
    else:
        return 2*(samples<nv).sum()/nPerms

def bootstrapSE(data,statFunc,nPerms=1000):
    """Bootstrap estimate of the standard error of the specified statistic"""
    L = len(data)
    samples = np.zeros(nPerms)
    for sampi in np.arange(nPerms):
        inds = randint(L,size=L)
        samples[sampi] = statFunc([data[i] for i in inds])
    return samples.std()
    
def bootstrapCI(data, statFunc=None, alpha=0.05, nPerms=10000, output='lowhigh'):
    """Wrapper around a function in the scikits_bootstrap module:
        https://pypi.python.org/pypi/scikits.bootstrap

    Parameters
    ----------
    data : np.ndarray
        Data for computing the confidence interval.
    statFunc : function
        Should take data and operate along axis=0
    alpha : float
        Returns the [alpha/2, 1-alpha/2] percentile confidence intervals.
    nPerms : int
    output : str
        Use 'lowhigh' or 'errorbar', for matplotlib errorbars"""
    if statFunc is None:
        statFunc = partial(np.nanmean, axis=0)
    try:
        out = ci(data=data, statfunction=statFunc, alpha=alpha, n_samples=nPerms, output='lowhigh')
    except IndexError:
        shp = list(data.shape)
        shp[0] = 2
        out = np.nan * np.ones(shp)
    
    if output == 'errorbar':
        mu = statFunc(data)
        shp = list(out.shape)
        
        out[0,:] = out[0,:] - mu
        out[1,:] = mu - out[1,:]
        out = np.reshape(out, shp)
    return out

"""Experiments in parallelism
from joblib import Parallel, delayed
Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(10))

def testStatFunc(data, ind):
    half = int(np.floor(len(ind)/2.))
    return np.mean(data[ind[:half]]) - np.mean(data[ind[half:]])

def parPermTest(data, statFunc, nperms=100, seed=110820, n_jobs=1):
    n = data.shape[0]
    tmpFunc = partial(statFunc, data)
    obs = tmpFunc(range(n))
    
    np.random.seed(seed)
    
    if ncpus == 1:
        robs = np.zeros(nperms)
        for i in range(nperms):
            robs[i] = tmpFunc(np.random.permutation(n))
    else:
        robs = Parallel(n_jobs=n_jobs)(delayed(tmpFunc)(np.random.permutation(n)) for i in range(nperms))
    
    pvalue = 2 * ((robs >= obs).sum() + 1) / (nperms + 1)
    return pvalue, obs, robs

data = np.concatenate(())

"""