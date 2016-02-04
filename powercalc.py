from __future__ import division
from scipy import stats
import numpy as np
import itertools
from functools import partial
from scikits_bootstrap import ci

__all_ = ['veEndpointsRequired',
          'eventRatePower',
          'magnitudePower',
          'powerBySim',
          'powerBySubsample',
          'fisherTestVector',
          'probGTEX',
          'eventCI',
          'spearmanCI',
          'correlationCI',
          'statDistByN',
          'sumAnyNBinom']

"""NOTE: Many of these were hacked together in very short time, but they code may still be useful for next time."""

try:
    """Attempt to use the fisher library (cython) if available (100x speedup)"""
    import fisher
    def fisherTest(tab,alternative='two-sided'):
        res = fisher.pvalue(tab[0][0],tab[0][1],tab[1][0],tab[1][1])
        OR = (tab[0][0] * tab[1][1]) / (tab[0][1] * tab[1][0])

        if alternative == 'two-sided':
            return (OR,res.two_tail)
        elif alternative == 'less':
            return (OR,res.left_tail)
        elif alternative == 'greater':
            return (OR,res.right_tail)
    print "Using Cython-powered Fisher's exact test"
except ImportError:
    print "Using scipy.stats Fisher's exact test (slow)"
    fisherTest = stats.fisher_exact

def veEndpointsRequired(rr0=1, rralt=0.5, targetpow=0.9, K=1, alpha=0.025):
    """Minimum number of infection endpoints to achieve given power
    of conditional test for ratio of binomial proportions for
    a two-sample problem with unequal sample sizes.

    Adapted from R code developed by Steve Self (FHCRC/UW, 2015)

    Note:  Minimum number is defined as the smallest n for which
           the power is no less that the target power for all
           greater values of n.  Because of the discretness of
           distributions, this value is different (larger) than
           the smallest n for which power is no less than the
           target value.

    Params
    ------
    rr0 : float [0,1]
        Relative-risk under the null hypothesis, H0
    rralt : float [0,1]
        Relative-risk under the alternative hypothesis (i.e. effect size)
    targetpow : float [0,1]
        Desired power.
    K : float
        Randomization ratio (e.g. K = 2 means 2:1 ratio of vaccine:placebo)
    alpha : float
        Size of 1-sided test

    Returns
    -------
    n : int
        Number of endpoints required (vaccine and placebo combined)
    xcrit : float
        Associated critical value for exact binomial test


    Example
    -------
    >> n,xcrit = veEndpointsRequired(rr0 = 0.5, rralt = 0.2, targetpow = 0.9, K = 2, alpha = 0.025)
    >> print 'N:', n
    N: 61
    """

    relRate0 = rr0 * K
    p0 = relRate0 / (1 + relRate0)

    rr = rralt * K
    p = rr / (1 + rr)

    curPower = 0
    """Start search at trials w/ > 5 infection endpts"""
    n = 5      
    """Find (approx?) upper bound for n"""
    while curPower < np.min(targetpow + 0.05,0.99):
        n = n + 1
        """Compute critical value for exact binomial test and power"""
        xcrit = stats.binom.ppf(alpha,n,p0) - 1
        curPower = stats.binom.cdf(xcrit,n,p)
    
    """Now search backwards for smallest n for which power is greater than target for all larger n"""
    while curPower > targetpow:     
        n = n - 1
        xcrit = stats.binom.ppf(alpha,n,p0) - 1
        curPower = stats.binom.cdf(xcrit,n,p)

    n = n + 1
    xcrit = stats.binom.ppf(alpha,n,p0) - 1
    return n, xcrit

def eventRatePower(rate, N, alpha=0.05,iterations=1e3,alternative='two-sided',rseed=820):
    """Use powerBySim() to compute power to detect a difference in event rate between two groups.
    Returns power, odds-ratio"""   

    """Funcs to generate vector of binary event data based on two rates"""
    dataFunc = [lambda N: np.random.rand(N) < rate[0], lambda N: np.random.rand(N) < rate[1]]
    testFunc = partial(fisherTestVector, alternative=alternative)
    return powerBySim(dataFunc, testFunc, N, alpha=alpha, iterations=iterations, rseed=rseed)

def magnitudePower(mu, N, sigma=None, CV=None, alpha=0.05, paired=False, iterations=1e3, rseed=820):
    """Use powerBySim() to compute power to detect difference in two
    distributions using a t-test (paired or independent samples)
    Returns power, magnitude"""
    if sigma is None:
        if CV is None:
            print 'Need to specify variability as sigma or CV!'
            return
        else:
            sigma = [m*c for c,m in zip(CV,mu)]
    dataFunc = [lambda N: np.random.randn(N)*sigma[0] + mu[0], lambda N: np.random.randn(N)*sigma[1] + mu[1]]
    
    if paired:
        if N[0] == N[1]:
            testFunc = stats.ttest_rel
        else:
            print 'Need equal N for a paired sample t-test simulation!'
            return
    else:
        testFunc = stats.ttest_ind
    return powerBySim(dataFunc, testFunc, N, alpha=alpha, iterations=iterations, rseed=rseed)

def powerBySim(dataFunc, testFunc, N, alpha=0.05, iterations=1e3, PMagInds=(1,0), rseed=820):
    """Calculate power for detecting difference in two groups,
    based on functions for simulating data and testing for difference.
    Returns power or for vector of alphas, a vector of powers.
    dataFunc - tuple of two functions that take 1 arg N
               (other args pre-specified using partial)
    testFunc - a function that takes 2 vector args of the data
               and return a magnitude and a pvalue
               Note: Fisher's test will require extra step of counting events
               (use p=fisherTestVector(eventVectorA, eventVectorB) for testFunc)
    PMagInds - indices into the result that is returned from testFunc to get the [pvalue, magnitude]
    N - tuple of 2 group sizes
    alpha - used to compute power
    iterations - number of simulated data sets
    wrapTestForPvalue - auto-wrap to turn typical (h,p) output into p for testFunc()
    """
    np.random.seed(rseed)

    res = np.array([testFunc(dataFunc[0](N[0]), dataFunc[1](N[1])) for i in range(iterations)])

    """Note that these are swapped from how many tests work [pvalue, mag] which is why default is (1,0)"""
    p = res[:,PMagInds[0]]
    magnitude = np.nanmean(res[:,PMagInds[1]])
    
    if isscalar(alpha):
        return (p < alpha).mean(), magnitude
    else:
        tiledPow = (np.tile(p[None,:], (len(alpha),1)) < np.tile(alpha[:,None],(1,iterations))).mean(axis=0)
        return tiledPow, magnitude

def fisherTestVector(aVec, bVec, alternative='two-sided'):
    """Take binary event vectors and return p-value for a 2x2 Fisher's exact test"""
    aS = aVec.sum()
    bS = bVec.sum()
    aL = len(aVec[:])
    bL = len(bVec[:])
    table = [[aS,aL-aS],[bS,bL-bS]]
    return fisherTest(table, alternative)

def powerBySubsample(df, func, subsamples, nStraps=1000, aplha=0.05):
    """Computes power to detect effect in df with a smaller sample.
    Takes data DataFrame and applies func to get observed p-value and effect size
    Then subsamples (with replacement) the data nStraps times, each time recalculating
    effect size and p-value to reject null hypothesis.
    Returns the fraction of times the null hyp was rejected in subsamples
    Data samples should be along the first dimension (rows)"""

    observedP,observedEffect = func(df)

    effectSize = np.zeros(nStraps)
    pvalue = np.zeros(nStraps)

    N = df.shape[0]

    for i in range(nStraps):
        rind = int(np.floor(np.random.rand(subsamples) * N))
        pvalue[i], effectSize[i] = func(df.values[rind,:])

    power = (pvalue < alpha).sum() / nStraps

    return power, effectSize, pvalue, observedEffect, observedP

def probGTEX(x,N,prob):
    """Probability of x or more events (x>0) given,
    N trials and per trial probability prob"""
    return 1 - stats.binom.cdf(x-1, N, prob)

def eventConfidenceInterval(countVec, N):
    """Return confidence interval on observing number of events in countVec
    given N trials (Agresti and Coull  2 sided 95% CI)
    Returns lower and upper confidence limits (lcl,ucl)"""
    p = countVec/N
    z = stats.norm.ppf(0.975)
    lcl= (p + (z**2)/(2*N) - z*np.sqrt((p*(1-p)+z**2/(4*N))/N)) / (1 + (z**2)/N)
    ucl= (p + (z**2)/(2*N) + z*np.sqrt((p*(1-p)+z**2/(4*N))/N)) / (1 + (z**2)/N)
    return lcl,ucl

def correlationCI(N, rho=None, alpha=0.05, bootstraps=None):
    """Calculates the p-values and CIs associated with a range of Spearman's
    rho estimates, given the sample size N.
    If bootstraps is None then it uses a Fisher's transformation for CI, 
    otherwise it computes a bootstrap CI.
    Returns rhoVec, pVec

    Reference for info:
    Borkowf CB. 2000. A new nonparametric method for variance estimation and
    confidence interval construction for Spearman's rank correlation 34:219-241"""
    if rho is None:
        rho = np.linspace(0.05,0.95,20)
    t = rho * np.sqrt((N-2) / (1-rho**2))
    p = stats.distributions.t.sf(np.abs(t), N-2)*2

    if bootstraps is None:
        #stderr = 1./np.sqrt(N - 3)
        #stderr = 1./np.sqrt(1.06/(N - 3))
        stderr = 1./np.sqrt((N - 3)/1.06)
        delta = stats.norm.ppf(1 - alpha/2) * stderr
        lower = np.tanh(np.arctanh(rho) - delta)
        upper = np.tanh(np.arctanh(rho) + delta)
    else:
        aL,bL = [],[]
        fakedata = np.random.randn(N,2)
        for i in range(len(rho)):
            res = induceRankCorr(fakedata, np.array([[1,rho[i]],[rho[i],1]]))
            aL.append(res[:,0])
            bL.append(res[:,1])

        out = np.array([spearmanCI(a,b, alpha=alpha, bootstraps=bootstraps) for a,b in zip(aL,bL)])
        rho = out[:,0]
        p = out[:,1]
        lower = out[:,2]
        upper = out[:,3]
    return rho, p, lower, upper

def spearmanCI(a, b, alpha=0.05, bootstraps=None):
    rho,p = stats.spearmanr(a,b)

    if bootstraps is None:
        stderr = 1./np.sqrt(N - 3)
        delta = stats.norm.ppf(1 - alpha/2) * stderr
        lower = np.tanh(np.arctanh(rho) - delta)
        upper = np.tanh(np.arctanh(rho) + delta)
    else:
        func = lambda *data: stats.spearmanr(*data)[0]
        lower,upper = ci((a,b), statfunction=func, alpha=alpha, n_samples=bootstraps)
    return rho, p, lower, upper

def statDistByN(N, CV, statFunc, alpha=0.05, straps=1000):
    """Computes the statistic statFunc on simulated data sets
    with size N and variation CV.
    Returns a distribution of the stats"""
    
    mu = 100
    result = np.zeros(straps)
    for i in range(straps):
        result[i] = statFunc(np.random.randn(N)*(CV*mu) + mu)
    return result.mean(), np.percentile(result,alpha/2), np.percentile(result, 1-alpha/2), result

def sumAnyNBinom(p, anyN=1):
    """Returns probability of a positive outcome given that 
    a positive outcome requires at least anyN events,
    and given that the independent probability of each event is in vector p.

    Parameters
    ----------
    p : list or 1darray
        Vector of probabilities of each independent event.
    anyN : int
        Minimum number of events required to be considered a positive outcome.

    Returns
    -------
    tot : float
        Overall probability of a positive outcome."""
    if isinstance(p, list):
        p = np.asarray(p)
    n = len(p)
    tmp = np.zeros(n)
    tot = np.zeros(2**n)
    for eventi,event in enumerate(itertools.product(*tuple([[0,1]]*n))):
        event = np.array(event)
        if np.sum(event) >= anyN:
            tmp[np.find(event==1)] = p[np.find(event==1)]
            tmp[np.find(event==0)] = 1 - p[np.find(event==0)]
            tot[eventi] = np.prod(tmp)
    return tot.sum()

def RRCI(a, b, c, d, alpha=0.05):
    """Compute relative-risk and confidence interval,
    given counts in each square of a 2 x 2 table.

    Assumes normal distribution of log-RR.

    Parameters
    ----------
    a,b,c,d : int
        Counts from a 2 x 2 table starting in upper-left and going clockwise.
    alpha : float
        Specifies the (1 - alpha)% confidence interval

    Returns
    -------
    rr : float
        Relative-risk
    lb : float
        Lower-bound
    ub : float
        Upper-bound"""
    se = np.sqrt((1/a + 1/c) - (1/(a+b) + 1/(c+d)))
    rr = a*(c+d)/(c*(a+b))
    delta = stats.norm.ppf(1 - alpha/2) * se
    ub = np.exp(np.log(rr) + delta)
    lb = np.exp(np.log(rr) - delta)
    return rr, lb, ub