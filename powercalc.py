from scipy import stats
import numpy as np

__all_ = ['veEndpointsRequired']

def veEndpointsRequired(rr0 = 1, rralt = 0.5, targetpow = 0.9, K = 1, alpha = 0.025):
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