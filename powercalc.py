from scipy import stats
import numpy as np
import itertools
from functools import partial
from scikits.bootstrap import ci

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
          'sumAnyNBinom',
          'sensitivityCI',
          'specificityCI',
          'rocStats',
          'ccsize',
          'mean_shift_estimation',
          'cont_2samp_power']

"""NOTE: Many of these were hacked together in very short time, but they code may still be useful for next time."""

try:
    """Attempt to use the fisher library (cython) if available (100x speedup)"""
    import fisher
    def fisherTest(tab,alternative='two-sided'):
        res = fisher.pvalue(tab[0][0], tab[0][1], tab[1][0], tab[1][1])
        OR = (tab[0][0] * tab[1][1]) / (tab[0][1] * tab[1][0])

        if alternative == 'two-sided':
            return (OR, res.two_tail)
        elif alternative == 'less':
            return (OR, res.left_tail)
        elif alternative == 'greater':
            return (OR, res.right_tail)
    #print("Using Cython-powered Fisher's exact test")
except ImportError:
    print("Using scipy.stats Fisher's exact test (slow)")
    fisherTest = stats.fisher_exact

def cont_2samp_power(alpha=None, difference=None, sigma=None, n=None, beta=None):
    """Compute power or sample size for the comparison of two normally distributed variables
    with shared variance.

    OR JUST USE sm.stats.power.tt_ind_solve_power

    If sample size is provided then power is calculated and if power is provided sample size
    is calculated."""

    crit_alpha = stats.norm.ppf(1-alpha/2)

    if n is None:
        crit_beta = stats.norm.ppf(beta)
        ss = 2*(sigma**2 / difference**2)*(crit_alpha + crit_beta)**2 
        return ss
    elif beta is None:
        effect_size = difference / sigma
        if type(n) is tuple:
            beta_z = effect_size / np.sqrt(1/n[0] + 1/n[1]) - stats.norm.ppf(1-alpha/2)
        else:
            beta_z = effect_size * np.sqrt(n/2) - stats.norm.ppf(1-alpha/2)
            # beta_z = effect_size / np.sqrt(1/n + 1/n) - stats.norm.ppf(1-alpha/2)

        power = stats.norm.cdf(beta_z)
        return power

#cont_2samp_power(alpha=0.05, difference=10, sigma=15, n=None, beta=0.8)
#cont_2samp_power(alpha=0.05, difference=10, sigma=15, n=36, beta=None)



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
    while curPower < np.min(targetpow + 0.05, 0.99):
        n = n + 1
        """Compute critical value for exact binomial test and power"""
        xcrit = stats.binom.ppf(alpha, n, p0) - 1
        curPower = stats.binom.cdf(xcrit, n, p)
    
    """Now search backwards for smallest n for which power is greater than target for all larger n"""
    while curPower > targetpow:     
        n = n - 1
        xcrit = stats.binom.ppf(alpha, n, p0) - 1
        curPower = stats.binom.cdf(xcrit, n, p)

    n = n + 1
    xcrit = stats.binom.ppf(alpha, n, p0) - 1
    return n, xcrit

def eventRatePower(rate, N, alpha=0.05, iterations=1e3, alternative='two-sided', rseed=820):
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
            print('Need to specify variability as sigma or CV!')
            return
        else:
            sigma = [m*c for c, m in zip(CV, mu)]
    dataFunc = [lambda N: np.random.randn(N)*sigma[0] + mu[0], lambda N: np.random.randn(N)*sigma[1] + mu[1]]
    
    if paired:
        if N[0] == N[1]:
            testFunc = stats.ttest_rel
        else:
            print('Need equal N for a paired sample t-test simulation!')
            return
    else:
        testFunc = stats.ttest_ind
    return powerBySim(dataFunc, testFunc, N, alpha=alpha, iterations=iterations, rseed=rseed)

def mean_shift_estimation(mu, N, sigma=None, CV=None, alpha=0.05):
    """Compute CI on the difference between two group means

    Parameters
    ----------
    mu : list, 2
        Means of two groups
    N : list, 2
        Number of observations in each group
    sigma : list, 2
        Standard deviation of the population for each group
    CV : list, 2
        Alternatively can provide a CV such that sigma changes with mu
    alpha : float
        Desired confidence level

    Returns
    -------
    mu_diff, lcl, ucl"""

    if sigma is None:
        if CV is None:
            print('Need to specify variability as sigma or CV!')
            return
        else:
            sigma = [m*c for c, m in zip(CV, mu)]
    dof = (N[0] - 1) + (N[1] - 1)
    criticalz = -stats.t.ppf(alpha / 2, dof)
    dmu = mu[1] - mu[0]
    se = np.sqrt(sigma[0]**2 / N[0] + sigma[1]**2 / N[1])
    lcl = dmu - criticalz * se
    ucl = dmu + criticalz * se
    return dmu, lcl, ucl

def powerBySim(dataFunc, testFunc, N, alpha=0.05, iterations=1e3, PMagInds=(1, 0), rseed=820):
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

    res = np.array([testFunc(dataFunc[0](N[0]), dataFunc[1](N[1])) for i in range(int(iterations))])

    """Note that these are swapped from how many tests work [pvalue, mag] which is why default is (1,0)"""
    p = res[:, PMagInds[0]]
    magnitude = np.nanmean(res[:, PMagInds[1]])
    
    if np.isscalar(alpha):
        return (p < alpha).mean(), magnitude
    else:
        tiledPow = (np.tile(p[None,:], (len(alpha), 1)) < np.tile(alpha[:, None], (1, iterations))).mean(axis=0)
        return tiledPow, magnitude

def fisherTestVector(aVec, bVec, alternative='two-sided'):
    """Take binary event vectors and return p-value for a 2x2 Fisher's exact test"""
    aS = aVec.sum()
    bS = bVec.sum()
    aL = len(aVec[:])
    bL = len(bVec[:])
    table = [[aS, aL-aS], [bS, bL-bS]]
    return fisherTest(table, alternative)

def powerBySubsample(df, func, subsamples, nStraps=1000, aplha=0.05):
    """Computes power to detect effect in df with a smaller sample.
    Takes data DataFrame and applies func to get observed p-value and effect size
    Then subsamples (with replacement) the data nStraps times, each time recalculating
    effect size and p-value to reject null hypothesis.
    Returns the fraction of times the null hyp was rejected in subsamples
    Data samples should be along the first dimension (rows)"""

    observedP, observedEffect = func(df)

    effectSize = np.zeros(nStraps)
    pvalue = np.zeros(nStraps)

    N = df.shape[0]

    for i in range(nStraps):
        rind = int(np.floor(np.random.rand(subsamples) * N))
        pvalue[i], effectSize[i] = func(df.values[rind,:])

    power = (pvalue < alpha).sum() / nStraps

    return power, effectSize, pvalue, observedEffect, observedP

def probGTEX(x, N, prob):
    """Probability of x or more events (x>0) given,
    N trials and per trial probability prob."""
    return 1 - stats.binom.cdf(x-1, N, prob)

def eventCI(x, N, alpha=0.05, method='score'):
    """Return confidence interval on observing number of events in x
    given N trials (Agresti and Coull  2 sided 95% CI)
    Returns lower and upper confidence limits (lcl,ucl)

    Code has been checked against R binom package. "Score" was derived
    from the Agresti paper and is equivalent to Wilson (copied from the R package).
    From the paper this seems to be the best in most situations.

    A. Agresti, B. A. Coull, T. A. Statistician, N. May,
    Approximate Is Better than "Exact" for Interval Estimation of Binomial Proportions,
    52, 119–126 (2007)."""

    x = np.asarray(x)
    if isinstance(N, list):
        N = np.asarray(N)
    p = x/N
    z = stats.norm.ppf(1.-alpha/2.)
    if method == 'score':
        lcl = (p + (z**2)/(2*N) - z*np.sqrt((p*(1-p)+z**2/(4*N))/N)) / (1 + (z**2)/N)
        ucl = (p + (z**2)/(2*N) + z*np.sqrt((p*(1-p)+z**2/(4*N))/N)) / (1 + (z**2)/N)
    elif method == 'wilson':
        """p1 <- p + 0.5 * z2/n
            p2 <- z * sqrt((p * (1 - p) + 0.25 * z2/n)/n)
            p3 <- 1 + z2/n
            lcl <- (p1 - p2)/p3
            ucl <- (p1 + p2)/p3"""
        p1 = p + 0.5 * (z**2 / N)
        p2 = z * np.sqrt((p * (1 - p) + 0.25 * z**2/N)/N)
        p3 = 1 + z**2 / N
        lcl = (p1 - p2)/p3
        ucl = (p1 + p2)/p3
    elif method == 'agresti-coull':
        """.x <- x + 0.5 * z2
        .n <- n + z2
        .p <- .x/.n
        lcl <- .p - z * sqrt(.p * (1 - .p)/.n)
        ucl <- .p + z * sqrt(.p * (1 - .p)/.n)"""
        xtmp = x + 0.5 * z**2
        ntmp = N + z**2
        ptmp = xtmp / ntmp
        se = np.sqrt(ptmp * (1 - ptmp)/ntmp)
        lcl = ptmp - z * se
        ucl = ptmp + z * se
    elif method == 'exact':
        """Clopper-Pearson (1934)"""
        """ x1 <- x == 0
            x2 <- x == n
            lb <- ub <- x
            lb[x1] <- 1
            ub[x2] <- n[x2] - 1
            lcl <- 1 - qbeta(1 - alpha2, n + 1 - x, lb)
            ucl <- 1 - qbeta(alpha2, n - ub, x + 1)
            if(any(x1)) lcl[x1] <- rep(0, sum(x1))
            if(any(x2)) ucl[x2] <- rep(1, sum(x2))"""
        lb = x.copy()
        ub = x.copy()
        lb[x == 0] = 1
        ub[x == N] = N - 1

        lcl = 1 - stats.beta.ppf(1 - alpha/2, N + 1 - x, lb)
        ucl = 1 - stats.beta.ppf(alpha/2, N - ub, x + 1)

        lcl[x == 0] = 0
        ucl[x == N] = 1
    elif method == 'wald':
        se = np.sqrt(p*(1-p)/N)
        ucl = p + z * se
        lcl = p - z * se
    return lcl, ucl

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
        rho = np.linspace(0.05, 0.95, 20)
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
        aL, bL = [], []
        fakedata = np.random.randn(N, 2)
        for i in range(len(rho)):
            res = induceRankCorr(fakedata, np.array([[1, rho[i]], [rho[i], 1]]))
            aL.append(res[:, 0])
            bL.append(res[:, 1])

        out = np.array([spearmanCI(a, b, alpha=alpha, bootstraps=bootstraps) for a, b in zip(aL, bL)])
        rho = out[:, 0]
        p = out[:, 1]
        lower = out[:, 2]
        upper = out[:, 3]
    return rho, p, lower, upper

def spearmanCI(a, b, alpha=0.05, bootstraps=None):
    rho, p = stats.spearmanr(a, b)

    if bootstraps is None:
        stderr = 1./np.sqrt(N - 3)
        delta = stats.norm.ppf(1 - alpha/2) * stderr
        lower = np.tanh(np.arctanh(rho) - delta)
        upper = np.tanh(np.arctanh(rho) + delta)
    else:
        func = lambda *data: stats.spearmanr(*data)[0]
        lower, upper = ci((a, b), statfunction=func, alpha=alpha, n_samples=bootstraps)
    return rho, p, lower, upper

def statDistByN(N, CV, statFunc, alpha=0.05, straps=1000):
    """Computes the statistic statFunc on simulated data sets
    with size N and variation CV.
    Returns a distribution of the stats"""
    
    mu = 100
    result = np.zeros(straps)
    for i in range(straps):
        result[i] = statFunc(np.random.randn(N)*(CV*mu) + mu)
    return result.mean(), np.percentile(result, alpha/2), np.percentile(result, 1-alpha/2), result

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
    for eventi, event in enumerate(itertools.product(*tuple([[0, 1]]*n))):
        event = np.array(event)
        if np.sum(event) >= anyN:
            tmp[np.find(event==1)] = p[np.find(event==1)]
            tmp[np.find(event==0)] = 1 - p[np.find(event==0)]
            tot[eventi] = np.prod(tmp)
    return tot.sum()

def RRCI(a, b, c, d, alpha=0.05, RR0=1, method='katz'):
    """Compute relative-risk and confidence interval,
    given counts in each square of a 2 x 2 table.

    Katz method and adj-log assume normal distribution of log-RR.

            OUTCOME
             +   -
           ---------
         + | a | b |
    PRED   |-------|
         - | c | d |
           ---------

    Fagerland MW, Lydersen S, Laake P. Recommended confidence intervals
        for two independent binomial proportions. Stat Methods Med Res. 2011
    

    Parameters
    ----------
    a,b,c,d : int
        Counts from a 2 x 2 table starting in upper-left and going clockwise.
        a = TP
        b = FP
        c = FN
        d = TN

    alpha : float
        Specifies the (1 - alpha)% confidence interval
    RR0 : float
        Null hypothesis for Wald test p-value
    method : str
        Currently support katz or adj-log which can handle 0s

    Returns
    -------
    rr : float
        Relative-risk
    lb : float
        Lower-bound
    ub : float
        Upper-bound
    pvalue : float
        P-value by inverting the Wald CI"""
    if np.all([np.isscalar(x) for x in [a, b, c, d]]):
        back2scalar = True
        a = np.asarray(a).reshape((1,))
        b = np.asarray(b).reshape((1,))
        c = np.asarray(c).reshape((1,))
        d = np.asarray(d).reshape((1,))
    elif np.any([np.isscalar(x) for x in [a, b, c, d]]):
        raise ValueError('Cannot currently handle mix of scalars and vectors.')
    else:
        a = np.asarray(a)
        b = np.asarray(b)
        c = np.asarray(c)
        d = np.asarray(d)
        back2scalar = False

    if method.lower() == 'katz':
        """Standard normal approximation of RR CI"""
        rr = (a / (a+b)) / (c / (c+d))
        rr[c == 0] = np.inf
       
        se = np.sqrt((1/a + 1/c) - (1/(a+b) + 1/(c+d)))
        se[np.isnan(se)] = np.inf

    elif method == 'adj-log':
        """Add 1/2 count to each square"""
        rr = ((a+0.5) / (a + b + 1)) / ((c+0.5) / (c + d + 1))
        se = np.sqrt((1/(a+0.5) + 1/(c+0.5)) - (1/(a + b + 1) + 1/(c + d + 1)))
        
    delta = stats.norm.ppf(1 - alpha/2) * se
    ub = np.exp(np.log(rr) + delta)
    lb = np.exp(np.log(rr) - delta)

    """Invert CI for H0: RR = 1"""
    pvalue = 1 - stats.norm.cdf(np.abs(np.log(rr) - np.log(RR0))/se)

    if back2scalar and len(rr) == 1:
        rr = rr[0]
        lb = lb[0]
        ub = ub[0]
        pvalue = pvalue[0]

    return rr, lb, ub, pvalue

def sensitivityCI(a, b, c, d, alpha=0.05, method='score'):
    """Compute sensitivity and confidence interval,
    given counts in each square of a 2 x 2 table.

            OUTCOME
             +   -
           ---------
         + | a | b |
    PRED   |-------|
         - | c | d |
           ---------

    Parameters
    ----------
    a,b,c,d : int
        Counts from a 2 x 2 table starting in upper-left and going clockwise.
        a = TP
        b = FP
        c = FN
        d = TN

    alpha : float
        Specifies the (1 - alpha)% confidence interval

    Returns
    -------
    sens : float
        Relative-risk
    lb : float
        Lower-bound
    ub : float
        Upper-bound"""
    if np.all([np.isscalar(x) for x in [a, c]]):
        back2scalar = True
        a = np.asarray(a).reshape((1,))
        c = np.asarray(c).reshape((1,))
    elif np.any([np.isscalar(x) for x in [a,c]]):
        raise ValueError('Cannot currently handle mix of scalars and vectors.')
    else:
        a = np.asarray(a)
        c = np.asarray(c)
        back2scalar = False

    sens = a / (a+c)
    lb, ub = eventCI(x=a, N=a+c, alpha=alpha, method=method)

    if back2scalar and len(sens) == 1:
        sens = sens[0]
        lb = lb[0]
        ub = ub[0]

    return sens, lb, ub

def specificityCI(a, b, c, d, alpha=0.05, method='score'):
    """Compute specificity and confidence interval,
    given counts in each square of a 2 x 2 table.

            OUTCOME
             +   -
           ---------
         + | a | b |
    PRED   |-------|
         - | c | d |
           ---------

    Parameters
    ----------
    a,b,c,d : int
        Counts from a 2 x 2 table
        a = TP
        b = FP
        c = FN
        d = TN

    alpha : float
        Specifies the (1 - alpha)% confidence interval

    Returns
    -------
    spec : float
        Relative-risk
    lb : float
        Lower-bound
    ub : float
        Upper-bound"""
    if np.all([np.isscalar(x) for x in [b, d]]):
        back2scalar = True
        b = np.asarray(b).reshape((1,))
        d = np.asarray(d).reshape((1,))
    elif np.any([np.isscalar(x) for x in [b, d]]):
        raise ValueError('Cannot currently handle mix of scalars and vectors.')
    else:
        b = np.asarray(b)
        d = np.asarray(d)
        back2scalar = False

    spec = d / (b+d)
    lb, ub = eventCI(x=d, N=b+d, alpha=alpha, method=method)

    if back2scalar and len(spec) == 1:
        spec = spec[0]
        lb = lb[0]
        ub = ub[0]

    return spec, lb, ub

def computeRR(df, outcome, predictor, alpha=0.05):
    """RR point-estimate, CI and p-value, without conditioning on the total number of events.

    Derived inputs
    --------------
    nneg : int
        Number of outcomes in the covariate negative group
        (false-negatives)
    Nneg : int
        Total number of participants in the covariate negative group
        (false-negatives + true-negatives)
    npos : int
        Number of outcomes in the covariate positive group
        (true-positives)
    Npos : int
        Total number of participants in the covariate positive group
        (true-positives + false-negatives)"""

    tmp = df[[outcome, predictor]].dropna()
    nneg = tmp[outcome].loc[tmp[predictor] == 0].sum()
    Nneg = (tmp[predictor] == 0).sum()
    npos = tmp[outcome].loc[tmp[predictor] == 1].sum()
    Npos = (tmp[predictor] == 1).sum()

    rr = (npos/(Npos)) / (nneg/(Nneg))

    se = np.sqrt((Nneg-nneg)/(nneg*Nneg) + (Npos-npos)/(npos*Npos))

    z = stats.norm.ppf(1 - alpha/2)

    ci = np.exp(np.array([np.log(rr) - se*z, np.log(rr) + se*z]))
    
    """Wald CI"""
    pvalue = stats.norm.cdf(np.log(rr)/se)

    return  pd.Series([rr, ci[0], ci[1], pvalue], index=['RR', 'LL', 'UL', 'pvalue'])


def rocStats(a, b, c, d, returnSeries=True):
    """Compute stats for a 2x2 table.

    Parameters
    ----------
    a,b,c,d : int
        Counts from a 2 x 2 table starting in upper-left and going clockwise.
        a = TP
        b = FN
        c = FP
        d = TN

    Optionally return a series with quantities labeled.

    Returns
    -------
    sens : float
        Sensitivity (1 - false-negative rate)
    spec : float
        Specificity (1 - false-positive rate)
    ppv : float
        Positive predictive value (1 - false-discovery rate)
    npv : float
        Negative predictive value
    acc : float
        Accuracy
    OR : float
        Odds-ratio of the observed event in the two predicted groups.
    rr : float
        Relative rate of the observed event in the two predicted groups.
    nnt : float
        Number needed to treat, to prevent one case.
        (assuming all predicted positives were "treated")"""

    sens = a / (a+b)
    spec = d / (c+d)
    ppv = a / (a+c)
    npv = d / (b+d)
    nnt = 1 / (a/(a+c) - b/(b+d))
    acc = (a + d)/n
    rr = (a / (a+c)) / (b / (b+d))
    OR = (a/b) / (c/d)

    if returnSeries:
        vec = [sens, spec, ppv, npv, nnt, acc, rr, OR]
        out = pd.Series(vec, name='ROC', index=['Sensitivity', 'Specificity', 'PPV', 'NPV', 'NNT', 'ACC', 'RR', 'OR'])
    else:
        out = (sens, spec, ppv, npv, nnt, acc, rr, OR)
    return out

def ccsize(n, q, pD, p1, theta, alpha=0.05):
    """Power with case-cohort design, from:
    Cai and Zeng. "Sample Size/Power Calculation for Case–Cohort Studies." Biometrics, 2004.

    R code from the gap package, except this code assumes a two-sided test.
    
    Based on a log-rank test comparing event rates in two groups, with full sampling of the
    cases and partial smapling of the controls.

    Parameters
    ----------
    n : int
        Total number of participants in the cohort
    q : float [0, 1]
        Sampling fraction of the subcohort
    pD : float [0, 1]
        Proportion of the failures in the full cohort
    p1 : float [0, 1]
        Proportions of the two groups (p2 = 1 - p1)
    theta : float
        Log-hazard ratio for two groups (effect size)
    alpha : float
        Two-sided significance level

    Returns
    -------
    power : float [0, 1]
        Power to reject the null hypothesis that the event rate
        is equal in the two groups."""

    p2 = 1 - p1
    z_alpha = stats.norm.ppf(alpha / 2)
    z = z_alpha + np.sqrt(n) * theta * np.sqrt(p1 * p2 / (1 / pD + (1 / q - 1)))
    power = stats.norm.cdf(z)
    return power
