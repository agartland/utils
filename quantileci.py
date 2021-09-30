import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

import scipy.stats as stats
from scipy.optimize import root, brentq
from scipy.interpolate import interp1d

__all__ = ['estimate_quantile']

def estimate_quantile(data, q, alpha=0.05, method='exact', weights=None, nsamples=10000):
    """Calculate lower and upper CI of a given quantile using exact method,
        based on beta distribution

        Alan D. Hutson (1999) Calculating nonparametric confidence intervals
            for quantiles using fractional order statistics, Journal of 
            Applied Statistics, 26:3, 343-353, DOI: 10.1080/02664769922458
        
        Wei L, Wang D, Hutson AD. An Investigation of Quantile Function
            Estimators Relative to Quantile Confidence Interval Coverage.
            Commun Stat Theory Methods. 2015;44(10):2107-2135.
            doi: 10.1080/03610926.2013.775304. PMID: 26924881;
            PMCID: PMC4768491.
    
    Parameters
    ----------
    data : np.array
        Data
    q : float, [0, 1]
        Quantile
    alpha : float
        Desired significance level
    method :str 
        "exact" or "approximate"
    
    Returns
    -------
    Lower and upper bound of the quantile
    """
    def _est_bound(n, q, b):
        """Function to estimate the upper and lower bound
        b is targeted lower or upper CI bound"""
        return brentq(lambda x: stats.beta.cdf(q, (n+1)*x, (n+1)*(1-x)) - b, 1e-8, 1-1e-8)
    if not weights is None:
        if method != 'bootstrap':
            print('Using bootstrap method to accomodate weights!')
        method = 'bootstrap'
    n = len(data)
    if q > (1 - 1e-7):
        q = 1 - 1e-7
    if q < 1e-7:
        q = 1e-7
    if method == 'exact':
        lb = _est_bound(n, q, 1 - (alpha/2))
        ub = _est_bound(n, q, alpha/2)
        estx, lx, ux = np.quantile(data, [q, lb, ub], interpolation='linear')
    elif method == 'approximate':
        pn = (n+1) * q
        qn = (n+1) * (1-q)

        lb = stats.beta.ppf(alpha/2, pn, qn)
        ub = stats.beta.ppf(1 - alpha/2, pn, qn)
        estx, lx, ux = np.quantile(data, [q, lb, ub], interpolation='linear')
    elif method == 'bootstrap':
        bsamp = np.zeros(nsamples)
        ndata = len(data)
        if weights is None:
            for i in range(nsamples):
                bsamp[i] = np.quantile(np.random.choice(data, size=ndata, replace=True), q)
            estx = np.quantile(data, q, interpolation='linear')
        else:
            w = weights / weights.sum()
            for i in range(nsamples):
                rind = np.random.choice(np.arange(ndata), size=ndata, replace=True)
                bsamp[i] = weighted_quantile(data[rind], q, weights=w[rind])
            estx = weighted_quantile(data, q, weights=w)

        lx, ux = np.quantile(bsamp, [alpha/2, 1 - alpha/2])
        lb, ub = np.nan, np.nan
    
    return estx, lx, ux, q, lb, ub

def ecdf(x, weights=None, reverse=True, make_step=False):
    """
    For reverse = True:
    Y is proportion of samples >= X or Pr(X>=x)
    
    For reverse = False:
    Y is proportion of samples <= X or Pr(X<=x)
    """
    if weights is None:
        weights = np.zeros(len(x))

    x = np.array(x, copy=True)
    x.sort()
    if reverse:
        x = x[::-1]
    nobs = len(x)
    y = np.linspace(1./nobs, 1, nobs)

    if make_step:
        x = np.concatenate(([x[0]], np.repeat(x[1:].ravel(), 2)))
        y = np.repeat(y.ravel(), 2)[:-1]
    return x, y

def weighted_quantile(data, q, inverse=False, weights=None, reverse=False):
    """
    q : quantile in [0-1]!
    weights
    inverse : bool
        If True then q is treated as a new data point and its corresponding quantile will be returned.
    https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
    """
    if weights is None:
        weights = np.ones(len(data))
    ix = np.argsort(data)
    if reverse:
        ix = ix[::-1]
    data = data[ix] # sort data
    weights = weights[ix] # sort weights
    cdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights) # 'like' a CDF function
    if not inverse:
        out = np.interp(q, cdf, data)
    else:
        out = np.interp(q, data, cdf, left=np.min(cdf)/2, right=1 - np.min(cdf)/2)
    return out

def plot_recdfs(data, quantiles=None, keys=None, logscale=True, make_step=False, alpha=0.05, method='exact', palette=None):
    """
    SLOW for large datasets because it computes the CI at every data point.
    Could easily speed this up if needed.
    """
    if keys is None:
        keys = data.keys()
    if palette is None:
        palette = mpl.cm.Set3.colors

    figh = plt.figure(figsize=(9, 7))
    axh = figh.add_axes([0.1, 0.1, 0.7, 0.8], xscale='log' if logscale else 'linear')
    
    for k, color in zip(keys, palette):
        dat = data[k]
        dat = dat[~np.isnan(dat)]

        x, y = ecdf(dat)

        if quantiles is None:
            qvec = y
        else:
            qvec = quantiles
        n = len(qvec)

        estx = np.zeros(n)
        lq = np.zeros(n)
        uq = np.zeros(n)
        for yi, yy in enumerate(qvec):
            estx[yi], lx, ux, estq, lq[yi], uq[yi] = estimate_quantile(dat, 1 - yy, alpha=alpha, method=method)
        plt.fill_between(estx, y1=1 - lq, y2=1 - uq, color=color, alpha=0.3)
        plt.plot(x, y, '-', color=color, label=k)
        plt.ylabel('Pr(X\u2265x)')
        plt.ylim((0, 1))
        plt.yticks(np.arange(11)/10)
        plt.legend(loc='upper left', bbox_to_anchor=[1, 1])
    return figh

def test_plot(n1=20, n2=10000):
    data = {'A1':np.random.normal(40, 5, size=n1),
                        'A2':np.random.normal(40, 5, size=n2),
                        'B1':np.random.lognormal(0.5, 0, size=n1),
                        'B2':np.random.lognormal(0.5, 0, size=n2)}
    """Plot AVG of 10 ECDFs based on n1 and see if it looks like n2 ECDF to check for bias"""
    xmat = []
    for i in range(5000):
        x, y = ecdf(np.random.normal(40, 5, size=n1))
        xmat.append(x[:,None])
    x1 = np.mean(np.concatenate(xmat, axis=1), axis=1)
    plt.figure(figsize=(10,10))
    plt.plot(x1, y)
    x2, y2 = ecdf(np.random.normal(40, 5, size=n2))
    plt.plot(x2, y2, '-r')
    plt.grid('both')
    plt.yticks(np.arange(21)/20)

    y2i = np.interp(x1, x2[::-1], y2[::-1])
    plt.figure(figsize=(10,10))
    plt.plot(x1, y2i - y, '-r')


    figh = plot_recdfs(data, keys=['A1', 'A2'], logscale=False)
    plot_recdfs(data, keys=['B1', 'B2'], logscale=False)

def test():
    # parama, paramb = 0.5, 0
    # dist = stats.lognorm
    parama, paramb = 40, 5
    dist = stats.norm
    frozen = dist.freeze(parama, paramb)
    nsamp = 40
    ssamps = 50000
    bootsamps = 10000
    qvec = np.linspace(0.02, 0.2, 10)
    alpha = 0.05

    res = []
    for q in qvec:
        """Simulated CI"""
        qsamps = np.zeros(ssamps)
        for i in range(ssamps):
            # qsamps[i] = np.quantile(pop[np.random.permutation(npop)[:nsamp]], q)
            qsamps[i] = np.quantile(frozen.rvs(nsamp), q)
        sest, slci, suci = np.quantile(qsamps, [0.5, alpha/2, 1-alpha/2])
        res.append({'quantile':q,
                    'method':'simulated',
                    'est':sest,
                    'lci':slci,
                    'uci':suci})

        for repi in range(1000):
            data = frozen.rvs(size=nsamp)
            """Bootstrap CI"""
            """
            best, blci, buci, qest, lb, ub = estimate_quantile(data, q, alpha=0.05, method='bootstrap', nsamples=bootsamps)
            res.append({'quantile':q,
                        'method':'bootstrap',
                        'est':best,
                        'lci':blci,
                        'uci':buci})
            """

            """Exact CI"""
            #lbq, ubq = quantile_ci(q, nsamp, alpha, method='exact')
            #elb, eub = frozen.ppf([lbq, ubq])
            eest, elci, euci, qest, lb, ub = estimate_quantile(data, q, alpha=0.05, method='exact')

            res.append({'quantile':q,
                        'method':'exact',
                        'est':eest,
                        'lci':elci,
                        'uci':euci})

        """Approximate CI"""
        #albq, aubq = quantile_ci(q, nsamp, alpha, method='approximate')
        #alb, aub = frozen.ppf([albq, aubq])

        """print(f'Sampled:   [{slb:1.2f}, {sub:1.2f}]')
        print(f'Bootstrap: [{blb:1.2f}, {bub:1.2f}]')
        print(f'Exact:     [{elb:1.2f}, {eub:1.2f}]')
        print(f'Exact est: [{lci:1.2f}, {uci:1.2f}]')
        print(f'Approx:    [{alb:1.2f}, {aub:1.2f}]')"""

    resdf = pd.DataFrame(res)
    summ = resdf.groupby(['method', 'quantile']).agg(np.mean).reset_index()
    
    plt.figure(figsize=(10,10))
    for method, color in zip(['simulated', 'exact'], ['crimson', 'dodgerblue', 'green']):
        tmp = summ.loc[summ['method'] == method]
        plt.plot(tmp['quantile'], tmp['est'], '-', color=color, label=method)
        plt.plot(tmp['quantile'], tmp['lci'], '--', color=color)
        plt.plot(tmp['quantile'], tmp['uci'], '--', color=color)
    plt.xlabel('Quantile')
    plt.ylabel('X (normally distributed random variable)')
    plt.legend(loc='lower right')