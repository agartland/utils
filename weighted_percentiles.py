import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

__all__ = ['weighted_percentiles',
           'weighted_swarmbox']

def weighted_swarmbox(x, y, weights, data,
                      order=None, colors=None, fill=False, violin=False,
                      swarm_alpha=1, swarm_size=5, swarm_max=None,
                      box_alpha=1, box_ec='k', box_facewhite=True, box_width=0.8):
    if order is None:
        order = sorted(data[x].unique())
    if colors is None:
        colors = ['k']*len(order)

    if not swarm_max is None:
        res = []
        for xval in order:
            tmp = data.loc[data[x] == xval]
            if tmp.shape[0] > swarm_max:
                tmp = tmp.dropna(subset=[y, weights]).sample(n=swarm_max, replace=False)
            res.append(tmp)
        swarm_data = pd.concat(res, axis=0)
    else:
        swarm_data = data

    swarm = sns.swarmplot(data=swarm_data,
                          x=x, y=y,
                          order=order,
                          linewidth=0.5,
                          edgecolor='black',
                          dodge=True,
                          alpha=swarm_alpha,
                          size=swarm_size,
                          palette=colors)

    outh = dict(swarm=swarm)
    desat_colors = sns.color_palette(colors, desat=0.5)
    for x_coord, xval in enumerate(order):
        tmp = data.loc[data[x] == xval, [y, weights]].dropna()
        if violin:
            handles = one_weighted_boxplot(data=tmp[y],
                                              data_range=None,
                                              weights=tmp[weights],
                                              x=x_coord,
                                              width=box_width,
                                              lw=0.5,
                                              fc=None if box_facewhite else desat_colors[x_coord],
                                              ec=box_ec,
                                              alpha=box_alpha,
                                              zorder=-4,
                                              violin=True)
        else:
            '''
            swarm = sns.violinplot(data=data,
                          x=x, y=y,
                          order=order,
                          linewidth=0.5,
                          edgecolor='black',
                          dodge=True,
                          alpha=swarm_alpha,
                          size=swarm_size,
                          palette=colors)
            '''
            handles = one_weighted_boxplot(data=tmp[y],
                                           weights=tmp[weights],
                                           x=x_coord,
                                           width=box_width,
                                           lw=1,
                                           fc=None if box_facewhite else desat_colors[x_coord],
                                           ec=box_ec,
                                           alpha=box_alpha,
                                           zorder=-4)
        outh['box_%1.0f' % x_coord] = handles
        
    return outh

def one_weighted_boxplot(data, weights, x, data_range=None, width=0.8, lw=0.5, fc=None, ec='k', alpha=1, zorder=1, violin=False):
    if not fc is None:
        fill = True
    else:
        fill = False
    y = weighted_percentiles(data, [25, 50, 75], weights=weights)
    iqr = y[2] - y[0]
    whisk_hi_y = np.min([y[2] + 1.5*iqr, np.max(data)])
    whisk_lo_y = np.max([y[0] - 1.5*iqr, np.min(data)])
    
    line_params = dict(lw=lw, color=ec, alpha=alpha, zorder=zorder)
    if violin:
        if data_range is None:
            mn, mx = np.min(data), np.max(data)
            rng =  np.max(data) - np.min(data)
            kde = stats.gaussian_kde(data, bw_method=None, weights=weights)
            y_rng = np.linspace(mn-rng, mx+rng, 200)
            pdf = kde.pdf(y_rng)
            mx = np.max(pdf)
            pdf = (pdf / mx)
            """Keep only the part of the pdf thats greater than 1% of max"""
            y_rng = y_rng[np.nonzero(pdf > 0.01)[0]]
            mn, mx = np.min(y_rng), np.max(y_rng)
        else:
            mn, mx = data_range
        
        y_rng = np.linspace(mn, mx, 200)
        pdf = kde.pdf(y_rng)
        mx = np.max(pdf)
        pdf = (pdf / mx) * (width/2)

        line_params = dict(lw=lw, alpha=alpha, zorder=zorder, facecolor=fc, edgecolor=fc)
        violin_lh = plt.fill_betweenx(y_rng, x*np.ones(len(pdf)), x - pdf, **line_params)
        violin_rh = plt.fill_betweenx(y_rng, x*np.ones(len(pdf)), x + pdf, **line_params)

        """Redraw line with ec now (to avoid centerline being draw by fillbetween)"""
        line_params = dict(lw=lw, alpha=alpha, zorder=zorder, color=ec)
        violin_line_lh = plt.plot(x - pdf, y_rng, **line_params)
        violin_line_rh = plt.plot(x + pdf, y_rng, **line_params)

        out = dict(violin_left=violin_lh,
                   violin_right=violin_rh,
                   violin_left_line=violin_line_lh,
                   violin_right_line=violin_line_rh)

        line_params = dict(lw=lw*2, color=ec, alpha=alpha, zorder=zorder)
        h = plt.plot([x - (kde.pdf(y[1]) / mx) * (width/2),
                      x + (kde.pdf(y[1]) / mx) * (width/2)], [y[1], y[1]], **line_params)
        out['median'] = h
        line_params = dict(lw=lw, color=ec, alpha=alpha, zorder=zorder)
        for lab,tmpy in [('25th', y[0]),
                         ('75th', y[2]),
                         ('min', y_rng[0]),
                         ('max', y_rng[-1])]:
            h = plt.plot([x - (kde.pdf(tmpy) / mx) * (width/2),
                          x + (kde.pdf(tmpy) / mx) * (width/2)], [tmpy, tmpy], **line_params)
            out[lab] = h
        
    else:
        recth = plt.Rectangle((x-width/2, y[0]),
                              width=width,
                              height=iqr,
                              fill=fill,
                              facecolor=fc,
                              edgecolor=ec,
                              zorder=zorder-1,
                              linewidth=lw,
                              alpha=alpha)
        plt.gca().add_patch(recth)
    
        medh = plt.plot([x-width/2, x+width/2], [y[1], y[1]], **line_params)
        whisk_hih = plt.plot([x, x], [y[2], whisk_hi_y], **line_params)
        whisk_loh = plt.plot([x, x], [y[0], whisk_lo_y], **line_params)

        whisk_hi_caph = plt.plot([x-width/4, x+width/4], [whisk_hi_y]*2, **line_params)
        whisk_lo_caph = plt.plot([x-width/4, x+width/4], [whisk_lo_y]*2, **line_params)
        
        out = dict(box=recth,
                   median=medh,
                   whisk_hi=whisk_hih,
                   whisk_lo=whisk_loh,
                   whisk_hi_cap=whisk_hi_caph,
                   whisk_lo_cap=whisk_lo_caph)
    return out

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
    Note that there are also several options in R for computing a weighted quantile,
    but I did not fully understand the motivation for each. The chosen option here was intuitive to me
    and agreed well with the empirical solution below.
    https://github.com/harrelfe/Hmisc/R/wtd.stats.s"""
    
    # ecdf = np.cumsum(weights) / weights.sum()
    ecdf = (np.cumsum(weights) - 0.5 * weights) / np.sum(weights)

    return np.interp(quantiles, ecdf, a)

def wp(data, wt, percentiles):
    """Compute weighted percentiles.
    Solution and code from:
    http://kochanski.org/gpk/code/speechresearch/gmisclib/gmisclib.weighted_percentile-pysrc.html#wp
    
    If the weights are equal, this is the same as normal percentiles.
    Elements of the C{data} and C{wt} arrays correspond to
    each other and must have equal length (unless C{wt} is C{None}).

    @param data: The data.
    @type data: A L{np.ndarray} array or a C{list} of numbers.
    @param wt: How important is a given piece of data.
    @type wt: C{None} or a L{np.ndarray} array or a C{list} of numbers.
    All the weights must be non-negative and the sum must be
    greater than zero.
    @param percentiles: what percentiles to use. (Not really percentiles,
    as the range is 0-1 rather than 0-100.)
    @type percentiles: a C{list} of numbers between 0 and 1.
    @rtype: [ C{float}, ... ]
    @return: the weighted percentiles of the data.
    """
    assert np.greater_equal(percentiles, 0.0).all(), "Percentiles less than zero"
    assert np.less_equal(percentiles, 1.0).all(), "Percentiles greater than one"
    data = np.asarray(data)
    assert len(data.shape) == 1
    
    if wt is None:
        wt = np.ones(data.shape, np.float)
    else:
        wt = np.asarray(wt, np.float)
    
    assert wt.shape == data.shape
    assert np.greater_equal(wt, 0.0).all(), "Not all weights are non-negative."
    assert len(wt.shape) == 1
    n = data.shape[0]
    assert n > 0
    i = np.argsort(data)
    sd = np.take(data, i, axis=0)
    sw = np.take(wt, i, axis=0)
    aw = np.add.accumulate(sw)
    if not aw[-1] > 0:
        raise ValueError("Nonpositive weight sum")
    
    w = (aw - 0.5 * sw) / aw[-1]
    spots = np.searchsorted(w, percentiles)
    o = []
    for (s, p) in zip(spots, percentiles):
        if s == 0:
            o.append(sd[0])
        elif s == n:
            o.append(sd[n-1])
        else:
            f1 = (w[s] - p)/(w[s] - w[s-1])
            f2 = (p - w[s-1])/(w[s] - w[s-1])
            assert f1>=0 and f2>=0 and f1<=1 and f2<=1
            assert np.abs(f1+f2-1.0) < 1e-6
            o.append(sd[s-1]*f1 + sd[s]*f2)
    return o

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

