import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.random import permutation, seed
import pandas as pd
import seaborn as sns
from itertools import cycle

from vectools import untangle

try:
    import numba as nb
    from bootstrap_nb import bootci_nb
    NUMBA = True

    @nb.njit()
    def _keepdims_mean(dat):
        return np.array([np.mean(dat[:, 0])])

except ImportError:
    from scikits.bootstrap import ci
    NUMBA = False


__all__ = ['scatterdots',
           'myboxplot',
           'manyboxplots',
           'swarmbox',
           'discrete_boxplot']

def scatterdots(data, x, axh=None, width=0.8, returnx=False, rseed=820, **kwargs):
    """Dots plotted with random x-coordinates and y-coordinates from data array.

    Parameters
    ----------
    data : ndarray
    x : float
        Specifies the center of the dot cloud on the x-axis.
    axh : matplotlib figure handle
        If None then use plt.gca()
    width : float
        Specifies the range of the dots along the x-axis.
    returnx : bool
        If True, return the x-coordinates of the plotted data points.
    rseed : float
        Random seed. Defaults to a constant so that regenerated figures of
        the same data are identical.

    Returns
    -------
    Optionally returns the x-coordinates as plotted."""

    if axh is None:
        axh = plt.gca()

    np.random.seed(rseed)

    if data is None or len(data) == 0:
        if returnx:
            return None
        return
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    validi = np.arange(len(data))
    if any(np.isnan(data)):
        validi = np.where(np.logical_not(np.isnan(data)))[0]
    ploty = data[validi]

    if len(ploty) == 0:
        if returnx:
            return None
        return
    w = width
    plotx = np.random.permutation(np.linspace(-w/2., w/2., len(ploty)) + x)
    axh.scatter(plotx, ploty, **kwargs)
    
    if returnx:
        outx = np.nan * np.ones(data.shape)
        outx[validi] = plotx
        return outx

def myboxplot(data, x = 1, axh=None, width=0.8, boxcolor='black',scatterwidth=0.6,dotcolor='red',returnx=False,subsetInd=None,altDotcolor='gray',violin=False,**kwargs):
    """Make a boxplot with scatterdots overlaid.

    Parameters
    ----------
    data : np.ndarray or pd.Series
    x : float
        Position of box along x-axis.
    axh : matplotlib figure handle
        If None then use plt.gca()
    width : float
        Width of the box.
    boxcolor : mpl color
    scatterwidth : float
        Width of the spread of the data points.
    dotcolor : mpl color
    subsetInd : boolean or int index
        Indicates a subset of the data that should be summarized in the boxplot.
        However, all data points will be plotted.
    altDotcolor : mpl color
        Specify the color of the data points that are not in the subset.
    returnx : bool
        Return the x-coordinates of the data points.
    violin : bool
        Specify whether the box is a violin plot.

    Returns
    -------
    outx : np.ndarray
        Optionall, an array of the x-coordinates as plotted."""

    if axh is None:
        axh = plt.gca()

    if isinstance(data, pd.Series):
        data = data.values

    if not subsetInd is None:
        if not (subsetInd.dtype == np.array([0, 1], dtype=bool).dtype):
            tmp = np.zeros(data.shape, dtype=bool)
            tmp[subsetInd] = True
            subsetInd = tmp
    else:
        subsetInd = np.ones(data.shape, dtype=bool)
    subsetInd = np.asarray(subsetInd)

    if not 's' in kwargs:
        kwargs['s'] = 20
    if not 'marker' in kwargs:
        kwargs['marker'] = 'o'
    if not 'linewidths' in kwargs:
        kwargs['linewidths'] = 0.5

    """Boxplot with dots overlaid"""
    outx = np.zeros(data.shape)
    if subsetInd.sum() > 0:
        if not boxcolor == 'none' and not boxcolor is None:
            if violin and False:
                sns.violinplot(data[subsetInd], color = boxcolor, positions = [x], alpha = 0.5)
            else:
                bp = axh.boxplot(data[subsetInd], positions = [x], widths = width, sym = '')
                for element in list(bp.keys()):
                    for b in bp[element]:
                        b.set_color(boxcolor)

        kwargs['c'] = dotcolor
        subsetx = scatterdots(data[subsetInd], x = x, axh = axh, width = scatterwidth, returnx = True, **kwargs)
        outx[subsetInd] = subsetx

    if (~subsetInd).sum() > 0:
        kwargs['c'] = altDotcolor
        subsetx = scatterdots(data[~subsetInd], x = x, axh = axh, width = scatterwidth, returnx = True, **kwargs)
        outx[~subsetInd] = subsetx

    if returnx:
        return outx

def manyboxplots(df, cols=None, axh=None, colLabels=None,annotation='N',horizontal=False,vRange=None,xRot=0, **kwargs):
    """Series of boxplots along x-axis (or flipped horizontally along y-axis [NOT IMPLEMENTED])

    WORK IN PROGRESS

    Optionally add annotation for each boxplot with:
        (1) "N"
        (2) "pctpos" (response rate, by additionally specifying responders)
            NOT YET IMPLEMENTED

    Parameters
    ----------
    df : pd.DataFrame
    cols : list
        Column names to be plotted
    axh : matplotlib figure handle
        If None then use plt.gca()
    colLabels : list
        Column labels (optional)
    annotation : str or None
        Specifies what the annotation should be: "N" or "pctpos"
    horizontal : bool
        Specifies whether boxplots should be vertical (default, False) or horizontal (True)
    kwargs : additional arguments
        Passed to myboxplot function to specify colors etc."""
    if axh is None:
        axh = plt.gca()
    if cols is None:
        cols = df.columns
    if colLabels is None:
        colLabels = cols
    elif len(colLabels)<cols:
        colLabels += cols[len(colLabels):]

    for x, c in enumerate(cols):
        myboxplot(df[c].dropna(), x = x, axh = axh, **kwargs)

    if not vRange is None:        
        plt.ylim(vRange)
    yl = plt.ylim()
    annotationKwargs = dict(xytext = (0, -10), textcoords = 'offset points', ha = 'center', va = 'top', size = 'medium')
    for x, c in enumerate(cols):
        tmp = df[c].dropna()

        if annotation == 'N':
            plt.annotate('%d' % len(tmp), xy = (x, yl[1]), **annotationKwargs)
        elif annotation == 'pctpos':
            pass
        
    plt.xlim((-1, x+1))
    plt.xticks(np.arange(x+1))
    xlabelsL = axh.set_xticklabels(colLabels, fontsize='large', rotation=xRot, fontname='Consolas')

def swarmbox(x, y, data, hue=None, palette=None, order=None, hue_order=None, connect=False, connect_on=[], legend_loc=0, legend_bbox=None, swarm_alpha=1, swarm_size=5, box_alpha=1, box_edgecolor='k', box_facewhite=False):
    """Based on seaborn boxplots and swarmplots.
    Adds the option to connect dots by joining on an identifier columns"""
    if palette is None and not hue is None:
        palette = sns.color_palette('Set2',  n_colors=data[hue].unique().shape[0])
    if hue_order is None and not hue is None:
        hue_order = sorted(data[hue].unique())
    if order is None:
        order = sorted(data[x].unique())
        
    params = dict(data=data, x=x, y=y, hue=hue, order=order, hue_order=hue_order)
    box_axh = sns.boxplot(**params,
                            fliersize=0,
                            linewidth=1,
                            palette=palette)
    for patch in box_axh.artists:
        patch.set_edgecolor((0, 0, 0, 1))
        r, g, b, a = patch.get_facecolor()
        if box_facewhite:
            patch.set_facecolor((1, 1, 1, 1))
        else:
            patch.set_facecolor((r, g, b, box_alpha))
    for line in box_axh.lines:
        line.set_color(box_edgecolor)

    swarm = sns.swarmplot(**params,
                            linewidth=0.5,
                            edgecolor='black',
                            dodge=True,
                            alpha=swarm_alpha,
                            size=swarm_size,
                            palette=palette)
    if connect and not hue is None:
        for i in range(len(hue_order) - 1):
            """Loop over pairs of hues (i.e. grouped boxes)"""
            curHues = hue_order[i:i+2]
            """Pull out just the swarm collections that are needed"""
            zipper = [order] + [swarm.collections[i::len(hue_order)], swarm.collections[i+1::len(hue_order)]]
            for curx, cA, cB in zip(*zipper):
                """Loop over the x positions (i.e. outer groups)"""
                indA = (data[x] == curx) & (data[hue] == curHues[0])
                indB = (data[x] == curx) & (data[hue] == curHues[1])
                
                """Locate the data and match it up with the points plotted for each hue"""
                tmpA = data[[x, hue, y] + connect_on].loc[indA].dropna()
                tmpB = data[[x, hue, y] + connect_on].loc[indB].dropna()
                plottedA = cA.get_offsets() # shaped (n_elements x 2)
                plottedB = cB.get_offsets()
                
                """Merge the data from each hue, including the new detangled x coords,
                based on what was plotted"""
                tmpA.loc[:, '_untangi'] = untangle(tmpA[y].values.astype(float), plottedA[:, 1])
                tmpB.loc[:, '_untangi'] = untangle(tmpB[y].values.astype(float), plottedB[:, 1])
                tmpA.loc[:, '_newx'] = plottedA[:, 0][tmpA['_untangi'].values]
                tmpB.loc[:, '_newx'] = plottedB[:, 0][tmpB['_untangi'].values]
                """Using 'inner' drops the data points that are in one hue grouping and not the other"""
                tmp = pd.merge(tmpA, tmpB, left_on=connect_on, right_on=connect_on, suffixes=('_A', '_B'), how='inner')
                """Plot them one by one"""
                for rind, r in tmp.iterrows():
                    plt.plot(r[['_newx_A', '_newx_B']],
                             r[[y + '_A', y + '_B']],
                             '-', color='gray', linewidth=0.5)
    elif connect and not order is None:
        for i in range(len(order) - 1):
            """Loop over pairs of hues (i.e. grouped boxes)"""
            cur_orders = order[i:i+2]
            """Pull out just the swarm collections that are needed"""
            c_a = swarm.collections[i]
            c_b = swarm.collections[i + 1]
            ind_a = (data[x] == cur_orders[0])
            ind_b = (data[x] == cur_orders[1])
                
            """Locate the data and match it up with the points plotted for each hue"""
            tmp_a = data[[x, y] + connect_on].loc[ind_a].dropna()
            tmp_b = data[[x, y] + connect_on].loc[ind_b].dropna()
            plotted_a = swarm.collections[i].get_offsets() # shaped (n_elements x 2)
            plotted_b = swarm.collections[i + 1].get_offsets()
                
            """Merge the data from each hue, including the new detangled x coords,
            based on what was plotted"""
            tmp_a.loc[:, '_untangi'] = untangle(tmp_a[y].values.astype(float), plotted_a[:, 1])
            tmp_b.loc[:, '_untangi'] = untangle(tmp_b[y].values.astype(float), plotted_b[:, 1])
            tmp_a.loc[:, '_newx'] = plotted_a[:, 0][tmp_a['_untangi'].values]
            tmp_b.loc[:, '_newx'] = plotted_b[:, 0][tmp_b['_untangi'].values]
            """Using 'inner' drops the data points that are in one hue grouping and not the other"""
            tmp = pd.merge(tmp_a, tmp_b, left_on=connect_on, right_on=connect_on, suffixes=('_A', '_B'), how='inner')
            """Plot them one by one"""
            for rind, r in tmp.iterrows():
                plt.plot(r[['_newx_A', '_newx_B']],
                         r[[y + '_A', y + '_B']],
                         '-', color='gray', linewidth=0.5)
    if not hue is None and not legend_loc is None:
        plt.legend([plt.Circle(1, color=c, alpha=1) for c in palette], hue_order, title=hue, loc=legend_loc, bbox_to_anchor=legend_bbox)
    if legend_loc is None:
        plt.gca().legend_.remove()

def _xspacing(v, mxWidth=0.3, idealNumPoints=4):
    xlim = min(mxWidth, (len(v)/idealNumPoints)*mxWidth/2)
    x = np.linspace(-xlim, xlim, len(v))
    x = np.random.permutation(x)
    """Use v*0 so that it has the right labels for apply"""
    return v*0 + x

def _yjitter(v, jitter=0.3):
    y = np.linspace(-jitter/2, jitter/2, len(v))
    y = np.random.permutation(y)
    return y + v

def discrete_boxplot(x, y, hue, data, yjitter=0.3, palette=None, order=None, hue_order=None, IQR=True, mean_df=None, pvalue_df=None):
    if order is None:
        order = data[x].unique()
    if len(order) == 1:
        xspacing = 2
    else:
        xspacing = 1
    if hue_order is None:
        hue_order = data[hue].unique()
    if palette is None:
        palette = [c for i,c in zip(range(len(hue_order)), cycle(mpl.cm.Set1.colors))]
    
    yl = (data[y].min() - 0.5, data[y].max() + 0.5) 

    plotx = 0
    xt = []
    xtl = []
    for xval in order:
        xcoords = []
        xcoords_labels = {}
        for hueval, color in zip(hue_order, palette):
            tmp = data.loc[(data[hue] == hueval) & (data[x] == xval), y]
            if mean_df is None:
                if IQR:
                    lcl, mu, ucl = np.percentile(tmp.values, [25, 50, 75])
                else:
                    if NUMBA:
                        """bootci_nb requires a 2D matrix and will operate along rows. statfunction needs to return a vector"""
                        mu, lcl, ucl = bootci_nb(tmp.values[:, None], statfunction=_keepdims_mean, alpha=0.05, n_samples=10000, method='bca').ravel()
                    else:
                        lcl, ucl = ci(tmp.values, statfunction=np.mean, n_samples=10000, method='bca')
                        mu = np.mean(tmp.values)
            else:
                mu, lcl, ucl = mean_df.loc[(mean_df[hue] == hueval) & (mean_df[x] == xval)].iloc[0][['mean', 'lcl', 'ucl']]

            plt.errorbar(x=plotx,
                         y=mu,
                         yerr=np.array([mu - lcl, ucl - mu])[:, None],
                         fmt='s-',
                         color=color,
                         lw=2)
            if yjitter > 0:
                yvec = _yjitter(tmp.values, jitter=yjitter)
            else:
                yvec = tmp.values
            xvec = _xspacing(tmp.values)
            plt.scatter(xvec + plotx, yvec, s=20, alpha=0.4, color=color, edgecolor='black', linewidth=1)
            xcoords.append(plotx)
            xcoords_labels[hueval] = plotx
            plotx += xspacing
        if not pvalue_df is None:
            for ann_y, (_, r) in enumerate(pvalue_df.loc[pvalue_df[x] == xval].iterrows()):
                if r['significant'] == 1:
                    stl, enl = r[hue].split(' - ')
                    stx, enx = xcoords_labels[stl], xcoords_labels[enl]
                    plt.plot((stx, enx), (yl[1] + ann_y, yl[1] + ann_y), '-', color='k', lw=2)
                    plt.plot((stx, stx), (yl[1] + ann_y, yl[1] + ann_y - 0.15), '-', color='k', lw=2)
                    plt.plot((enx, enx), (yl[1] + ann_y, yl[1] + ann_y - 0.15), '-', color='k', lw=2)
                    plt.annotate('p = %1.3f' % r['pvalue'],
                                 xy=(np.min([enx, stx]) + np.abs(enx - stx)/2, yl[1] + ann_y),
                                 va='bottom', ha='center', size=12,
                                 textcoords='offset points', xytext=(0,1))
        xt.append(np.median(xcoords))
        xtl.append(xval)
        plotx += xspacing
    
    plt.ylabel(y)
    if len(order) > 1:
        plt.xticks(xt, xtl)
        plt.xlabel(x)
    else:
        plt.xticks(xcoords, hue_order, rotation=45)
    plt.xlim((-1, np.max(xcoords) + 1))
    plt.ylim((yl[0], yl[1] + len(hue_order)))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend([plt.Rectangle((0,0), 1, 1, color=c) for c in palette],
               hue_order,
               loc='upper left', bbox_to_anchor=(1,1))
