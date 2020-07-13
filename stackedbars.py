import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib as mpl
import numpy as np
import pandas as pd

__all__ = ['stackedbars']

def stackedbars(data, stack, x, y, hue, facetrow=None, stackorder=None, xorder=None, hueorder=None, facetorder=None, nhues=None, colors=None, handle=None, legend=True, yrange=None):
    """Create a stacked bar plot.

    Parameters
    ----------
    data : pd.DataFrame (longform)
        Must contain column for stack, x, y, hue amd facetrow
    stack : str
        Column in data identifying each stack of bars
    x : str
        Column in data designating the groups of stacks
    y : str
        Column in data to plot on the y-axis
    hue : str
        Column in data representing the bars to stack
    facetrow : str
        Optional, column in data that is an extra grouping variable
        for rows of plots.
    stackorder : list
        Order for stacks within groups, default is sorted within each group
    xorder : list
        Order for groupings along x-axis
    hueorder : list
        Order for colors within a stack, default is sorted by overall mean
    facetorder : list
        Order for rows, top to bottom
    nhues : int
        Number of hues to plot.
    colors : list
        Colors for hue variable
    handle : axes or figure handle
        Figure or axes to plot into
    legend : bool
        Display legend.

    Returns
    -------
    axh : handle
        Matplotlib axes or figure handle depending
        on whether or not it draws a multi-axis figure"""

    if xorder is None:
        xorder = sorted(data[x].unique())

    if facetrow is None:
        dd = data[[stack, x]].drop_duplicates()
        xsizes = dd.groupby(x)[stack].agg(lambda v: v.unique().shape[0])
    else:
        dd = data[[stack, x, facetrow]].drop_duplicates()
        xsizes = dd.groupby([x, facetrow])[stack].agg(lambda v: v.unique().shape[0]).unstack(facetrow).max(axis=1)
    
    xcenters = (xsizes + 2).loc[xorder].cumsum()
    xlefts = xcenters - xsizes/2

    xl = [None, None]
    yl = [None, None]
    if hueorder is None:
        h = data.groupby(hue)[y].agg(np.mean)
        hueorder = h.sort_values(ascending=False).index.tolist()
        if not nhues is None:
            hueorder = hueorder[:nhues]
    if colors is None:
        colors = mpl.cm.Set3.colors[:min(12, len(hueorder))]
    if not facetrow is None:
        if facetorder is None:
            facetorder = sorted(data[facetrow].unique())
    else:
        facetorder = [0]

    if handle is None and facetrow is None:
        figh = plt.gcf()
    elif handle is None:
        figh = plt.gcf()
        figh.clf()
    elif not facetrow is None:
        figh = handle
        figh.clf()
            
    if legend:
        """Leave room for the legend"""
        gs = GridSpec(nrows=len(facetorder), ncols=1, right=0.6)
    else:
        gs = GridSpec(nrows=len(facetorder), ncols=1, right=0.9)

    stack_labels = []
    axesHandles = []
    for rowi, row in enumerate(facetorder):
        if not facetrow is None:
            axh = plt.subplot(gs[rowi])
        else:
            axh = plt.gca()
            axh.cla()
        axesHandles.append(axh)
        if not facetrow is None:
            rowData = data.loc[data[facetrow] == row]
        else:
            rowData = data

        for xi, curx in enumerate(xorder):
            plotDf = rowData.loc[rowData[x] == curx].set_index([stack, hue])[y].unstack(hue)
            if plotDf.shape[0] > 0:
                if stackorder is None:
                    ho = [h for h in hueorder if h in plotDf.columns]
                    plotDf = plotDf[ho].sort_values(by=ho, ascending=False)
                    sorder = plotDf.index
                else:
                    sorder = stackorder

                for stacki, s in enumerate(sorder):
                    bottom = 0
                    for bari, barl in enumerate(hueorder):
                        if barl in plotDf.columns and not np.isnan(plotDf.loc[s, barl]):
                            axh.bar(x=(xcenters[curx] - len(sorder)/2) + stacki,
                                    bottom=bottom,
                                    height=plotDf.loc[s, barl],
                                    width=0.9,
                                    color=colors[bari])
                            # print(curx, s, barl, bottom, plotDf.loc[s, barl])
                            bottom += plotDf.loc[s, barl]

                    if xi == 0 and rowi == 0:
                        stack_labels.append(((xcenters[curx] - len(sorder)/2) + stacki, s))
        
        curxl = plt.xlim()
        curyl = plt.ylim()

        if xl[0] is None or curxl[0] < xl[0]:
            xl[0] = curxl[0]
        if xl[1] is None or curxl[1] > xl[1]:
            xl[1] = curxl[1]
        if yl[0] is None or curyl[0] < yl[0]:
            yl[0] = curyl[0]
        if yl[1] is None or curyl[1] > yl[1]:
            yl[1] = curyl[1]

        if rowi == len(facetorder) - 1:
            plt.xlabel(x)
            """Applied this -0.5 fix to work for two stacks but not sure about more"""
            plt.xticks(xcenters - 0.5*len(sorder)/2, xorder)
        else:
            plt.xticks(xcenters - 0.5*len(sorder)/2, ['']*len(xcenters))

        if facetrow is None:
            plt.ylabel(y)
        else:
            plt.ylabel(row)

        if rowi == 0 and legend:
            plt.legend([plt.Rectangle((0,0), 1, 1, color=c) for c in colors][::-1],
                        hueorder[::-1], 
                        loc='upper left',
                        bbox_to_anchor=(1, 1),
                        title=hue)
    if not yrange is None:
        yl = yrange
    for axh in axesHandles:
        axh.set_ylim(yl)
        axh.set_xlim(xl)
        axh.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator())

    if facetrow is None:
        handle = axh
    else:
        handle = figh
        plt.annotate(xy=(0.02, 0.5), s=y,
                     xycoords='figure fraction',
                     horizontalalignment='center',
                     verticalalignment='center',
                     rotation='vertical')
    for x, s in stack_labels:
        plt.sca(axesHandles[0])
        plt.annotate(xy=(x, yl[1]), s=s,
                     xycoords='data',
                     textcoords='offset points',
                     xytext=(0, 2),
                     horizontalalignment='center',
                     verticalalignment='bottom',
                     rotation='vertical')
    return handle