import numpy as np
import matplotlib.pyplot as plt
from numpy.random import permutation,seed
import pandas as pd
#import seaborn as sns

__all__ = ['scatterdots',
           'myboxplot',
           'manyboxplots']

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
    if not type(data) == np.ndarray:
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

    if type(data) is pd.Series:
        data = data.values

    if not subsetInd is None:
        if not (subsetInd.dtype == np.array([0,1], dtype=bool).dtype):
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
                for element in bp.keys():
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

    for x,c in enumerate(cols):
        myboxplot(df[c].dropna(), x = x, axh = axh, **kwargs)

    if not vRange is None:        
        plt.ylim(vRange)
    yl = plt.ylim()
    annotationKwargs = dict(xytext = (0,-10), textcoords = 'offset points', ha = 'center', va = 'top', size = 'medium')
    for x,c in enumerate(cols):
        tmp = df[c].dropna()

        if annotation == 'N':
            plt.annotate('%d' % len(tmp), xy = (x, yl[1]), **annotationKwargs)
        elif annotation == 'pctpos':
            pass
        
    plt.xlim((-1,x+1))
    plt.xticks(np.arange(x+1))
    xlabelsL = axh.set_xticklabels(colLabels, fontsize = 'large', rotation = xRot, fontname = 'Consolas')
