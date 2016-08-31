from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable
import itertools
from functools import partial

from matplotlib.gridspec import GridSpec
from matplotlib import cm
import scipy.cluster.hierarchy as sch
from corrplots import validPairwiseCounts, partialcorr,combocorrplot
import statsmodels.api as sm
from scipy import stats
import seaborn as sns

sns.set(style='darkgrid', palette='muted', font_scale=1.75)

__all__ = ['corrDmatFunc',
            'hierClusterFunc',
            'plotHierClust',
            'combocorrplot',
            'plotHeatmap']
"""
Example:
dmatDf = corrDmatFunc(df, metric='pearson-signed', dfunc=None, minN=10)
labels, Z = hierClusterFunc(dmatDf, K=6, method='complete', returnLinkageMat=True)
plotHierClust(dmatDf, Z, labels=labels, titleStr=None, vRange=None, tickSz='small', cmap=None, cmapLabel='')
"""

def imputeNA(df, strategy='median', axis=0, copy=True):
    imp = Imputer(strategy=strategy, axis=axis, copy=copy)
    return pd.DataFrame(imp.fit_transform(df.values), columns=df.columns, index=df.index)

def corrTDmatFunc(df, *args, **kwargs):
    return corrDmatFunc(df.T, *args, **kwargs)

def corrDmatFunc(df, metric='pearson-signed', dfunc=None, minN=10):
    if dfunc is None:
        if metric in ['spearman', 'pearson']:
            """Anti-correlations are also considered as high similarity and will cluster together"""
            dmat = 1 - df.corr(method = metric, min_periods = minN).values**2
            dmat[np.isnan(dmat)] = 1
        elif metric in ['spearman-signed', 'pearson-signed']:
            """Anti-correlations are considered as dissimilar and will NOT cluster together"""
            dmat = ((1 - df.corr(method = metric.replace('-signed',''), min_periods = minN).values) / 2)
            dmat[np.isnan(dmat)] = 1
        else:
            raise NameError('metric name not recognized')
    else:
        ncols = df.shape[1]
        dmat = np.zeros((ncols, ncols))
        for i in range(ncols):
            for j in range(ncols):
                """Assume distance is symetric"""
                if i <= j:
                    tmpdf = df.iloc[:,[i,j]]
                    tmpdf = tmpdf.dropna()
                    if tmpdf.shape[0] >= minN:
                        d = dfunc(df.iloc[:,i],df.iloc[:,j])
                    else:
                        d = np.nan
                    dmat[i,j] = d
                    dmat[j,i] = d
    return pd.DataFrame(dmat, columns = df.columns, index = df.columns)

def hierClusterFunc(dmatDf, K=6, method='complete', returnLinkageMat=False):
    hclusters = sch.linkage(dmatDf.values, method = method)
    labelsVec = sch.fcluster(hclusters, K, criterion = 'maxclust')
    labels = pd.Series(labelsVec, index = dmatDf.columns)
    if not returnLinkageMat:
        return labels
    else:
        return labels, hclusters

def _colors2labels(labels, setStr='Set1', cmap=None):
    """Return pd.Series of colors based on labels"""
    uLabels = sorted(np.unique(labels))
    if cmap is None:
        N = max(3, min(12, len(uLabels)))
        cmap = palettable.colorbrewer.get_map(setStr,'Qualitative',N).mpl_colors
    cmapLookup = {k:col for k,col in zip(uLabels, itertools.cycle(cmap))}
    if type(labels) is pd.Series:
        return labels.map(cmapLookup.get)
    else:
        return [cmapLookup[v] for v in labels]

def _clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    ax.set_axis_bgcolor('white')

def plotHeatmap(df, labels=None, titleStr=None, vRange=None, tickSz='small', cmap=None, cmapLabel='', annotation=False, xtickRot=90):
    """Display a heatmap with labels."""
    if vRange is None:
        vmin = np.min(np.ravel(df.values))
        vmax = np.max(np.ravel(df.values))
    else:
        vmin,vmax = vRange
    
    if cmap is None:
        if vmin < 0 and vmax > 0 and vmax <= 1 and vmin >= -1:
            cmap = cm.RdBu_r
        else:
            cmap = cm.YlOrRd

    fig = plt.gcf()
    fig.clf()

    if labels is None:
        heatmapAX = fig.add_subplot(GridSpec(1,1,left=0.05,bottom=0.05,right=0.78,top=0.85)[0,0])
        scale_cbAX = fig.add_subplot(GridSpec(1,1,left=0.87,bottom=0.05,right=0.93,top=0.85)[0,0])
    else:
        cbAX = fig.add_subplot(GridSpec(1,1,left=0.05,bottom=0.05,right=0.09,top=0.85)[0,0])
        heatmapAX = fig.add_subplot(GridSpec(1,1,left=0.1,bottom=0.05,right=0.78,top=0.85)[0,0])
        scale_cbAX = fig.add_subplot(GridSpec(1,1,left=0.87,bottom=0.05,right=0.93,top=0.85)[0,0])

    my_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if not labels is None:
        cbSE = _colors2labels(labels)
        axi = cbAX.imshow(cbSE.values, interpolation='nearest', aspect='auto', origin='lower')
        _clean_axis(cbAX)

    """Heatmap plot"""
    axi = heatmapAX.imshow(df.values, interpolation='nearest', aspect='auto', origin='lower', norm=my_norm, cmap=cmap)
    _clean_axis(heatmapAX)

    if annotation:
        for i,j in itertools.product(range(df.shape[0]), range(df.shape[1])):
            v = df.values[i,j]
            heatmapAX.annotate('%1.2f' % v, xy=(i,j), size='x-large', weight='bold', color='white',ha='center',va='center')

    """Column tick labels along the rows"""
    if tickSz is None:
        heatmapAX.set_yticks(())
        heatmapAX.set_xticks(())
    else:
        heatmapAX.set_yticks(np.arange(df.shape[0]))
        heatmapAX.yaxis.set_ticks_position('right')
        heatmapAX.set_yticklabels(df.index, fontsize=tickSz, fontname='Consolas')

        """Column tick labels"""
        heatmapAX.set_xticks(np.arange(df.shape[1]))
        heatmapAX.xaxis.set_ticks_position('top')
        xlabelsL = heatmapAX.set_xticklabels(df.columns, fontsize=tickSz, rotation=xtickRot, fontname='Consolas')

        """Remove the tick lines"""
        for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
            l.set_markersize(0)

    """Add a colorbar"""
    cb = fig.colorbar(axi,scale_cbAX) # note that we could pass the norm explicitly with norm=my_norm
    cb.set_label(cmapLabel)
    """Make colorbar labels smaller"""
    """for t in cb.ax.yaxis.get_ticklabels():
        t.set_fontsize('small')"""

    """Add title as xaxis label"""
    if not titleStr is None:
        heatmapAX.set_xlabel(titleStr, size='x-large')
    plt.show()

def plotHierClust(dmatDf, Z, labels=None, titleStr=None, vRange=None, tickSz='small', cmap=None, cmapLabel='', plotLegend=False):
    """Display a hierarchical clustering result."""
    if vRange is None:
        vmin = np.min(np.ravel(dmatDf.values))
        vmax = np.max(np.ravel(dmatDf.values))
    else:
        vmin,vmax = vRange
    
    if cmap is None:
        if vmin < 0 and vmax > 0 and vmax <= 1 and vmin >= -1:
            cmap = cm.RdBu_r
        else:
            cmap = cm.YlOrRd

    fig = plt.gcf()
    fig.clf()

    if labels is None:
        denAX = fig.add_subplot(GridSpec(1,1,left=0.05,bottom=0.05,right=0.15,top=0.85)[0,0])
        heatmapAX = fig.add_subplot(GridSpec(1,1,left=0.16,bottom=0.05,right=0.78,top=0.85)[0,0])
        scale_cbAX = fig.add_subplot(GridSpec(1,1,left=0.87,bottom=0.05,right=0.93,top=0.85)[0,0])
    else:
        denAX = fig.add_subplot(GridSpec(1,1,left=0.05,bottom=0.05,right=0.15,top=0.85)[0,0])
        cbAX = fig.add_subplot(GridSpec(1,1,left=0.16,bottom=0.05,right=0.19,top=0.85)[0,0])
        heatmapAX = fig.add_subplot(GridSpec(1,1,left=0.2,bottom=0.05,right=0.78,top=0.85)[0,0])
        scale_cbAX = fig.add_subplot(GridSpec(1,1,left=0.87,bottom=0.05,right=0.93,top=0.85)[0,0])

    my_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    """Dendrogaram along the rows"""
    plt.sca(denAX)
    denD = sch.dendrogram(Z, color_threshold=np.inf, orientation='right')
    colInd = denD['leaves']
    _clean_axis(denAX)

    if not labels is None:
        cbSE = _colors2labels(labels)
        axi = cbAX.imshow([[x] for x in cbSE.iloc[colInd].values],interpolation='nearest',aspect='auto',origin='lower')
        _clean_axis(cbAX)
        if plotLegend:
            uLabels = np.unique(labels)
            handles = [mpl.patches.Patch(facecolor=c, edgecolor='k') for c in _colors2labels(uLabels)]
            # fig.legend(handles, uLabels, loc=(0, 0), title=labels.name)
            # bbox = mpl.transforms.Bbox(((0,0),(1,1))).anchored('NE')
            fig.legend(handles, uLabels, loc='upper left', title=labels.name)
            

    """Heatmap plot"""
    axi = heatmapAX.imshow(dmatDf.values[colInd,:][:,colInd],interpolation='nearest',aspect='auto',origin='lower',norm=my_norm,cmap=cmap)
    _clean_axis(heatmapAX)

    """Column tick labels along the rows"""
    if tickSz is None:
        heatmapAX.set_yticks(())
        heatmapAX.set_xticks(())
    else:
        heatmapAX.set_yticks(np.arange(dmatDf.shape[1]))
        heatmapAX.yaxis.set_ticks_position('right')
        heatmapAX.set_yticklabels(dmatDf.columns[colInd],fontsize=tickSz,fontname='Consolas')

        """Column tick labels"""
        heatmapAX.set_xticks(np.arange(dmatDf.shape[1]))
        heatmapAX.xaxis.set_ticks_position('top')
        xlabelsL = heatmapAX.set_xticklabels(dmatDf.columns[colInd],fontsize=tickSz,rotation=90,fontname='Consolas')

        """Remove the tick lines"""
        for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
            l.set_markersize(0)

    """Add a colorbar"""
    cb = fig.colorbar(axi,scale_cbAX) # note that we could pass the norm explicitly with norm=my_norm
    cb.set_label(cmapLabel)
    """Make colorbar labels smaller"""
    for t in cb.ax.yaxis.get_ticklabels():
        t.set_fontsize('small')

    """Add title as xaxis label"""
    if not titleStr is None:
        heatmapAX.set_xlabel(titleStr,size='x-large')
    plt.show()

