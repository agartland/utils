
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable
import itertools
from functools import partial
from scipy.spatial.distance import squareform

from matplotlib.gridspec import GridSpec
from matplotlib import cm
import scipy.cluster.hierarchy as sch
from corrplots import validPairwiseCounts, partialcorr, combocorrplot
import statsmodels.api as sm
from scipy import stats
import seaborn as sns
from objhist import objhist

from sklearn.impute import SimpleImputer

sns.set(style='darkgrid', palette='muted', font_scale=1.75)

__all__ = ['corrDmatFunc',
            'corrTDmatFunc',
            'hierClusterFunc',
            'plotHierClust',
            'combocorrplot',
            'plotHeatmap',
            'pivotDiagnostic']
"""
Example:
dmatDf = corrDmatFunc(df, metric='pearson-signed', dfunc=None, minN=10)
labels, Z = hierClusterFunc(dmatDf, K=6, method='complete', returnLinkageMat=True)
plotHierClust(dmatDf, Z, labels=labels, titleStr=None, vRange=None, tickSz='small', cmap=None, cmapLabel='')
"""

def imputeNA(df, strategy='median', axis=0, copy=True):
    imp = SimpleImputer(strategy=strategy, copy=copy)
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
            dmat = ((1 - df.corr(method = metric.replace('-signed', ''), min_periods = minN).values) / 2)
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
                    tmpdf = df.iloc[:, [i, j]]
                    tmpdf = tmpdf.dropna()
                    if tmpdf.shape[0] >= minN:
                        d = dfunc(df.iloc[:, i], df.iloc[:, j])
                    else:
                        d = np.nan
                    dmat[i, j] = d
                    dmat[j, i] = d
    return pd.DataFrame(dmat, columns = df.columns, index = df.columns)

def hierClusterFunc(dmatDf, K=6, method='complete', returnLinkageMat=False):
    hclusters = sch.linkage(squareform(dmatDf.values, force='tovector'),
                            method=method)
    labelsVec = sch.fcluster(hclusters, K, criterion='maxclust')
    labels = pd.Series(labelsVec, index=dmatDf.columns)
    if not returnLinkageMat:
        return labels
    else:
        return labels, hclusters

def _colors2labels(labels, setStr='Set1', cmap=None, freqSort=True):
    """Return pd.Series of colors based on labels"""
    if freqSort:
        oh = objhist(labels)
        uLabels = sorted(np.unique(labels), key=oh.get, reverse=True)
    else:
        uLabels = sorted(np.unique(labels))
    if cmap is None:
        N = max(3, min(9, len(uLabels)))
        cmap = palettable.colorbrewer.get_map(setStr, 'Qualitative', N).mpl_colors
    cmapLookup = {k:col for k, col in zip(uLabels, itertools.cycle(cmap))}
    if isinstance(labels, pd.Series):
        return labels.map(cmapLookup.get)
    else:
        return [cmapLookup[v] for v in labels]

def _clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in list(ax.spines.values()):
        sp.set_visible(False)
    ax.grid(False)
    ax.set_facecolor('white')

def plotHeatmap(df, row_labels=None, col_labels=None, titleStr=None, vRange=None, tickSz='small', cmap=None, cmapLabel='', annotation=False, xtickRot=90, xtickLabels=False, ytickLabels=False, row_cmap=None, col_cmap=None):
    """Display a heatmap with labels."""
    if vRange is None:
        vmin = np.min(np.ravel(df.values))
        vmax = np.max(np.ravel(df.values))
    else:
        vmin, vmax = vRange
    
    if cmap is None:
        if vmin < 0 and vmax > 0 and vmax <= 1 and vmin >= -1:
            cmap = cm.RdBu_r
        else:
            cmap = cm.YlOrRd

    fig = plt.gcf()
    fig.clf()

    if row_labels is None and col_labels is None:
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.05, right=0.78, top=0.90)[0, 0])
        scale_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.87, bottom=0.05, right=0.93, top=0.85)[0, 0])
        outAX = {'heatmap':heatmapAX,
                'scale':scale_cbAX}
    elif col_labels is None:
        row_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.05, right=0.11, top=0.90)[0, 0])
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.12, bottom=0.05, right=0.78, top=0.90)[0, 0])
        scale_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.87, bottom=0.05, right=0.93, top=0.90)[0, 0])
        outAX = {'heatmap':heatmapAX,
                 'scale':scale_cbAX,
                 'rowCB':row_cbAX}
    elif row_labels is None:
        col_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.05, right=0.78, top=0.09)[0, 0])
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.1, right=0.78, top=0.90)[0, 0])
        scale_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.87, bottom=0.05, right=0.93, top=0.90)[0, 0])
        outAX = {'heatmap':heatmapAX,
                 'scale':scale_cbAX,
                 'colCB':col_cbAX}
    else:
        row_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.1, right=0.11, top=0.90)[0, 0])
        col_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.12, bottom=0.05, right=0.78, top=0.09)[0, 0])
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.12, bottom=0.1, right=0.78, top=0.90)[0, 0])
        scale_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.87, bottom=0.05, right=0.93, top=0.90)[0, 0])
        outAX = {'heatmap':heatmapAX,
                 'scale':scale_cbAX,
                 'colCB':col_cbAX,
                 'rowCB':row_cbAX}

    my_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    if not row_labels is None:
        row_cbSE = np.concatenate(_colors2labels(row_labels, cmap=row_cmap, freqSort=False).values).reshape((row_labels.shape[0], 1, 3))
        axi = row_cbAX.imshow(row_cbSE, interpolation='nearest', aspect='auto', origin='lower')
        _clean_axis(row_cbAX)
    if not col_labels is None:
        col_cbSE = np.concatenate(_colors2labels(col_labels, freqSort=False).values, cmap=col_cmap).reshape((1, col_labels.shape[0], 3))
        axi = col_cbAX.imshow(col_cbSE, interpolation='nearest', aspect='auto', origin='lower')
        _clean_axis(col_cbAX)

    """Heatmap plot"""
    axi = heatmapAX.imshow(df.values, interpolation='nearest', aspect='auto', origin='lower', norm=my_norm, cmap=cmap)
    _clean_axis(heatmapAX)

    if annotation:
        for i, j in itertools.product(list(range(df.shape[0])), list(range(df.shape[1]))):
            v = df.values[i, j]
            heatmapAX.annotate('%1.2f' % v, xy=(i, j), size='x-large', weight='bold', color='white', ha='center', va='center')

    """Column tick labels along the rows"""
    if tickSz is None:
        heatmapAX.set_yticks(())
        heatmapAX.set_xticks(())
    else:
        if ytickLabels:
            heatmapAX.set_yticks(np.arange(df.shape[0]))
            heatmapAX.yaxis.set_ticks_position('right')
            heatmapAX.set_yticklabels(df.index, fontsize=tickSz, fontname='Consolas')
            for l in heatmapAX.get_yticklines():
                l.set_markersize(0)
        else:
            heatmapAX.set_yticks(())


        if xtickLabels:
            """Column tick labels"""
            heatmapAX.set_xticks(np.arange(df.shape[1]))
            heatmapAX.xaxis.set_ticks_position('top')
            xlabelsL = heatmapAX.set_xticklabels(df.columns, fontsize=tickSz, rotation=xtickRot, fontname='Consolas')
            for l in heatmapAX.get_xticklines():
                l.set_markersize(0)
        else:
            heatmapAX.set_xticks(())

    """Add a colorbar"""
    cb = fig.colorbar(axi, scale_cbAX) # note that we could pass the norm explicitly with norm=my_norm
    cb.set_label(cmapLabel)
    """Make colorbar labels smaller"""
    """for t in cb.ax.yaxis.get_ticklabels():
        t.set_fontsize('small')"""

    """Add title as xaxis label"""
    if not titleStr is None:
        heatmapAX.set_xlabel(titleStr, size='x-large')
    plt.show()
    return outAX

def plotHierClust(dmatDf, Z, labels=None, titleStr=None, vRange=None, tickSz='small', cmap=None, cmapLabel='', plotLegend=False, plotColorbar=True):
    """Display a hierarchical clustering result."""
    if vRange is None:
        vmin = np.min(np.ravel(dmatDf.values))
        vmax = np.max(np.ravel(dmatDf.values))
    else:
        vmin, vmax = vRange
    
    if cmap is None:
        if vmin < 0 and vmax > 0 and vmax <= 1 and vmin >= -1:
            cmap = cm.RdBu_r
        else:
            cmap = cm.YlOrRd

    fig = plt.gcf()
    fig.clf()

    if labels is None:
        denAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.05, right=0.15, top=0.85)[0, 0])
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.16, bottom=0.05, right=0.78, top=0.85)[0, 0])
    else:
        denAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.05, right=0.15, top=0.85)[0, 0])
        cbAX = fig.add_subplot(GridSpec(1, 1, left=0.16, bottom=0.05, right=0.19, top=0.85)[0, 0])
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.2, bottom=0.05, right=0.78, top=0.85)[0, 0])

    if plotColorbar:
        scale_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.87, bottom=0.05, right=0.93, top=0.85)[0, 0])

    my_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    """Dendrogaram along the rows"""
    plt.sca(denAX)
    denD = sch.dendrogram(Z, color_threshold=np.inf, orientation='left')
    colInd = denD['leaves']
    _clean_axis(denAX)

    if not labels is None:
        cbSE = _colors2labels(labels, freqSort=False)
        axi = cbAX.imshow([[x] for x in cbSE.iloc[colInd].values], interpolation='nearest', aspect='auto', origin='lower')
        _clean_axis(cbAX)
        if plotLegend:
            uLabels = np.unique(labels)
            handles = [mpl.patches.Patch(facecolor=c, edgecolor='k') for c in _colors2labels(uLabels, freqSort=False)]
            # fig.legend(handles, uLabels, loc=(0, 0), title=labels.name)
            # bbox = mpl.transforms.Bbox(((0,0),(1,1))).anchored('NE')
            fig.legend(handles, uLabels, loc='upper left', title=labels.name)
            

    """Heatmap plot"""
    axi = heatmapAX.imshow(dmatDf.values[colInd,:][:, colInd], interpolation='nearest', aspect='auto', origin='lower', norm=my_norm, cmap=cmap)
    _clean_axis(heatmapAX)

    """Column tick labels along the rows"""
    if tickSz is None:
        heatmapAX.set_yticks(())
        heatmapAX.set_xticks(())
    else:
        heatmapAX.set_yticks(np.arange(dmatDf.shape[1]))
        heatmapAX.yaxis.set_ticks_position('right')
        heatmapAX.set_yticklabels(dmatDf.columns[colInd], fontsize=tickSz, fontname='Consolas')

        """Column tick labels"""
        heatmapAX.set_xticks(np.arange(dmatDf.shape[1]))
        heatmapAX.xaxis.set_ticks_position('top')
        xlabelsL = heatmapAX.set_xticklabels(dmatDf.columns[colInd], fontsize=tickSz, rotation=90, fontname='Consolas')

        """Remove the tick lines"""
        for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
            l.set_markersize(0)

    """Add a colorbar"""
    if plotColorbar:
        cb = fig.colorbar(axi, scale_cbAX) # note that we could pass the norm explicitly with norm=my_norm
        cb.set_label(cmapLabel)
        """Make colorbar labels smaller"""
        for t in cb.ax.yaxis.get_ticklabels():
            t.set_fontsize('small')

    """Add title as xaxis label"""
    if not titleStr is None:
        heatmapAX.set_xlabel(titleStr, size='x-large')
    plt.show()


def pivotDiagnostic(df, index, columns, values):
    """Attempt to pivot the DataFrame. If there are duplicate
    entries in the pivot table then return a df of duplicates

    Parameters
    ----------
    df : pd.DataFrame
    index : str
    columns : str or list
    values : str or list

    Returns
    -------
    df :pd.DataFrame
        If no exception then returns pivot table,
        otherwise returns duplicate rows in df."""

    try:
        p = df.pivot(index=index, columns=columns, values=values)
        print('Pivot success: return pivoted df')
        return p
    except ValueError:
        if type(columns) in [str, str]:
            columns = [columns]
        if not isinstance(columns, list):
            columns = list(columns)

        if type(values) in [str, str]:
            values = [values]
        if not isinstance(values, list):
            values = list(values)

        dupInd = df[[index] + columns].duplicated(keep=False)
        print('Index contains %d duplicate entries, cannot reshape' % dupInd.sum())
        print('Returning duplicate rows')
        return df[[index] + columns + values].loc[dupInd].sort_values(by=[index] + columns + values) 
