from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import palettable
from sklearn.decomposition import KernelPCA, PCA
from sklearn.preprocessing import Imputer, StandardScaler
import itertools
from functools import partial

from matplotlib.gridspec import GridSpec
from matplotlib import cm
import scipy.cluster.hierarchy as sch

from corrplots import validPairwiseCounts, partialcorr,combocorrplot
import statsmodels.api as sm
from scipy import stats
import sklearn
import seaborn as sns
sns.set(style = 'darkgrid', palette = 'muted', font_scale = 1.75)

__all__ = ['corrDmatFunc',
            'hierClusterFunc',
            'screeplot',
            'biplot',
            'plotHierClust',
            'combocorrplot']

def imputeNA(df, strategy='median', axis=0, copy=True):
    imp = Imputer(strategy=strategy, axis=axis, copy=copy)
    return pd.DataFrame(imp.fit_transform(df.values), columns=df.columns, index=df.index)

def corrTDmatFunc(df, *args, **kwargs):
    return corrDmatFunc(df.T, *args, **kwargs)

def corrDmatFunc(df, metric = 'pearson-signed', dfunc = None, minN = 10):
    if dfunc is None:
        if metric in ['spearman', 'pearson']:
            """Anti-correlations are also considered as high similarity and will cluster together"""
            dmat = (1 - df.corr(method = metric, min_periods = minN).values**2)
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

def hierClusterFunc(dmatDf, K = 6, method = 'complete', returnLinkageMat = False):
    hclusters = sch.linkage(dmatDf.values, method = method)
    labelsVec = sch.fcluster(hclusters, K, criterion = 'maxclust')
    labels = pd.Series(labelsVec, index = dmatDf.columns)
    if not returnLinkageMat:
        return labels
    else:
        return labels, hclusters

def _colors2labels(labels, setStr = 'Set3', cmap = None):
    """Return pd.Series of colors based on labels"""
    if cmap is None:
        N = max(3,min(12,len(np.unique(labels))))
        cmap = palettable.colorbrewer.get_map(setStr,'Qualitative',N).mpl_colors
    cmapLookup = {k:col for k,col in zip(sorted(np.unique(labels)),itertools.cycle(cmap))}
    return labels.map(cmapLookup.get)

def _clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    ax.set_axis_bgcolor('white')

def plotHierClust(dmatDf, Z, labels=None, titleStr=None, vRange=None, tickSz='small', cmap=None, cmapLabel=''):
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

    my_norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)

    """Dendrogaram along the rows"""
    plt.sca(denAX)
    denD = sch.dendrogram(Z, color_threshold=np.inf, orientation='right')
    colInd = denD['leaves']
    _clean_axis(denAX)

    if not labels is None:
        cbSE = _colors2labels(labels)
        axi = cbAX.imshow([[x] for x in cbSE.iloc[colInd].values],interpolation='nearest',aspect='auto',origin='lower')
        
        _clean_axis(cbAX)

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

def _computePCA(df, method='pca', n_components=2, standardize=False, dmatFunc=None):
    if method == 'kpca':
        """By using KernelPCA for dimensionality reduction we don't need to impute missing values"""
        if dmatFunc is None:
            dmatFunc = corrTDmatFunc
        pca = KernelPCA(kernel='precomputed', n_components=n_components)
        dmat = dmatFunc(df).values
        gram = 1 - (dmat / dmat.max())
        xy = pca.fit_transform(gram)
        pca.components_ = pca.alphas_
        pca.explained_variance_ratio_ = pca.lambdas_ / pca.lambdas_.sum()
    elif method == 'pca':
        if standardize:
            normed = StandardScaler().fit_transform(df)
        else:
            normed = df
        pca = PCA(n_components=n_components)
        
        xy = pca.fit_transform(normed)
    return xy, pca

def screeplot(df, method='pca', n_components=10, standardize=False, dmatFunc=None):
    n_components = int(np.min([n_components,df.columns.shape[0]]))
    xy,pca = _computePCA(df, method, n_components, standardize, dmatFunc)
    
    figh = plt.gcf()
    figh.clf()
    axh1 = figh.add_subplot(2,1,1)
    axh1.bar(left=range(n_components),
             height=pca.explained_variance_ratio_[:n_components],
             align='center')
    plt.ylabel('Fraction of\nvariance explained')
    plt.xticks(())

    axh2 = figh.add_subplot(2,1,2)
    for compi in range(n_components):
        bottom = 0
        for dimi,col in zip(range(df.shape[1]), itertools.cycle(palettable.colorbrewer.qualitative.Set3_12.mpl_colors)):
            height = np.abs(pca.components_[compi,dimi]) / np.abs(pca.components_[compi,:]).sum()
            #height = np.abs(pca.components_[compi,dimi])
            axh2.bar(left=compi, bottom=bottom, height=height, align='center', color=col)
            if height > 0.1:
                note = df.columns[dimi].replace(' ','\n')
                note += '(+)' if pca.components_[compi,dimi] >= 0 else '(-)'
                axh2.annotate(note, xy=(compi, bottom+height/2), ha='center', va='center',size='small')
            bottom += height
    plt.xticks(range(n_components),['PCA%d' % (i+1) for i in range(n_components)],rotation=90)
    plt.ylim([0,1])
    plt.ylabel('Fraction of\ncomponent variance')

def biplot(df, labels=None, method='pca', plotLabels=True, plotDims=[0,1], plotVars='all', standardize=False, dmatFunc=None, varThresh=0.2, dropna=False):
    """Perform PCA on df, reducing along columns.
    Plot in two-dimensions.
    Color by labels.
    Method is PCA or KernelPCA with dmatFunc
    """
    if labels is None:
        labels = pd.Series(np.zeros(df.index.shape[0]), index=df.index)
    if plotVars == 'all':
        plotVars = df.columns

    assert labels.shape[0] == df.shape[0]
    assert np.all(labels.index == df.index)

    if dropna:
        keepInd = (~df.isnull()).all(axis=1)
        df = df.loc[keepInd]
        labels = labels.loc[keepInd]

    uLabels = np.unique(labels).tolist()
    n_components = max(plotDims) + 1
    xy,pca = _computePCA(df, method, n_components, standardize, dmatFunc)

    colors = palettable.colorbrewer.get_map('Set1', 'qualitative', min(12,max(3,len(uLabels)))).mpl_colors
    plt.clf()
    figh = plt.gcf()
    axh = figh.add_axes([0.1,0.1,0.8,0.8])
    axh.axis('on')
    figh.set_facecolor('white')
    annotationParams = dict(xytext=(0,5), textcoords='offset points', size='medium')
    alpha = 0.8
    for i,obs in enumerate(df.index):
        if plotLabels:
            axh.annotate(obs, xy = (xy[i,plotDims[0]], xy[i,plotDims[1]]), **annotationParams)
    for labi, lab in enumerate(uLabels):
        col = colors[labi]
        ind = np.where(labels==lab)[0]
        axh.scatter(xy[ind, plotDims[0]], xy[ind, plotDims[1]], marker='o', s=50, alpha=alpha, c=col, label=lab)
        axh.scatter(xy[ind, plotDims[0]].mean(axis=0), xy[ind, plotDims[1]].mean(axis=0), marker='o', s=300, alpha=alpha/1.5, c=col)
    arrowParams = dict(arrowstyle='<-',
                        connectionstyle='Arc3',
                        color='black',
                        lw=1)
    annotationParams = dict(xy=(0,0),
                            textcoords='data',
                            color='black',
                            arrowprops=arrowParams,
                            ha='center',
                            va='center')
    for i,v in enumerate(df.columns):
        if v in plotVars and not method == 'kpca':
            mxx = max(xy[:,plotDims[0]])
            mxy = max(xy[:,plotDims[1]])
            arrowx = pca.components_[plotDims[0],i] * mxx
            arrowy = pca.components_[plotDims[1],i] * mxy
            if (np.abs(arrowx) > varThresh*mxx) or (np.abs(arrowy) > varThresh*mxy):
                axh.annotate(v, xytext=(arrowx,arrowy), **annotationParams)
    plt.xlabel('PCA%d (%1.1f%%)' % (plotDims[0] + 1,pca.explained_variance_ratio_[plotDims[0]] * 100))
    plt.ylabel('PCA%d (%1.1f%%)' % (plotDims[1] + 1,pca.explained_variance_ratio_[plotDims[1]] * 100))
    plt.xticks([0])
    plt.yticks([0])
    if len(uLabels) > 1:
        plt.legend(loc=0)
    plt.draw()