import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import palettable
import scipy
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE, MDS

from kernel_regression import dist2kernel
from objhist import objhist
from plot_ellipse import plot_point_cov

try:
    import tsne
except ImportError:
    import pytsne as tsne
    print("seqdistance: Could not load tsne; falling back on pytsne")

"""Embedding of pairwise-distance matrices."""

__all__ = [ 'embedDistanceMatrix',
            'computePWDist',
            'plotEmbedding',
            'clusteredScatter']

def embedDistanceMatrix(dmatDf, method='kpca', n_components=2):
    """Two-dimensional embedding of sequence distances in dmatDf,
    returning Nx2 x,y-coords: tsne, isomap, pca, mds, kpca, sklearn-tsne"""
    if isinstance(dmatDf, pd.DataFrame):
        dmat = dmatDf.values
    else:
        dmat = dmatDf

    if method == 'tsne':
        xy = tsne.run_tsne(dmat, no_dims=n_components)
    elif method == 'isomap':
        isoObj = Isomap(n_neighbors=10, n_components=n_components)
        xy = isoObj.fit_transform(dmat)
    elif method == 'mds':
        mds = MDS(n_components=n_components,
                  max_iter=3000,
                  eps=1e-9,
                  random_state=15,
                  dissimilarity="precomputed",
                  n_jobs=1)
        xy = mds.fit(dmat).embedding_
        rot = PCA(n_components=n_components)
        xy = rot.fit_transform(xy)
    elif method == 'pca':
        pcaObj = PCA(n_components=None)
        xy = pcaObj.fit_transform(dmat)[:, :n_components]
    elif method == 'kpca':
        pcaObj = KernelPCA(n_components=dmat.shape[0], kernel='precomputed', eigen_solver='dense')
        try:
            gram = dist2kernel(dmat)
        except:
            print('Could not convert dmat to kernel for KernelPCA; using 1 - dmat/dmat.max() instead')
            gram = 1 - dmat / dmat.max()
        xy = pcaObj.fit_transform(gram)[:, :n_components]
    elif method == 'lle':
        lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=n_components, method='standard')
        xy = lle.fit_transform(dist)
    elif method == 'sklearn-tsne':
        tsneObj = TSNE(n_components=n_components, metric='precomputed', random_state=0)
        xy = tsneObj.fit_transform(dmat)
    else:
        print('Method unknown: %s' % method)
        return

    assert xy.shape[0] == dmatDf.shape[0]
    xyDf = pd.DataFrame(xy[:, :n_components], index=dmatDf.index, columns=np.arange(n_components))
    if method == 'kpca':
        """Not sure how negative eigenvalues should be handled here, but they are usually
        small so it shouldn't make a big difference"""
        xyDf.explained_variance_ = pcaObj.lambdas_[:n_components]/pcaObj.lambdas_[pcaObj.lambdas_>0].sum()
    return xyDf

def computePWDist(df, metric='pearson-signed', dfunc=None, minN=10, symetric=True):
    """Compute pairwise distance matrix using correlation or arbitrary function.

    Parameters
    ----------
    df : pd.DataFrame
        Samples along the rows and features along the columns.
    metric : str
        Possible values: pearson-signed, pearson, spearman-signed, spearman
        or any other scipy distance
    dfunc : function(pd.Series, pd.Series)
        Function will override the metric string.
        Called with two rows of df (e.g. df.iloc[:, i])
    minN : int
        Requires minimum number of non-NA rows to have a non-NA distance.
    symetric : bool
        Assume that the distance is symetric.

    Returns
    -------
    dmatDf : pd.DataFrame
        Distance matrix with index and columns matching input df.index"""
    if dfunc is None:
        if metric in ['spearman', 'pearson']:
            """Anti-correlations are also considered as high similarity and will cluster together"""
            dmat = 1. - df.T.corr(method=metric, min_periods=minN).values**2
            dmat[np.isnan(dmat)] = 1.
        elif metric in ['spearman-signed', 'pearson-signed']:
            """Anti-correlations are considered as dissimilar and will NOT cluster together"""
            dmat = ((1 - df.T.corr(method=metric.replace('-signed', ''), min_periods=minN).values) / 2.)
            dmat[np.isnan(dmat)] = 1.
        else:
            try:
                dvec = scipy.spatial.distance.pdist(df.values, metric=metric)
                dmat = scipy.spatial.distance.squareform(dvec, force='tomatrix', checks=True)
            except:
                raise NameError('metric name not recognized')
    else:
        nrows = df.shape[0]
        dmat = np.zeros((nrows, nrows))
        for i in range(nrows):
            for j in range(nrows):
                """Assume distance is symetric"""
                if symetric and i <= j:
                    tmpdf = df.iloc[:, [i, j]]
                    tmpdf = tmpdf.dropna()
                    if tmpdf.shape[0] >= minN:
                        d = dfunc(df.iloc[:, i], df.iloc[:, j])
                    else:
                        d = np.nan
                    dmat[i, j] = d
                    dmat[j, i] = d
                else:
                    tmpdf = df.iloc[:, [i, j]]
                    tmpdf = tmpdf.dropna()
                    if tmpdf.shape[0] >= minN:
                        d = dfunc(df.iloc[:, i], df.iloc[:, j])
                    else:
                        d = np.nan
                    dmat[i, j] = d

    return pd.DataFrame(dmat, columns=df.index, index=df.index)

def plotEmbedding(dmatDf,
                  xyDf=None,
                  labels=None,
                  method='kpca',
                  plotLabels=False,
                  plotDims=[0, 1],
                  plotElipse=False,
                  weights=None,
                  txtSize='large',
                  alpha=0.8,
                  sz=50,
                  mxSz=500,
                  markers=None,
                  plotLegend=True,
                  colors=None,
                  markerLabels=None,
                  continuousLabel=False):
    """Two-dimensional plot of embedded distance matrix, colored by labels"""
    
    if labels is None:
        labels = np.zeros(dmatDf.shape[0])

    assert dmatDf.shape[0] == dmatDf.shape[1]
    assert labels.shape[0] == dmatDf.shape[0]

    oh = objhist(labels)
    uLabels = sorted(np.unique(labels), key=oh.get, reverse=True)
    
    if xyDf is None:
        xyDf = embedDistanceMatrix(dmatDf, method=method, n_components=np.max(plotDims) + 1)
    
    clusteredScatter(xyDf,
                     labels=labels,
                     plotDims=plotDims,
                     plotElipse=plotElipse,
                     weights=weights,
                     alpha=alpha,
                     sz=sz,
                     mxSz=mxSz,
                     markerLabels=markerLabels,
                     markers=markers,
                     continuousLabel=continuousLabel,
                     colors=colors)
   
    if plotLabels:
        annotationParams = dict(xytext=(0, 5), textcoords='offset points', size=txtSize)
        for coli, col in enumerate(dmatDf.columns):
            if plotLabels:
                axh.annotate(col, xy=(xyDf.loc[col, plotDims[0]], xyDf.loc[col, plotDims[1]]), **annotationParams)

    if len(uLabels) > 1 and plotLegend:
        plt.legend(loc=0)
        # colorLegend(colors[:len(uLabels)], uLabels)
    if hasattr(xyDf, 'explained_variance_'):
        plt.xlabel('KPCA %1.0f (%1.0f%% variance explained)' % (plotDims[0]+1, 100*xyDf.explained_variance_[plotDims[0]]))
        plt.ylabel('KPCA %1.0f (%1.0f%% variance explained)' % (plotDims[1]+1, 100*xyDf.explained_variance_[plotDims[1]]))
    else:
        plt.xlabel('KPCA %1.0f' % (plotDims[0]+1))
        plt.ylabel('KPCA %1.0f' % (plotDims[1]+1))
    plt.show()
    return xyDf

def clusteredScatter(xyDf,
                     labels=None,
                     plotDims=[0, 1],
                     plotElipse=False,
                     weights=None,
                     alpha=0.8,
                     sz=50,
                     mxSz=500,
                     markerLabels=None,
                     markers=None,
                     colors=None,
                     continuousLabel=False):
    """Produce a scatter plot with axes, shaded by values in labels and with specified markers

    Parameters
    ----------
    xyDf : pd.DataFrame
        One column for each plot dimension specified
    labels : pd.Series
        Holds categorical or continuous metadata used for coloring
    plotDims : list len 2
        Specifies the columns in xyDf for plotting on X and Y axes
    plotElipse : bool
        Indicates whether a colored elipse should be plotted for the
        categorical labels. Ellipse is 2 standard deviations wide along each axis
    weights : pd.Series
        Relative weights that are mapped for sizing each symbol
    alpha : float
        Transparency of each point
    sz : float
        Size of each symbol or minimum size of a symbol if there are weights
    mxSz : float
        Maximum size of a symbol if there are weights
    markerLabels : pd.Series
        Holds categorical labels used for plotting different symbols
    markers : list
        List of marker symbols to use. Defaults to: "ov8sp*hDPX"
    colors : list
        List of colors to use for categorical labels or a colormap
        for a continuousLabel. Defaults to colors from Set1 or the YlOrRd colormap
    continuousLabel : bool
        Indicates whether labels are categorical or continuous"""
    if weights is None:
        sVec = sz * pd.Series(np.ones(xyDf.shape[0]), index=xyDf.index)
    else:
        sVec = weights * mxSz + sz
    
    if labels is None:
        labels = pd.Series(np.ones(xyDf.shape[0]), index=xyDf.index)
        useColors = False
    else:
        useColors = True

    if continuousLabel:
        if not colors is None:
            cmap = colors
        else:
            cmap = palettable.colorbrewer.sequential.YlOrRd_9.mpl_colormap
    else:
        cmap = None
        oh = objhist(labels)
        uLabels = sorted(np.unique(labels), key=oh.get, reverse=True)
        
        if colors is None:
            nColors = min(max(len(uLabels), 3), 9)
            colors = palettable.colorbrewer.get_map('Set1', 'Qualitative', nColors).mpl_colors
        elif isinstance(colors, pd.Series):
            colors = colors[uLabels].values

    if markerLabels is None:
        markerLabels = pd.Series(np.ones(xyDf.shape[0]), index=xyDf.index)
        useMarkers = False
    else:
        useMarkers = True

    oh = objhist(markerLabels)
    uMLabels = sorted(np.unique(markerLabels), key=oh.get, reverse=True)
    
    if markers is None:
        nMarkers = len(uMLabels)
        markers = ['o', 'v', '8', 's', 'p', '*', 'h', 'D', 'P', 'X'][:nMarkers]
    elif isinstance(markers, pd.Series):
        markers = markers[uMLabels].values

    figh = plt.gcf()
    plt.clf()
    axh = figh.add_axes([0.05, 0.05, 0.9, 0.9])
    axh.patch.set_facecolor('white')
    # axh.axis('off')
    figh.patch.set_facecolor('white')

    if continuousLabel:
        for mi, m in enumerate(uMLabels):
            ind = markerLabels == m
            if useMarkers:
                labS = '%s (N=%d)' % (m, ind.sum())
            else:
                labS = None
            
            plt.scatter(xyDf.loc[ind, plotDims[0]],
                        xyDf.loc[ind, plotDims[1]],
                        marker=markers[mi],
                        s=sVec.loc[ind],
                        alpha=alpha,
                        c=labels,
                        label=labS,
                        cmap=cmap)
    else:
        for vi, v in enumerate(uLabels):
            for mi, m in enumerate(uMLabels):
                ind = (labels == v) & (markerLabels == m)
                if useMarkers and useColors:
                    labS = '%s|%s (N=%d)' % (v, m, ind.sum())
                elif useColors:
                    labS = '%s (N=%d)' % (v, ind.sum())
                elif useMarkers:
                    labS = '%s (N=%d)' % (m, ind.sum())
                else:
                    labS = None
                
                plt.scatter(xyDf.loc[ind, plotDims[0]],
                            xyDf.loc[ind, plotDims[1]],
                            marker=markers[mi],
                            s=sVec.loc[ind],
                            alpha=alpha,
                            c=[colors[vi % len(colors)], ] * ind.sum(),
                            label=labS,
                            cmap=cmap)
            if ind.sum() > 2 and plotElipse:
                Xvar = xyDf[plotDims].loc[ind].values
                plot_point_cov(Xvar, ax=axh, color=colors[vi % len(colors)], alpha=0.2)
        axh.set_xticks(())
    axh.set_yticks(())
    plt.show()
    