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
    print "seqdistance: Could not load tsne; falling back on pytsne"

"""Embedding of pairwise-distance matrices."""

__all__ = [ 'embedDistanceMatrix',
            'computePWDist',
            'plotEmbedding']

def embedDistanceMatrix(dmatDf, method='kpca', n_components=2):
    """Two-dimensional embedding of sequence distances in dmatDf,
    returning Nx2 x,y-coords: tsne, isomap, pca, mds, kpca, sklearn-tsne"""
    if type(dmatDf) is pd.DataFrame:
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
        pcaObj = PCA(n_components=n_components)
        xy = pcaObj.fit_transform(dmat)
    elif method == 'kpca':
        pcaObj = KernelPCA(n_components=n_components, kernel='precomputed')
        try:
            gram = dist2kernel(dmat)
        except:
            print 'Could not convert dmat to kernel for KernelPCA; using 1 - dmat/dmat.max() instead'
            gram = 1 - dmat / dmat.max()
        xy = pcaObj.fit_transform(gram)
    elif method == 'lle':
        lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=n_components, method='standard')
        xy = lle.fit_transform(dist)
    elif method == 'sklearn-tsne':
        tsneObj = TSNE(n_components=n_components, metric='precomputed', random_state=0)
        xy = tsneObj.fit_transform(dmat)
    else:
        print 'Method unknown: %s' % method
        return

    assert xy.shape[0] == dmatDf.shape[0]
    xyDf = pd.DataFrame(xy[:,:n_components], index=dmatDf.index, columns=np.arange(n_components))
    return xyDf

def computePWDist(df, metric='pearson-signed', dfunc=None, minN=10):
    """Compute pairwise distance matrix using correlation or arbitrary function."""
    if dfunc is None:
        if metric in ['spearman', 'pearson']:
            """Anti-correlations are also considered as high similarity and will cluster together"""
            dmat = 1 - df.T.corr(method=metric, min_periods=minN).values**2
            dmat[np.isnan(dmat)] = 1
        elif metric in ['spearman-signed', 'pearson-signed']:
            """Anti-correlations are considered as dissimilar and will NOT cluster together"""
            dmat = ((1 - df.T.corr(method=metric.replace('-signed',''), min_periods=minN).values) / 2)
            dmat[np.isnan(dmat)] = 1
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
                if i <= j:
                    tmpdf = df.iloc[:,[i,j]]
                    tmpdf = tmpdf.dropna()
                    if tmpdf.shape[0] >= minN:
                        d = dfunc(df.iloc[:,i], df.iloc[:,j])
                    else:
                        d = np.nan
                    dmat[i,j] = d
                    dmat[j,i] = d
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
                  marker='o'):
    """Two-dimensional plot of embedded distance matrix, colored by labels"""
    
    if labels is None:
        labels = np.zeros(dmatDf.shape[0])

    if weights is None:
        sVec = sz
    else:
        sVec = weights * 200 + 10
    
    assert dmatDf.shape[0] == dmatDf.shape[1]
    assert labels.shape[0] == dmatDf.shape[0]

    oh = objhist(labels)
    uLabels = sorted(np.unique(labels), key=oh.get, reverse=True)
    
    if xyDf is None:
        xyDf = embedDistanceMatrix(dmatDf, method=method, n_components=np.max(plotDims) + 1)

    nColors = min(max(len(uLabels), 3), 9)
    colors = palettable.colorbrewer.get_map('Set1', 'Qualitative', nColors).mpl_colors

    figh = plt.gcf()
    plt.clf()
    axh = figh.add_axes([0.02, 0.02, 0.96, 0.96])
    axh.axis('off')
    figh.set_facecolor('white')

    for vi,v in enumerate(uLabels):
        ind = (labels == v)
        plt.scatter(xyDf.loc[ind, plotDims[0]],
                    xyDf.loc[ind, plotDims[1]],
                    marker=marker,
                    s=sVec,
                    alpha=alpha,
                    c=colors[vi],
                    label='%s (N=%d)' % (v, ind.sum()))
        if ind.sum() > 2 and plotElipse:
            Xvar = xyDf[plotDims].loc[ind].values
            plot_point_cov(Xvar, ax=axh, color=colors[vi], alpha=0.2)
    
    if plotLabels:
        annotationParams = dict(xytext=(0,5), textcoords='offset points', size=txtSize)
        for coli,col in enumerate(dmatDf.columns):
            if plotLabels:
                axh.annotate(col, xy=(xyDf[col, plotDims[0]], xyDf[col, plotDims[1]]), **annotationParams)

    if len(uLabels) > 1:
        plt.legend(loc=0)
        #colorLegend(colors[:len(uLabels)], uLabels)
    plt.show()
    return xyDf
