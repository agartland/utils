import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse
from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kernel_regression import kernel2dist, dist2kernel

import seaborn as sns
sns.set(style='darkgrid', palette='muted', font_scale=1.5)

__all__ = ['corrSmatFunc',
            'corrTSmatFunc',
            'screeplot',
            'biplot',
            'kernel2dist',
            'dist2kernel']

def corrTSmatFunc(df, *args, **kwargs):
    """Correlation similarity function performed on the transpose
    of the input pd.DataFrame. Useful for clustering features
    and reducing instance space.

    Parameters
    ----------
    df : pd.DataFrame (n_instances, n_features)

    *args and **kwargs passed to corrSmatFunc()

    Returns
    -------
    smatDf : pd.DataFrame (n_features, n_features)"""

    return corrSmatFunc(df.T, *args, **kwargs)

def corrSmatFunc(df, metric='pearson-signed', simFunc=None, minN=None):
    """Compute a pairwise correlation matrix and return as a similarity matrix.

    Parameters
    ----------
    df : pd.DataFrame (n_instances, n_features)

    metric : str
        Method for correlation similarity: pearson or spearman, optionally "signed" (e.g. pearson-signed)
        A "signed" similarity means that anti-correlated instances will have low similarity.
    simFunc : function
        Optionally supply an arbitrary distance function.
        Function takes two instances and returns their distance.
    minN : int
        Minimum number of non-NA values in order for correlation to be non-NA.

    Returns
    -------
    smatDf : pd.DataFrame (n_instances, n_instances)"""

    if minN is None:
        minN = df.shape[0]

    if simFunc is None:
        if metric in ['spearman', 'pearson']:
            """Anti-correlations are also considered as high similarity and will cluster together"""
            smat = df.corr(method=metric, min_periods=minN).values**2
            smat[np.isnan(smat)] = 0
        elif metric in ['spearman-signed', 'pearson-signed']:
            """Anti-correlations are considered as dissimilar and will NOT cluster together"""
            smat = df.corr(method=metric.replace('-signed', ''), min_periods=minN).values
            smat = (smat**2 * np.sign(smat) + 1)/2
            smat[np.isnan(smat)] = 0
        else:
            raise NameError('metric name not recognized')
    else:
        ncols = df.shape[1]
        smat = np.zeros((ncols, ncols))
        for i in range(ncols):
            for j in range(ncols):
                """Assume distance is symetric"""
                if i <= j:
                    tmpdf = df.iloc[:, [i, j]]
                    tmpdf = tmpdf.dropna()
                    if tmpdf.shape[0] >= minN:
                        d = simFunc(df.iloc[:, i], df.iloc[:, j])
                    else:
                        d = np.nan
                    smat[i, j] = d
                    smat[j, i] = d
    return pd.DataFrame(smat, columns=df.columns, index=df.columns)

def _dimReduce(df, method='pca', n_components=2, labels=None, standardize=False, smatFunc=None, ldaShrinkage='auto'):
    if method == 'kpca':
        """By using KernelPCA for dimensionality reduction we don't need to impute missing values"""
        if smatFunc is None:
            smatFunc = corrTSmatFunc
        pca = KernelPCA(kernel='precomputed', n_components=n_components)
        smat = smatFunc(df).values
        xy = pca.fit_transform(smat)
        pca.components_ = pca.alphas_
        pca.explained_variance_ratio_ = pca.lambdas_ / pca.lambdas_.sum()
        return xy, pca
    elif method == 'pca':
        if standardize:
            normed = df.apply(lambda vec: (vec - vec.mean())/vec.std(), axis=0)
        else:
            normed = df.apply(lambda vec: vec - vec.mean(), axis=0)
        pca = PCA(n_components=n_components)
        xy = pca.fit_transform(normed)
        return xy, pca
    elif method == 'lda':
        if labels is None:
            raise ValueError('labels needed to perform LDA')
        if standardize:
            normed = df.apply(lambda vec: (vec - vec.mean())/vec.std(), axis=0)
        else:
            normed = df.apply(lambda vec: vec - vec.mean(), axis=0)
        
        if df.shape[1] > df.shape[0]:
            """Pre-PCA step"""
            ppca = PCA(n_components=int(df.shape[0]/1.5))
            normed = ppca.fit_transform(df)

        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage=ldaShrinkage, n_components=n_components)
        lda.fit(normed, labels.values)
        lda.explained_variance_ratio_ = np.abs(lda.explained_variance_ratio_) / np.abs(lda.explained_variance_ratio_).sum()
        xy = lda.transform(normed)
        return xy, lda
    elif method == 'pls':
        if labels is None:
            raise ValueError('labels needed to perform PLS')
        if standardize:
            normed = df.apply(lambda vec: (vec - vec.mean())/vec.std(), axis=0)
        else:
            normed = df.apply(lambda vec: vec - vec.mean(), axis=0)
        
        pls = PLSRegression(n_components=n_components)
        pls.fit(normed, labels)
        
        pls.explained_variance_ratio_ = np.zeros(n_components)
        xy = pls.x_scores_
        return xy, pls

def screeplot(df, method='pca', n_components=10, standardize=False, smatFunc=None):
    """Create stacked bar plot of compents and the fraction contributed by each feature"""
    n_components = int(np.min([n_components, df.columns.shape[0]]))
    xy, pca = _dimReduce(df, method, n_components, standardize, smatFunc)
    
    figh = plt.gcf()
    figh.clf()
    axh1 = figh.add_subplot(2, 1, 1)
    axh1.bar(left=list(range(n_components)),
             height=pca.explained_variance_ratio_[:n_components],
             align='center')
    plt.ylabel('Fraction of\nvariance explained')
    plt.xticks(())

    axh2 = figh.add_subplot(2, 1, 2)
    for compi in range(n_components):
        bottom = 0
        for dimi, col in zip(list(range(df.shape[1])), itertools.cycle(mpl.cm.Set3.colors)):
            height = pca.components_[compi, dimi]**2 / (pca.components_[compi,:]**2).sum()
            if height > 0.01:
                axh2.bar(left=compi, bottom=bottom, height=height, align='center', color=col)
            if height > 0.1:
                note = df.columns[dimi].replace(' ', '\n')
                note += '(+)' if pca.components_[compi, dimi] >= 0 else '(-)'
                axh2.annotate(note, xy=(compi, bottom+height/2), ha='center', va='center', size='small')
            if height > 0.01:
                bottom += height
    plt.xticks(list(range(n_components)), ['PC%d' % (i+1) for i in range(n_components)], rotation=90)
    plt.ylim([0, 1])
    plt.ylabel('Fraction of\ncomponent variance')

def biplot(df, labels=None, method='pca', plotLabels=True, plotDims=[0, 1],
           plotVars='all', label_order=None, standardize=False, smatFunc=None, varThresh=0.1,
           ldaShrinkage='auto', dropna=False, plotElipse=True, colors=None):
    """Perform dimensionality reduction on columns of df using PCA, KPCA or LDA,
    then produce a biplot in two-dimensions.
    
    Parameters
    ----------
    df : pd.DataFrame
    labels : pd.Series
        Class labels used for grouping/coloring and LDA.
    method : str
        Method for dimensionality reduction: PCA, KPCA, LDA
    plotLabels : bool
        If True, show instance labels.
    plotDims : list of len 2
        Components of the transformed space to plot as [x, y]
    plotVars : list or 'all'
        List of columns in df to project as vectors.
    standardize : bool
        If True, scale to unit variance.
    smatFunc : function
        Function to apply to df to get a pairwise similarity
        matrix to be used in KernelPCA. Return should have
        shape (df.shape[0], df.shape[0])
    varThresh : float
        Threshold for which variables are plotted as vectors.
        If a variable explains a higher fraction of variance in any dimension
        than the threshold then it is plotted.
    ldaShrinkage : str or None
        Passed to sklearn.discriminant_analysis.LinearDiscriminantAnalysis
    plotElipse : bool
        Draw elipse representing 80% CI, default True

    Returns
    -------
    None"""

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
    if label_order is None:
        uLabels = np.unique(labels).tolist()
    else:
        uLabels = label_order
    n_components = max(plotDims) + 1
    xy, pca = _dimReduce(df, method=method, n_components=n_components, standardize=standardize, smatFunc=smatFunc, labels=labels, ldaShrinkage=ldaShrinkage)

    if colors is None:
        colors = mpl.cm.Set3.colors
    axh = plt.gca()
    axh.axis('on')
    # plt.gcf().set_facecolor('white')

    annotationParams = dict(xytext=(0, 5), textcoords='offset points', size='medium')
    alpha = 0.8
    for i, obs in enumerate(df.index):
        if plotLabels:
            axh.annotate(obs, xy=(xy[i, plotDims[0]], xy[i, plotDims[1]]), **annotationParams)
    colorArray = np.zeros((1, 3))
    for labi, lab in enumerate(uLabels):
        colorArray[0, :] = colors[labi]
        ind = np.where(labels==lab)[0]
        axh.scatter(xy[ind, plotDims[0]], xy[ind, plotDims[1]], marker='o', s=50, alpha=alpha, c=colorArray, label=str(lab) + '(N = %d)' % len(ind))
        #axh.scatter(xy[ind, plotDims[0]].mean(axis=0), xy[ind, plotDims[1]].mean(axis=0), marker='o', s=300, alpha=alpha/1.5, c=col)
        Xvar = xy[ind,:][:, plotDims]
        if len(ind) > 2 and plotElipse:
            plot_point_cov(Xvar, ax=axh, color=colors[labi], alpha=0.2)
    arrowParams = dict(arrowstyle='<-',
                        connectionstyle='Arc3',
                        color='black',
                        lw=1)
    annotationParams = dict(xy=(0, 0),
                            textcoords='data',
                            color='black',
                            arrowprops=arrowParams,
                            ha='center',
                            va='center',
                            size='small')
    mxx = np.max(np.abs(xy[:, plotDims[0]]))
    mxy = np.max(np.abs(xy[:, plotDims[1]]))
    scalar = min(mxx, mxy) * 0.7
    
    if method in ['lda', 'pca']:
        """Project a unit vector for each feature, into the new space"""    
        arrowxy = pca.transform(np.diag(np.ones(df.shape[1])))
        mxarr = np.max(np.abs(arrowxy))
        """By using the squared transform the magnitude of the vector along each component
        reflects the fraction of variance explained by that feature along the component (e.g. PCA1)"""
        varfracxy = (arrowxy**2) * np.sign(arrowxy)
        for vi, v in enumerate(df.columns):
            arrowx, arrowy = arrowxy[vi,:] * scalar/mxarr
            #arrowx = varfracxy[vi,0] * mxx
            #arrowy = varfracxy[vi,1] * mxy
            if v in plotVars and np.any(np.abs(varfracxy[vi,:]) > varThresh):
                axh.annotate(v, xytext=(arrowx, arrowy), **annotationParams)

    plt.xlabel('%s%d (%1.1f%% var. exp.)' % (method.upper(), plotDims[0] + 1, pca.explained_variance_ratio_[plotDims[0]] * 100))
    plt.ylabel('%s%d (%1.1f%% var. exp.)' % (method.upper(), plotDims[1] + 1, pca.explained_variance_ratio_[plotDims[1]] * 100))

    #plt.xticks([0])
    #plt.yticks([0])
    padding = 0.05
    xl = plt.xlim()
    dx = xl[1] - xl[0]
    plt.xlim((xl[0] - dx*padding, xl[1] + dx*padding))
    yl = plt.ylim()
    dy = yl[1] - yl[0]
    plt.ylim((yl[0] - dy*padding, yl[1] + dy*padding))
    if len(uLabels) > 1:
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

def plot_point_cov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).

    Credit: https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py

    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plot_cov_ellipse(cov, pos, nstd, ax, **kwargs)

def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def _test_iris():
    """Import the iris dataset from sklearn, and return as a result"""
    from sklearn import datasets

    iris = datasets.load_iris()
    index = np.arange(150)+1
    irisDf = pd.DataFrame(iris['data'], columns=iris['feature_names'], index=index)
    labels = pd.Series(iris['target_names'][iris['target']], index=index)

    xyPCA, pcaObj = _dimReduce(irisDf, method='pca')
    xyLDA, ldaObj = _dimReduce(irisDf, labels=labels, method='lda')
    xyKPCA, kpcaObj = _dimReduce(irisDf, labels=labels, method='kpca')
    return irisDf, labels


