import numpy as np
import matplotlib.pyplot as plt

__all__ = ['computeGapStat',
            'plotGapStat']

def computeGapStat(data, pdistFunc, clusterFunc, maxK, bootstraps = 10):
    """Compute the gap statistic, varying the number of clusters (K)
    to determine the optimal number of clusters.

    The optimal number of clusters is the smallest K for which:

    $Gap(k) - (Gap(k+1) - \sigma_{k+1}) > 0$

    The statistic is described here:
    Tibshirani, R., Walther, G., Hastie, T., 2001. Estimating the number of clusters in a data set via the gap statistic.
        J. R. Stat. Soc. Ser. B. (Statistical Methodol.) 63, 411-423

    and a good example of python code is here:
    https://datasciencelab.wordpress.com/2013/12/27/finding-the-k-in-k-means-clustering/


    Parameters
    ----------
    data : np.array or pd.DataFrame
        Matrix of data with observations on the rows and variables along the columns.
        Data will be row-bootstrapped for each column to create null datasets.
    pdistFunc : function
        Function for computing pairwise distance matrix from data.
        Use partial to prespecify metric arguments if neccessary.
    clusterFunc : function
        Function that takes the distance matrix and a prespecified number of clusters
        and returns cluster labels.
        Use partial to prespecify method arguments if neccessary.
    maxK : int
        Maximum number of clusters to consider.
    bootstraps : int
        Number of bootstrap samples to compute for each K.

    Returns
    -------
    lsICD : np.array of shape (maxK,)
        Log of the sum of the intra cluster distances (LSICD), for each K
    mBSICD : np.array of shape (maxK,)
        Average LSICD over random bootstraps, for each K
    errBSICD : np.array of shape (maxK,)
        Bootstrap error LSICD over random bootstraps, for each K
    gap : np.array of shape (maxK,)
        Gap statistic for each K
    """
    dmat = pdistFunc(data)

    lsICD = np.zeros(maxK)
    mBSICD = np.zeros(maxK)
    stdBSICD = np.zeros(maxK)

    for K in (np.arange(maxK) + 1):
        labels = clusterFunc(dmat, K)
        lsICD[K-1] = np.log(_intra_cluster_distances(dmat, labels))
        reps = np.zeros(bootstraps)
        for i in range(bootstraps):
            tmpDmat = pdistFunc(_bootstrap_each_column(data))
            labels = clusterFunc(tmpDmat, K)
            reps[i] = np.log(_intra_cluster_distances(tmpDmat, labels))
        mBSICD[K-1] = reps.mean()
        stdBSICD[K-1] = reps.std()

    gap = mBSICD - lsICD
    errBSICD = sqrt(1 + 1./bootstraps) * stdBSICD

    return lsICD, mBSICD, errBSICD, gap

def plotGapStat(lsICD, mBSICD, errBSICD, gap):
    """Descriptive plot of the Gap statistic.
    Parameters are simply the output from computeGapStat."""

    maxK = len(gap)
    plt.clf()
    plt.subplot(2,2,1)
    plt.plot(np.arange(maxK) + 1, np.exp(lsICD) / np.exp(lsICD[0]), 'o-', color = 'black', label = 'Observed data')
    plt.xticks(np.arange(maxK) + 1)
    plt.ylabel('Summed intra-cluster distances\nas a fraction of total pairwise distance')
    plt.xlabel('Number of clusters (K)')
    plt.ylim((0,1))

    plt.subplot(2,2,2)
    plt.plot(np.arange(maxK) + 1, lsICD, 'o-', color = 'black', label = 'Observed data')
    plt.plot(np.arange(maxK) + 1, mBSICD, 'o-', color = 'red', label = 'Null data')
    plt.xticks(np.arange(maxK) + 1)
    plt.ylabel('$log(W_k)$')
    plt.xlabel('Number of clusters (K)')
    plt.legend(loc = 0)

    plt.subplot(2,2,3)
    plt.plot(np.arange(maxK) + 1, gap, 'o-')
    plt.xticks(np.arange(maxK) + 1)
    plt.ylabel('Gap statistic')
    plt.xlabel('Number of clusters (K)')

    plt.subplot(2,2,4)
    q = gap[:-1] - (gap[1:] - errBSICD[1:])
    plt.bar(left = np.arange(maxK-1) + 1, height = q, color = 'blue', align = 'center')
    plt.xticks(np.arange(maxK) + 1)
    plt.ylabel('$Gap(k) - (Gap(k+1) - \sigma_{k+1})$')
    plt.xlabel('Number of clusters (K)')
    plt.tight_layout()
    
def _intra_cluster_distances(dmat, labels):
    """Sum of the intra-cluster distances (Wk)"""
    K = len(unique(labels))
    tot = 0
    for k in unique(labels):
        ind = labels == k
        nk = ind.sum()
        tot += (dmat[ind,:][:,ind].flatten()**2).sum() / (2 * nk)
    return tot

def _bootstrap_each_column(d):
    """Returns a copy of data with row-bootstraped values
    substituted for each column independently."""
    out = d.copy()

    if isinstance(out, pd.DataFrame):
        out = out.apply(lambda col: col[np.floor(np.random.rand(len(col)) * len(col)).astype(int)], axis = 0, raw = True)
    else:
        for ci in range(d.shape[1]):
            rind = np.floor(np.random.rand(d.shape[0]) * d.shape[0]).astype(int)
            out[:,ci] = out[rind, ci]
   
    return out