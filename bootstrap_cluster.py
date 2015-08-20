import numpy as np
import pandas as pd
import itertools
#import matplotlib.pyplot as plt

__all__ = ['bootstrapFeatures', 'bootstrapObservations']

def bootstrapFeatures(dmat, clusterFunc, bootstraps = 100, rseed = 110820):
    """Determine the reliability of clusters using a bootstrap.

    This algorithm is from this article as it was applied to gene chips:
    "Bagging to improve the accuracy of a clustering procedure"
    http://bioinformatics.oxfordjournals.org/content/19/9/1090.full.pdf+html

    Parameters
    ----------
    dmat : np.array or pd.DataFrame
        Pairwise distance matrix.
    clusterFunc : function
        Function that takes the distance matrix and returns cluster labels.
        Use partial to prespecify method arguments if neccessary.
    bootstraps : int
        Number of bootstrap samples to use.
    rseed : int
        Sets the random state of numpy for reproducible clustering results.

    Returns
    -------
    pwrel : np.array or pd.DataFrame
        Distance matrix based on the fraction of times each variable clusters together.
        (actually 1 - fraction)
    clusters : np.ndarray or pd.Series
        Array of labels after applying the clustering function
        to the reliability distance matrix pwrel
    """
    np.random.seed(rseed)

    assert dmat.shape[0] == dmat.shape[1]

    N = dmat.shape[0]

    pwrel = np.zeros((N,N))
    """Keep track of the number of times that two variables are sampled together"""
    tot = np.zeros((N,N))
  
    for i in range(bootstraps):
        """Use tmp arrays because there can only be 1 count per bootstrap maximum"""
        tmpTot = np.zeros((N,N))
        tmpRel = np.zeros((N,N))

        rind = np.floor(np.random.rand(N) * N).astype(int)
        if isinstance(dmat, pd.DataFrame):
            rdmat = dmat.iloc[:,rind].iloc[rind,:]
        else:
            rdmat = dmat[:,rind][rind,:]
        labels = clusterFunc(rdmat)

        for rj,rk in itertools.product(range(N),range(N)):
            """Go through all pairs of variables (lower half of the pwdist matrix)"""
            
            """Keep track of indices into original dmat (j,k) and those into the
            resampled dmat (rj, rk)"""
            j,k = rind[rj], rind[rk]
            if j<k:
                if rj != rk and tmpTot[j,k] == 0:
                    tmpTot[j,k] = 1 
                    if labels[rj] == labels[rk]:
                        tmpRel[j,k] = 1
        pwrel += tmpRel
        tot += tmpTot
    for j,k in itertools.product(range(N),range(N)):
        if j<k:
            pwrel[j,k] = pwrel[j,k] / tot[j,k]
            pwrel[k,j] = pwrel[j,k]
        elif j == k:
            pwrel[j,k] = 1
    pwrel = 1 - pwrel

    if isinstance(dmat, pd.DataFrame):
        pwrel = pd.DataFrame(pwrel, index = dmat.index, columns = dmat.columns)
    clusters = clusterFunc(pwrel)
    if isinstance(dmat, pd.DataFrame):
        clusters = pd.Series(clusters, index = dmat.index)
    return pwrel, clusters


def bootstrapObservations(df, dmatFunc, clusterFunc, bootstraps = 100, rseed = 110820):
    """Determine the reliability of clusters using a bootstrap.

    This algorithm bootstraps the observations and repeats the clustering.

    Parameters
    ----------
    df : np.array or pd.DataFrame
        Features on the columns and observations on the rows.
    dmatFunc : function
        Function that takes the df and produces a square distance matrix.
    clusterFunc : function
        Function that takes the distance matrix and returns cluster labels.
        Use partial to prespecify method arguments if neccessary.
    bootstraps : int
        Number of bootstrap samples to use.
    rseed : int
        Sets the random state of numpy for reproducible clustering results.

    Returns
    -------
    pwrel : np.array or pd.DataFrame
        Distance matrix based on the fraction of times each variable clusters together.
        (actually 1 - fraction)
    clusters : np.ndarray or pd.Series
        Array of labels after applying the clustering function
        to the reliability distance matrix pwrel"""
    np.random.seed(rseed)

    dmat = dmatFunc(df)

    Nobs = df.shape[0]
    N = df.shape[1]

    assert N == dmat.shape[0]
    assert N == dmat.shape[1]

    pwrel = np.zeros((N,N))
    """Keep track of the number of times that two variables are sampled together"""
    tot = np.zeros((N,N))
  
    for i in range(bootstraps):
        """Resample observations (rows) with replacement"""
        rind = np.floor(np.random.rand(Nobs) * Nobs).astype(int)
        if isinstance(df, pd.DataFrame):
            rdmat = dmatFunc(df.iloc[rind,:])
        else:
            rdmat = dmatFunc(df[rind,:])

        labels = clusterFunc(rdmat)

        for j,k in itertools.product(range(N),range(N)):
            """Go through all pairs of variables (lower half of the pwdist matrix)"""
            if j<k:
                if labels[j] == labels[k]:
                    pwrel[j,k] += 1.

    for j,k in itertools.product(range(N),range(N)):
        if j<k:
            pwrel[j,k] = pwrel[j,k] / bootstraps
            pwrel[k,j] = pwrel[j,k]
        elif j == k:
            pwrel[j,k] = 1
    pwrel = 1 - pwrel

    if isinstance(dmat, pd.DataFrame):
        pwrel = pd.DataFrame(pwrel, index = dmat.index, columns = dmat.columns)
    clusters = clusterFunc(pwrel)
    if isinstance(dmat, pd.DataFrame):
        clusters = pd.Series(clusters, index = dmat.index)
    return pwrel, clusters
