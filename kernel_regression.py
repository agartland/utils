import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = ['kreg_perm',
            'argrank',
            'dist2kernel',
            'kernel2dist',
            'computePWDist']
"""Tested one kernel, logistic regression.
Multiple kernels and continuous outcome should be tested."""

def kreg_perm(y, Ks, X=None, binary=True, nperms=9999, seed=110820, returnPerms=False):
    """Kernel regression of Y adjusting for covariates in X
    Implemented the permutation method from:
    
    Zhao N, Chen J, Carroll IM, Ringel-Kulka T, Epstein MP, Zhou H, et al.
        Testing in microbiome-profiling studies with MiRKAT, the microbiome
        regression-based kernel association test. Am J Hum Genet. The
        American Society of Human Genetics; 2015;96(5):797-807.

    Parameters
    ----------
    y : pd.Series or np.ndarray shape (n,1)
        Endogenous/outcome variable.
        Can be continuous or binary (use binary=True)
    Ks : pd.DataFrame or np.ndaray shape (n,n) OR list
        Kernel or list of kernels
        (square, symetric and positive semi-definite)
    X : pd.DataFrame OR np.ndarray shape (n,i)
        Matrix of covariates for adjustment.
    binary : bool
        Use True for logistic regression.
    nperms : int
        Number of permutations for the test.
    returnPerms : bool
        If True and Ks is a single kernel,
        return the observed and permuted Q-statistics.

    Returns
    -------
    pvalue : float
        Permutation P-value or global/omnibus P-value,
        if there are multiple kernels
    pIndivid : ndarray shape (k)
        Optionally returns vector of p-values, one
        for each kernel provided."""

    def computeQ(K, resid, s2):
        return np.linalg.multi_dot((resid/s2, K, resid))

    n = len(y)
    if binary:
        family = sm.families.Binomial()
    else:
        family = sm.families.Gaussian()

    if X is None:
        X = np.ones(y.shape, dtype=float)
        p = 1
    else:
        if len(X.shape) == 1:
            p = 2
        else:
            p = X.shape[1] + 1

    model = sm.GLM(endog=y.astype(float), exog=sm.add_constant(X.astype(float)), family=family)
    result = model.fit()
    resid = result.resid_response

    """Squared standard error of the parameters (i.e. Beta_se)"""
    s2 = (1. / (n-p)) * np.sum(resid**2.)

    np.random.seed(seed)
    if type(Ks) is list:
        """If there are multiple kernels then use a min-P omnibus test"""
        Ks = [K.values if type(K) is pd.DataFrame else K for K in Ks]

        obsQ = np.array([computeQ(K, resid, s2) for K in Ks])[None,:]
        randQ = np.nan * np.zeros((nperms, len(Ks)))
        for permi in range(nperms):
            rind = np.random.permutation(n)
            randQ[permi, :] = [computeQ(K[:, rind][rind, :], resid, s2) for K in Ks]
        
        Qall = np.concatenate((obsQ, randQ), axis=0)
        pall = np.zeros(Qall.shape)
        for qi in range(len(Ks)):
            pall[:, qi] = 1 - argrank(Qall[:, qi]) / (nperms + 1.)
        pIndivid = pall[0,:]
        minPall= pall.min(axis=1)
        pGlobal = argrank(minPall)[0] / (nperms + 1.)
        return pGlobal, pIndivid
    else:
        if type(Ks) is pd.DataFrame:
            Ks = Ks.values

        obsQ = computeQ(Ks, resid, s2)
        randQ = np.nan * np.zeros(nperms)
        for permi in range(nperms):
            rind = np.random.permutation(n)
            randQ[permi] = computeQ(Ks[:, rind][rind, :], resid, s2)
        pvalue = (np.sum(randQ > obsQ) + 1.) / (nperms + 1.)
        if returnPerms:
            return pvalue, obsQ, randQ
        else:
            return pvalue

def computeKregStat(y, Ks, X=None, binary=True):
    """Compute the statistic that is subjected to permutation testing
    in kernel regression of Y adjusting for covariates in X.

    Parameters
    ----------
    y : pd.Series or np.ndarray shape (n,1)
        Endogenous/outcome variable.
        Can be continuous or binary (use binary=True)
    K : pd.DataFrame or np.ndaray shape (n,n)
        Kernel (square, symetric and positive semi-definite)
    X : pd.DataFrame OR np.ndarray shape (n,i)
        Matrix of covariates for adjustment.
    binary : bool
        Use True for logistic regression.

    Returns
    -------
    Q : float
        Regression coefficient for K
    pIndivid : ndarray shape (k)
        Optionally returns vector of p-values, one
        for each kernel provided."""

    n = len(y)
    if binary:
        family = sm.families.Binomial()
    else:
        family = sm.families.Gaussian()

    if X is None:
        X = np.ones(y.shape, dtype=float)

    model = sm.GLM(endog=y.astype(float), exog=sm.add_constant(X.astype(float)), family=family)
    result = model.fit()
    resid = result.resid_response

    """Squared standard error of the parameters (i.e. Beta_se)"""
    s2 = float(result.bse**2)

    if type(K) is pd.DataFrame:
        K = K.values

    Q = np.linalg.multi_dot((resid/s2, K, resid))
    return Q


def argrank(vec):
    """Return the ascending rank (0-based) of the elements in vec
    Parameters
    ----------
    vec : np.ndarray shape (n,) or (n,1)

    Returns
    -------
    ranks : np.ndarray shape (n,)
        Ranks of each element in vec (dtype int)"""
    sorti = np.argsort(vec)
    ranks = np.empty(len(vec), int)
    try:
        ranks[sorti] = np.arange(len(vec))
    except IndexError:
        ranks[sorti.values] = np.arange(len(vec))
    return ranks

def dist2kernel(dmat):
    """Convert a distance matrix into a similarity kernel
    for KernelPCA or kernel regression methods.

    Implementation of D2K in MiRKAT, Zhao et al., 2015

    Parameters
    ----------
    dmat : ndarray shape (n,n)
        Pairwise-distance matrix.

    Returns
    -------
    kernel : ndarray shape (n,n)"""

    n = dmat.shape[0]
    I = np.identity(n)
    """m = I - dot(1,1')/n"""
    m = I - np.ones((n,n))/float(n)
    kern = -0.5 * np.linalg.multi_dot((m, dmat**2, m))

    if type(dmat) is pd.DataFrame:
        return pd.DataFrame(kern, index=dmat.index, columns=dmat.columns)
    else:
        return kern

def kernel2dist(kern):
    """Recover a distance matrix from the kernel.

    Implementation of K2D in MiRKAT, Zhao et al., 2015

    d_ij^2 = K_ii + K_jj - 2K_ij

    Parameters
    ----------
    kernel : ndarray shape (n,n)

    Returns
    -------
    dmat : ndarray shape (n,n)"""
    dmat = np.sqrt(np.diag(kern)[None,:] + np.diag(kern)[:, None] - 2 * kern)
    return dmat

def posSDCorrection(kernel):
    """Positive semi-definite correction.
    Eigenvalue decomposition is used to ensure that the kernel
    matrix is positive semi-definite."""
    u, s, v = np.linalg.svd(kernel)
    return np.linalg.multi_dot((u, np.diag(np.abs(s)), v))

def computePWDist(series1, series2, dfunc, symetric=True):
    """Compute and assemble a pairwise distance matrix
    given two pd.Series and a function to compare them.

    Parameters
    ----------
    series1, series2 : pd.Series
        Items for comparison. Will compute all pairs of distances
        between items in series1 and series2.
    dfunc : function
        Function takes 2 parameters and returns a float
    symetric : bool
        If True, only compute half the distance matrix and duplicate.

    Returns
    -------
    dmat : pd.DataFrame
        Distance matrix with series1 along rows and
        series 2 along the columns."""
    nrows = series1.shape[0]
    ncols = series2.shape[0]
    dmat = np.zeros((nrows, ncols))
    for i in range(nrows):
        for j in range(ncols):
            if symetric:
                if i <= j:
                    d = dfunc(series1.iloc[i], series2.iloc[j])
                    dmat[i,j] = d
                    dmat[j,i] = d
            else:
                dmat[i,j] = dfunc(series1.iloc[i], series2.iloc[j])
    return pd.DataFrame(dmat, columns=series2.index, index=series1.index)