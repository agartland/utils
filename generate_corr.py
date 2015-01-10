from numpy.linalg import cholesky, inv, eigh
from numpy.random import rand
import numpy as np
from scipy import special
from scipy import stats
from copy import deepcopy

__all__ = ['generateNormalCorr',
           'induceRankCorr',
           'generateBinVars']

def generateNormalCorr(N,k,C,method = 'cholesky'):
    """Induces correlation specified by covariance matrix Cstar

    From SciPy cookbook:
    http://wiki.scipy.org/Cookbook/CorrelatedRandomSamples

    Parameters
    ----------
    N : int
        Number of samples.
    k : int
        Number of variables.
    C : ndarray [k x k]
        Positive, symetric covariance matrix.

    Returns
    -------
    R : ndarray [N x k]
        Array of random correlated samples."""

    if method == 'cholesky':
        U = cholesky(C)
    else:
        evals, evecs = np.eigh(C)
        U = np.dot(evecs, np.diag(np.sqrt(evals)))

    R = np.dot(rand(N,k),U)
    return R

def induceRankCorr(R,Cstar):
    """Induces rank correlation Cstar onto a sample R [N x k].
    Note that it is easy to specify correlations that are not possible to generate.
    Results generated with a given Cstar should be checked.

    Iman, R. L., and W. J. Conover. 1982. A Distribution-free Approach to Inducing Rank
    Correlation Among Input Variables. Communications in Statistics: Simulation and
    Computations 11:311-334.
    
    Parameters
    ----------
    R : ndarray [N x k]
        Matrix of random samples (with no pre-existing correlation)
    Cstar : ndarray [k x k]
        Desired positive, symetric correlation matrix with ones along the diagonal.
    
    Returns
    -------
    corrR : ndarray [N x k]
        A correlated matrix of samples."""

    """Define inverse complimentary error function (erfcinv in matlab)
    x is on interval [0,2]
    its also defined in scipy.special"""
    #erfcinv = lambda x: -stats.norm.ppf(x/2)/sqrt(2)

    C = Cstar
    N,k = R.shape
    """Calculate the sample correlation matrix T"""
    T = np.corrcoef(R.T)

    """Calculate lower triangular cholesky
        decomposition of Cstar (i.e. P*P' = C)"""
    P = cholesky(C).T

    """Calculate lower triangular cholesky decomposition of T, i.e. Q*Q' = T"""
    Q = cholesky(T).T

    """S*T*S' = C"""
    S = P.dot(inv(Q))

    """Replace values in samples with corresponding
    rank-indices and convert to van der Waerden scores"""

    RvdW = -np.sqrt(2) * special.erfcinv(2*((_columnRanks(R)+1)/(N+1)))

    """Matrix RBstar has a correlation matrix exactly equal to C"""
    RBstar = RvdW.dot(S.T)
    
    """Match up the rank pairing in R according to RBstar"""
    ranks = _columnRanks(RBstar)
    sortedR = np.sort(R,axis=0)
    corrR = np.zeros(R.shape)
    for j in np.arange(k):
        corrR[:,j] = sortedR[ranks[:,j],j]

    return corrR

def generateBinVars(p,N):
    """Generate random binary variables with specified correlation

    "A simple method for generating correlated binary variates."
    Park C, Park T, Shin D. 1996. Am. Stat. 50:306:310.

    Parameters
    ----------
    p : ndarray [k x k]
        Positive, symetric correlation matrix with p_ii on the diagonal and p_ij off the diagonal
    N : int
        Number of samples to generate.
    
    Returns
    -------
    Z : ndarray [N x k]
        Correlated random binary samples.
    
    Example
    -------

    p = array([[0.9,0.1,0.5],
               [0.1,0.8,0.5],
               [0.5,0.5,0.7]])
    
    Z = generateBinVars(p,1e3)
    """
    def alphaFunc(p):
        q = 1-p
        d = np.diag(q)/np.diag(p)
        imat = np.tile(d.reshape((1,p.shape[0])),(p.shape[0],1))
        jmat = np.tile(d.reshape((p.shape[0],1)),(1,p.shape[0]))
       
        ijmat = np.log(1 + p*np.sqrt(imat*jmat))
        dind = np.diag_indices(p.shape[0])
        ijmat[dind] = -np.log(diag(p))
        return ijmat

    a = alphaFunc(p)

    ana = deepcopy(a)
    tind = np.triu_indices(a.shape[0])
    ana[np.tril_indices(a.shape[0])] = nan
    ana[np.diag_indices(a.shape[0])] = a[np.diag_indices(a.shape[0])]

    betaL = []
    rsL = []
    slL = []
    while np.any(ana[tind]>0):
        ana[ana==0] = nan
        #print ana
      
        rs = list(np.unravel_index(np.nanargmin(ana),a.shape))
        mn = np.nanmin(ana)
        if ana[rs[0],rs[0]] == 0 or ana[rs[1],rs[1]] == 0:
            break
        betaL.append(mn)
        rsL.append(rs)
        #print rs

        rs = set(rs)
        for i in range(a.shape[0]):
            if np.all(ana[list(rs),i]>0):
                rs.add(i)
        slL.append(rs)
        #print rs

        for i in rs:
            for j in rs:
                ana[i,j] = ana[i,j]-mn

    poissonVars = []
    for b in betaL:
        poissonVars.append(stats.poisson.rvs(b,size=(N,)))
    Y = np.zeros((N,a.shape[0]))
    for i in range(Y.shape[1]):
        for sl,pv in zip(slL, poissonVars):
            if i in sl:
                Y[:,i] = Y[:,i]+pv
    Z = Y<1

    #print around(np.corrcoef(Z,rowvar=0),decimals=2)
    #print around(Z.sum(axis=0)/N,decimals=2)
    return Z

def _columnRanks(u):
    """For matrix u, turn each element into its rank along the column
    Returns a matrix of same shape"""

    out = np.zeros(u.shape)
    for j in np.arange(u.shape[1]):
        out[:,j] = _argrank(u[:,j])
    return out.astype(int)

def _argrank(vec):
    """Return the rank (0 based) of the elements in vec"""
    sorti = np.argsort(vec)
    ranks = np.empty(len(vec), int)
    try:
        ranks[sorti] = np.arange(len(vec))
    except IndexError:
        ranks[sorti.values] = np.arange(len(vec))
    return ranks
