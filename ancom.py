from __future__ import division
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from itertools import product
from mytstats import tstatistic

__all__ = ['ancom']

def _dmeanStat(mat, boolInd, axis=0):
    return mat[boolInd, :].mean(axis=axis) - mat[~boolInd, :].mean(axis=axis)
def _tStat(mat, boolInd, axis=0):
    return tstatistic(mat[boolInd, :], mat[~boolInd, :], axis=axis, equal_var=True)

def ancom(otuDf, labels, alpha=0.2, statfunc=_dmeanStat, nperms=0, adjMethod='fdr_bh'):
    """Calculates pairwise log ratios between all OTUs and performs
    permutation tests to determine if there is a significant difference
    in OTU ratios with respect to the label variable of interest.

    Algorithm is from:
    Mandal, Siddhartha, Will Van Treuren, Richard A White, Merete Eggesbo,
        Rob Knight, and Shyamal D Peddada. 2015. "Analysis of Composition
        of Microbiomes: A Novel Method for Studying Microbial Composition."
        Microbial Ecology in Health and Disease 26: 27663. doi:10.3402/mehd.v26.27663.
    
    Parameters
    ----------
    otuDf : pd.DataFrame [samples x OTUs]
        Contains relative abundance [0-1] for all samples (rows) and OTUs (colums)
    labels: pd.Series (bool or int)
        Contains binary variable indicating membership into one of two categories
        (e.g. treatment conditions). Must share index with otuDf.
    alpha : float
        Alpha cutoff for rejecting a log-ratio hypothesis.
        If multiAdj is True, then this is a FDR-adjusted q-value cutoff.
    statfunc : function
        Takes a np.array [n x k] and boolean index [n] as parameters and
        returns a 1-D array of the statistic [k].
    nperms : int
        Number of iterations for the permutation test.
        If nperms = 0, then use Wilcoxon ranksum test to compute pvalue.
    adjMethod : string
        Passed to sm.stats.multipletests for p-value multiplicity adjustment.
        If value is None then no adjustment is made.
    wCutoff : int
        Cutoff for the number of Q/p-values required for a significant OTU
        Reject H0 if sum(q < alpha) >= (nOTUs - wCutoff)
    
    Returns:
    --------
    rej : pd.Series [index: OTUs]
        Boolean indicating whether the null hypothesis is rejected for each OTU.
    otuQvalues : pd.DataFrame [index: OTUs, columns: nOTUs - 1]
        Q/P-value for each of the log-ratios for each OTU.
    qvalues : pd.Series [index: (OTU1,OTU2) for each log-ratio]
        Q/P-values for each log-ratio computed. otuQvalues is a reorganization of this.
    logRatio : pd.DataFrame [index: (OTU1,OTU2) for each log-ratio]
        Log-ratio statistic for each comparison"""

    nSamples, nOTUs = otuDf.shape

    labelBool = labels.values.astype(bool)

    """Define minimum OTU abundance to avoid log(0)"""
    constant = otuDf.values[otuDf.values>0].min()/2
    logOTU = np.log(otuDf + constant).values
    
    nRatios = int(nOTUs * (nOTUs-1) / 2)
    logRatio = np.zeros((nSamples, nRatios))

    """List of tuples of two indices for each ratio [nRatios]"""
    ratioIndices = [(otui,otuj) for otui in range(nOTUs - 1) for otuj in range(otui+1,nOTUs)]

    """List of indices corresponding to the ratios that contain each OTU"""
    otuIndices = [[j for j in range(nRatios) if otui in ratioIndices[j]] for otui in range(nOTUs)]
    
    ratioCount = 0
    for otui in range(nOTUs - 1):
        tmpCount = int(nOTUs - (otui+1))
        logRatio[:, ratioCount:(ratioCount+tmpCount)] =  logOTU[:, otui+1:] - logOTU[:,otui][:,None]
        ratioCount += tmpCount
    
    if nperms > 0:
        samples = np.zeros((nperms, nRatios))
        obs = statfunc(logRatio, labelBool)
        for permi in range(nperms):
            rind = np.random.permutation(nSamples)
            samples[permi, :] = statfunc(logRatio, labelBool[rind])
        pvalues = ((np.abs(samples) >= np.abs(obs[None, :])).sum(axis=0) + 1) / (nperms + 1)
    else:
        pvalues = np.zeros(nRatios)
        for ratioi in range(nRatios):
            _,pvalues[ratioi] = stats.ranksums(logRatio[labelBool,ratioi], logRatio[~labelBool,ratioi])   

    if adjMethod is None or adjMethod.lower() == 'none':
        qvalues = pvalues
    else:
        qvalues = _pvalueAdjust(pvalues, method=adjMethod)

    otuQvalues = np.asarray([qvalues[ind] for ind in otuIndices])

    """Number of hypotheses rejected, for each OTU"""
    W = (otuQvalues < alpha).sum(axis=1)

    """Use cutoff of (nOTUs - 1), requiring that all log-ratios are significant for a given OTU (quite conservative)"""
    rej = pd.Series(W >= (nOTUs-1), index=otuDf.columns)

    otuQvalues = pd.DataFrame(otuQvalues, index=otuDf.columns, columns=['ratio_%d' % i for i in range(nOTUs-1)])
    cols = [(otuDf.columns[ratioIndices[r][0]], otuDf.columns[ratioIndices[r][1]]) for r in range(nRatios)]
    qvalues = pd.Series(qvalues, index=cols)
    logRatio = pd.DataFrame(logRatio, index=otuDf.index, columns=cols)
    
    return rej, otuQvalues, qvalues, logRatio

def _pvalueAdjust(pvalues, method='fdr_bh'):
    """Convenience function for doing p-value adjustment.
    Accepts any matrix shape and adjusts across the entire matrix.
    Ignores nans appropriately.

    1) Pvalues can be DataFrame or Series or array
    2) Turn it into a one-dimensional vector
    3) Qvalues intialized at p to copy nans in the right places
    4) Drop the nans, calculate qvalues, copy to qvalues vector
    5) Reshape qvalues
    6) Return same type as pvalues"""

    p = np.asarray(pvalues).ravel()
    qvalues = p.copy()
    nanInd = np.isnan(p)
    _,q,_,_ = sm.stats.multipletests(p[~nanInd], alpha=0.2, method=method)
    qvalues[~nanInd] = q
    qvalues = qvalues.reshape(pvalues.shape)

    if isinstance(pvalues, pd.core.frame.DataFrame):
        return pd.DataFrame(qvalues, columns=[x+'_q' for x in pvalues.columns], index=pvalues.index)
    elif isinstance(pvalues, pd.core.series.Series):
        return pd.Series(qvalues, name=pvalues.name+'_q', index=pvalues.index)
    else:
        return qvalues

    """Code for using a different cutoff for W from the ANCOM supplement"""
    """W = np.zeros(n_otu)
    for i in range(n_otu):
        W[i] = sum(logratio_mat[i,:] < alpha)
    par = n_otu-1 #cutoff

    c_start = max(W)/par
    cutoff = c_start - np.linspace(0.05,0.25,5)
    D = 0.02 # Some arbituary constant
    dels = np.zeros(len(cutoff))
    prop_cut = np.zeros(len(cutoff),dtype=np.float32)
    for cut in range(len(cutoff)):
        prop_cut[cut] = sum(W > par*cutoff[cut])/float(len(W))
    for i in range(len(cutoff)-1):
        dels[i] = abs(prop_cut[i]-prop_cut[i+1])
        
    if (dels[1]<D) and (dels[2]<D) and (dels[3]<D):
        nu=cutoff[1]
    elif (dels[1]>=D) and (dels[2]<D) and (dels[3]<D):
        nu=cutoff[2]
    elif (dels[2]>=D) and (dels[3]<D) and (dels[4]<D):
        nu=cutoff[3]
    else:
        nu=cutoff[4]
    up_point = min(W[W>nu*par])
    results = otu_table.columns[W>=nu*par]
    return results"""