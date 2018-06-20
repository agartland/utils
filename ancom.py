
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from itertools import product
from mytstats import tstatistic
from skbio.stats import composition
from skbio.stats.composition import clr, multiplicative_replacement

__all__ = ['otuLogRatios',
           'ancom',
           'globalLRPermTest',
           'LRPermTest',
           'ratios2otumat',
           'loadAbundance',
           '_dmeanStat',
           '_sumDmeanStat',
           '_maxDmeanStat',
           '_tStat',
           '_sumTStat',
           '_maxTStat']

def _dmeanStat(mat, boolInd, axis=0):
    return mat[boolInd,:].mean(axis=axis) - mat[~boolInd,:].mean(axis=axis)
def _sumDmeanStat(mat, boolInd):
    return (_dmeanStat(mat, boolInd)**2).sum()
def _maxDmeanStat(mat, boolInd):
    return (_dmeanStat(mat, boolInd)**2).max()
def _tStat(mat, boolInd, axis=0):
    return tstatistic(mat[boolInd,:], mat[~boolInd,:], axis=axis, equal_var=True)
def _sumTStat(mat, boolInd, axis=0):
    return np.abs(_tStat(mat, boolInd)).sum()
def _maxTStat(mat, boolInd, axis=0):
    return np.abs(_tStat(mat, boolInd)).max()

def _rhoStat(mat, x, axis=0):
    assert mat.shape[axis] == x.shape[0]
    if axis == 0:
        r = [
            stats.spearmanr(x, mat[:, i]).correlation
            for i in range(mat.shape[1 - axis])
        ]
    else:
        r = [
            stats.spearmanr(x, mat[i, :]).correlation
            for i in range(mat.shape[1 - axis])
        ]
    r = np.array(r)
    assert r.shape[0] == mat.shape[1 - axis], (r.shape[0], mat.shape[1 - axis])
    return r
def _sumRhoStat(mat, x):
    return (_rhoStat(mat, x)**2).sum()
def _maxRhoStat(mat, x):
    return (_rhoStat(mat, x)**2).max()


def loadAbundance(filename, compositionNorm=True, truncate=True):
    """Load OTU counts file (phylum, genus or species level)
    with OTUs along the rows and samples along the columns.

    Parameters
    ----------
    filename : str
        Excel file from QIIME pipeline.
        Contains OTUs along the rows and samples along the columns,
        with a few header rows.
    compositionNorm : bool
        Add delta count to zeros and normalize each sample by the
        total number of reads. (uses skbio.stats.composition.multiplicative_replacement)
    truncate : bool
        Discard taxa with less than 0.5% of total reads.
        Discard taxa that are not present in 25% of samples.
        """
    def _cleanCountDf(df):
        """Drop extra columns/headers and transpose so that
        samples are along rows and OTUs along columns.

        Returns
        -------
        outDf : pd.DataFrame [index: samples, columns: OTUs]"""

        df = df.drop(['tax_id', 'rank'], axis = 1)
        df = df.dropna(subset=['tax_name'], axis = 0)
        df = df.rename_axis({'tax_name':'OTU'}, axis=1)
        df = df.set_index('OTU')
        df = df.drop(['specimen'], axis = 0)
        df = df.T
        df = df.dropna(subset=['label'], axis=0)
        df['sid'] = df.label.str.replace('Sample-', 'S')
        df = df.set_index('sid')
        df = df.drop('label', axis=1)
        df = df.astype(float)
        return df

    def _discardLow(df, thresh=0.005):
        """Discard taxa/columns with less than 0.5% of reads"""
        totReads = df.values.sum()
        keepInd1 = (df.sum(axis=0)/totReads) > thresh
        
        """Also discard taxa that are not present in 25% of samples"""
        keepInd2 = (df>0).sum(axis=0)/df.shape[0] > 0.25
        
        return df.loc[:, keepInd1 & keepInd2]
    
    df = pd.read_excel(filename)
    df = _cleanCountDf(df)
        
    if truncate:
        df = _discardLow(df)

    if compositionNorm:
        values = composition.multiplicative_replacement(df.values)
        df = pd.DataFrame(values, columns=df.columns, index=df.index)

    cols = [c for c in df.columns if not c in ['sid']]
    
    print('Abundance data: %s samples, %s taxa' % (df.shape[0], len(cols)))
    return df, cols

def ratios2otumat(otuDf, lrvec):
    """Reshape a vector of log-ratios back into a matrix of OTU x OTU
    using columns in otuDf

    Example
    -------
    qbyOTU = ratios2otumat(qvalues)

    Parameters
    ----------
    otuDf : pd.DataFrame [samples x OTUs]
        Contains relative abundance [0-1] for all samples (rows) and OTUs (colums)
    
    Returns:
    --------
    mat : pd.DataFrame [index: OTUs, columns: OTUs]"""

    nSamples, nOTUs = otuDf.shape
    otuMat = pd.DataFrame(np.zeros((nOTUs, nOTUs)), columns=otuDf.columns, index=otuDf.columns)
    for ind in lrvec.index:
        i = np.where(otuDf.columns == ind[0])[0]
        j = np.where(otuDf.columns == ind[1])[0]
        otuMat.values[i, j] = lrvec[ind]
        otuMat.values[j, i] = lrvec[ind]
    return otuMat


def otuLogRatios(otuDf):
    """Calculates pairwise log ratios between all OTUs for all samples.

    TODO: Use skbio.stats.composition.perturb_inv for simplicity and consistency
    (though I think the result will be identical)

    Parameters
    ----------
    otuDf : pd.DataFrame [samples x OTUs]
        Contains relative abundance [0-1] for all samples (rows) and OTUs (colums)
    
    Returns:
    --------
    logRatio : pd.DataFrame [index: (OTU1,OTU2) for each log-ratio]
        Log-ratio statistic for each comparison"""

    nSamples, nOTUs = otuDf.shape

    """Define minimum OTU abundance to avoid log(0)
    multiplicative_replacement takes matrix [samples x OTUs]"""
    assert otuDf.min().min() > 0, "Cannot input 0 values to otuLogRatios (min value {})".format(otuDf.min().min())
    logOTU = np.log(otuDf).values
    
    nRatios = int(nOTUs * (nOTUs-1) / 2)
    logRatio = np.zeros((nSamples, nRatios))

    """List of tuples of two indices for each ratio [nRatios]"""
    ratioIndices = [(otui, otuj) for otui in range(nOTUs - 1) for otuj in range(otui+1, nOTUs)]

    """List of indices corresponding to the ratios that contain each OTU"""
    otuIndices = [[j for j in range(nRatios) if otui in ratioIndices[j]] for otui in range(nOTUs)]
    
    ratioCount = 0
    for otui in range(nOTUs - 1):
        tmpCount = int(nOTUs - (otui+1))
        logRatio[:, ratioCount:(ratioCount+tmpCount)] =  logOTU[:, otui+1:] - logOTU[:, otui][:, None]
        ratioCount += tmpCount

    cols = [(otuDf.columns[ratioIndices[r][0]], otuDf.columns[ratioIndices[r][1]]) for r in range(nRatios)]
    logRatio = pd.DataFrame(logRatio, index=otuDf.index, columns=cols)
    return logRatio


def globalCLRPermTest(otuDf, labels, statfunc=_sumRhoStat, nperms=999, seed=110820, binary=False):
    """Calculates centered-log-ratios (CLR) for each sample and performs global
    permutation tests to determine if there is a significant correlation
    over all log-median-ratios, with respect to the label variable of interest.

    Parameters
    ----------
    otuDf : pd.DataFrame [samples x OTUs]
        Contains relative abundance [0-1] for all samples (rows) and OTUs (colums)
    labels: pd.Series (float)
        Contains binary variable indicating membership into one of two categories
        (e.g. treatment conditions). Must share index with otuDf.
    statfunc : function
        Takes a np.ndarray [n x k] and float index [n] as parameters and
        returns a float summarizing over k.
    nperms : int
        Number of iterations for the permutation test.
    seed :int
        Seed for random permutation generation.
    
    Returns:
    --------
    pvalue : float
        Global p-value for a significant association of OTU log-median-ratios
        with label, based on the summary statistic.
    obs : float
        Statistic summarizing the label difference."""

    nSamples, nOTUs = otuDf.shape

    if binary:
        labelValues = labels.values.astype(bool)
    else:
        labelValues = labels.values.astype(float)

    # Make proportions
    otuDf = otuDf / otuDf.sum()
    # Apply multiplicative replacement for zero values
    otuMR = multiplicative_replacement(otuDf.values)
    # Calculate the CLR
    otuCLR = clr(otuMR)
    # Make into a DataFrame
    otuCLR = pd.DataFrame(otuCLR, index=otuDf.index, columns=otuDf.columns)

    np.random.seed(seed)
    obs = statfunc(otuCLR.values, labelValues)
    samples = np.array([
        statfunc(otuCLR.values, labelValues[np.random.permutation(nSamples)])
        for permi in range(nperms)
    ])
    
    """Since test is based on the abs statistic it is inherently two-sided"""
    pvalue = ((np.abs(samples) >= np.abs(obs)).sum() + 1) / (nperms + 1)

    return pvalue, obs


def CLRPermTest(otuDf, labels, statfunc=_rhoStat, nperms=999, adjMethod='fdr_bh', seed=110820, binary=False):
    """Calculates centered-log-ratio (CLR) for all OTUs and performs
    permutation tests to determine if there is a significant correlation
    in OTU ratios with respect to the label variable of interest.

    Parameters
    ----------
    otuDf : pd.DataFrame [samples x OTUs]
        Contains relative abundance [0-1] for all samples (rows) and OTUs (colums)
    labels: pd.Series (float)
        Contains binary variable indicating membership into one of two categories
        (e.g. treatment conditions). Must share index with otuDf.
    statfunc : function
        Takes a np.array [n x k] and float index [n] as parameters and
        returns a 1-D array of the statistic [k].
    nperms : int
        Number of iterations for the permutation test.
    adjMethod : string
        Passed to sm.stats.multipletests for p-value multiplicity adjustment.
        If value is None then no adjustment is made.
    seed :int
        Seed for random permutation generation.
    
    Returns:
    --------
    qvalues : pd.Series [index: OTU]
        Q/P-values for each OTU computed.
    observed : pd.Series [index: OTU]
        Log-ratio statistic summarizing across samples."""

    nSamples, nOTUs = otuDf.shape

    if binary:
        labelValues = labels.values.astype(bool)
    else:
        labelValues = labels.values.astype(float)

    # Make proportions
    otuDf = otuDf / otuDf.sum()
    # Apply multiplicative replacement for zero values
    otuMR = multiplicative_replacement(otuDf.values)
    # Calculate the CLR
    otuCLR = clr(otuMR)
    # Make into a DataFrame
    otuCLR = pd.DataFrame(otuCLR, index=otuDf.index, columns=otuDf.columns)

    obs = statfunc(otuCLR.values, labelValues)

    np.random.seed(seed)
    samples = np.zeros((nperms, nOTUs))

    for permi in range(nperms):
        samples[permi, :] = statfunc(
            otuCLR.values,
            labelValues[np.random.permutation(nSamples)]
        )

    pvalues = ((np.abs(samples) >= np.abs(obs[None, :])).sum(
        axis=0) + 1) / (nperms + 1)

    if adjMethod is None or adjMethod.lower() == 'none':
        qvalues = pvalues
    else:
        qvalues = _pvalueAdjust(pvalues, method=adjMethod)

    qvalues = pd.Series(qvalues, index=otuDf.columns)
    observed = pd.Series(obs, index=otuDf.columns)

    return qvalues, observed

def globalLRPermTest(otuDf, labels, statfunc=_sumTStat, nperms=999, seed=110820):
    """Calculates pairwise log ratios between all OTUs and performs global
    permutation tests to determine if there is a significant difference
    over all log-ratios, with respect to the label variable of interest.

    Parameters
    ----------
    otuDf : pd.DataFrame [samples x OTUs]
        Contains relative abundance [0-1] for all samples (rows) and OTUs (colums)
    labels: pd.Series (bool or int)
        Contains binary variable indicating membership into one of two categories
        (e.g. treatment conditions). Must share index with otuDf.
    statfunc : function
        Takes a np.ndarray [n x k] and boolean index [n] as parameters and
        returns a float summarizing over k.
    nperms : int
        Number of iterations for the permutation test.
    seed :int
        Seed for random permutation generation.
    
    Returns:
    --------
    pvalue : float
        Global p-value for a significant association of OTU log-ratios
        with label, based on the summary statistic.
    obs : float
        Statistic summarizing the label difference."""

    nSamples, nOTUs = otuDf.shape

    # Make sure the label values are binary
    assert labels.unique().shape[0] == 2

    labelBool = labels.values.astype(bool)
    assert labels.unique().shape[0] == 2

    logRatio = otuLogRatios(otuDf)
    
    np.random.seed(seed)
    samples = np.zeros(nperms)
    obs = statfunc(logRatio.values, labelBool)
    for permi in range(nperms):
        rind = np.random.permutation(nSamples)
        samples[permi] = statfunc(logRatio.values, labelBool[rind])
    """Since test is based on the abs statistic it is inherently two-sided"""
    pvalue = ((np.abs(samples) >= np.abs(obs)).sum() + 1) / (nperms + 1)
    
    return pvalue, obs

def LRPermTest(otuDf, labels, statfunc=_dmeanStat, nperms=999, adjMethod='fdr_bh', seed=110820):
    """Calculates pairwise log ratios between all OTUs and performs
    permutation tests to determine if there is a significant difference
    in OTU ratios with respect to the label variable of interest.

    Parameters
    ----------
    otuDf : pd.DataFrame [samples x OTUs]
        Contains relative abundance [0-1] for all samples (rows) and OTUs (colums)
    labels: pd.Series (bool or int)
        Contains binary variable indicating membership into one of two categories
        (e.g. treatment conditions). Must share index with otuDf.
    statfunc : function
        Takes a np.array [n x k] and boolean index [n] as parameters and
        returns a 1-D array of the statistic [k].
    nperms : int
        Number of iterations for the permutation test.
    adjMethod : string
        Passed to sm.stats.multipletests for p-value multiplicity adjustment.
        If value is None then no adjustment is made.
    seed :int
        Seed for random permutation generation.
    
    Returns:
    --------
    qvalues : pd.Series [index: (OTU1,OTU2) for each log-ratio]
        Q/P-values for each log-ratio computed. otuQvalues is a reorganization of this.
    observed : pd.Series [index: (OTU1,OTU2) for each log-ratio]
        Log-ratio statistic summarizing across samples."""

    nSamples, nOTUs = otuDf.shape

    # Make sure the label values are binary
    assert labels.unique().shape[0] == 2

    labelBool = labels.values.astype(bool)
    assert labels.unique().shape[0] == 2

    nRatios = int(nOTUs * (nOTUs-1) / 2)

    """List of tuples of two indices for each ratio [nRatios]"""
    ratioIndices = [(otui, otuj) for otui in range(nOTUs - 1) for otuj in range(otui+1, nOTUs)]

    """List of indices corresponding to the ratios that contain each OTU"""
    otuIndices = [[j for j in range(nRatios) if otui in ratioIndices[j]] for otui in range(nOTUs)]

    logRatio = otuLogRatios(otuDf)
    
    np.random.seed(seed)
    samples = np.zeros((nperms, nRatios))
    obs = statfunc(logRatio.values, labelBool)
    for permi in range(nperms):
        rind = np.random.permutation(nSamples)
        samples[permi,:] = statfunc(logRatio.values, labelBool[rind])
    pvalues = ((np.abs(samples) >= np.abs(obs[None,:])).sum(axis=0) + 1) / (nperms + 1)

    if adjMethod is None or adjMethod.lower() == 'none':
        qvalues = pvalues
    else:
        qvalues = _pvalueAdjust(pvalues, method=adjMethod)

    cols = [(otuDf.columns[ratioIndices[r][0]], otuDf.columns[ratioIndices[r][1]]) for r in range(nRatios)]
    qvalues = pd.Series(qvalues, index=cols)
    observed = pd.Series(obs, index=cols)
    
    return qvalues, observed

def ancom(otuDf, labels, alpha=0.2, statfunc=_dmeanStat, nperms=0, adjMethod='fdr_bh', seed=110820):
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
    seed :int
        Seed for random permutation generation (if nperms > 0)
    
    Returns:
    --------
    rej : pd.Series [index: OTUs]
        Boolean indicating whether the null hypothesis is rejected for each OTU.
    otuQvalues : pd.DataFrame [index: OTUs, columns: nOTUs - 1]
        Q/P-value for each of the log-ratios for each OTU.
    qvalues : pd.Series [index: (OTU1,OTU2) for each log-ratio]
        Q/P-values for each log-ratio computed. otuQvalues is a reorganization of this.
    logRatio : pd.DataFrame [index: samples, coluns: (OTU1,OTU2) for each log-ratio]
        Log-ratio statistic for each comparison"""

    nSamples, nOTUs = otuDf.shape

    labelBool = labels.values.astype(bool)

    nRatios = int(nOTUs * (nOTUs-1) / 2)

    """List of tuples of two indices for each ratio [nRatios]"""
    ratioIndices = [(otui, otuj) for otui in range(nOTUs - 1) for otuj in range(otui+1, nOTUs)]

    """List of indices corresponding to the ratios that contain each OTU"""
    otuIndices = [[j for j in range(nRatios) if otui in ratioIndices[j]] for otui in range(nOTUs)]

    logRatio = otuLogRatios(otuDf)
    
    if nperms > 0:
        np.random.seed(seed)
        samples = np.zeros((nperms, nRatios))
        obs = statfunc(logRatio.values, labelBool)
        for permi in range(nperms):
            rind = np.random.permutation(nSamples)
            samples[permi,:] = statfunc(logRatio.values, labelBool[rind])
        pvalues = ((np.abs(samples) >= np.abs(obs[None,:])).sum(axis=0) + 1) / (nperms + 1)
    else:
        pvalues = np.zeros(nRatios)
        for ratioi in range(nRatios):
            _, pvalues[ratioi] = stats.ranksums(logRatio.values[labelBool, ratioi], logRatio.values[~labelBool, ratioi])   

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
    _, q, _, _ = sm.stats.multipletests(p[~nanInd], alpha=0.2, method=method)
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
