import pandas as pd
import numpy as np

__all__ = ['parseICS',
           'parseBAMA',
           'parseNAB',
           'parseRx',
           'unstackIR',
           'irLabels',
           'icsTicks',
           'icsTickLabels',
           'imputeNA']

icsTicks = np.log10([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1])
icsTickLabels = ['0.01', '0.025', '0.05', '0.1', '0.25', '0.5', '1']
# icsTicks = np.log10([0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
#icsTickLabels = ['0.01','0.025', '0.05', '0.1','0.2','0.4','0.6','0.8', '1']

def irLabels(c):
    poss = ['CD4+', 'CD8+', 'IgG', 'IgA']
    for p in poss:
        if c.find(p) >=0:
            return p
    return 'Other'

def unstackIR(df, uVars):
    """Return a response and magnitude df with one row per ptid
    and columns for each combination of uVars"""
    varFunc = lambda r: ' '.join(r[uVars])
    tmpDf = df.copy()

    tmpDf['var'] = tmpDf.apply(varFunc, axis=1)
    responseDf = tmpDf.pivot(index='ptid', columns='var', values='response')
    magDf = tmpDf.pivot(index='ptid', columns='var', values='mag')

    return responseDf, magDf
def _parsePTID(v):
    """Returns a string version of a PTID"""
    if pd.isnull(v):
        out = 'NA'
    elif np.isreal(v):
        out = '%1.0f' % v
    else:
        out = v
    out = out.replace('-', '')
    if out[-2:] == '.0':
        out = out[:-2]
    return out

def _parseIR(fn, uVars, mag, subset={}, printUnique=False, sep=','):
    raw = pd.read_csv(fn, dtype={'ptid':str, 'Ptid':str}, skipinitialspace=True, sep=sep)
    raw = raw.rename_axis({'Ptid':'ptid'}, axis=1)
    raw.loc[:, 'ptid'] = raw.loc[:, 'ptid'].map(_parsePTID)
    allCols = raw.columns.tolist()
    if uVars is None:
        uVars = raw.columns.drop(['ptid', 'response', mag]).tolist()

    if printUnique:
        for v in uVars:
            u = raw[v].astype(str).unique()
            if raw[v].dtype == object or len(u) <= 20:
                print('%s: %s' % (v, ', '.join(u)))
            else:
                print('%s: mean %1.2f' % (v, np.nanmean(raw[v])))
        return

    cols = []        
    for c in ['ptid', 'response']:
        if c in allCols and not c in uVars:
            cols = cols + [c]
    cols = cols + uVars + ['mag']

    raw['mag'] = raw[mag]

    """Keep rows that have one of the values in v for column k,
    for every key/value in subset dict"""
    for k, v in list(subset.items()):
        raw = raw.loc[raw[k].isin(v)]
    
    ptids = raw['ptid'].unique().shape[0]
    total = raw.shape[0]
    tmp = raw.set_index(uVars)
    conditions = tmp.index.unique().shape[0]
    
    printTuple = (ptids*conditions - total, ptids, conditions, ptids*conditions, total)
    if total > (ptids * conditions):
        print('uVars are not sufficiently unique (%d): expected %d PTIDs x %d conditions = %d assays, found %d' % printTuple)
    elif tmp.shape[0] < (tmp['ptid'].unique().shape[0] * tmp.index.unique().shape[0]):
        print('Missing %d assays: expected %d PTIDs x %d conditions = %d assays, found %d' % printTuple)

    """What about dropping the negative controls?"""
    return raw[cols]

def parseICS(fn, uVars=['visitno', 'tcellsub', 'cytokine', 'antigen'], mag='pctpos_adj', subset={}, printUnique=False):
    """Parse a processed ICS file.
    Returns one row per response, subsetting on subset values."""
    out = _parseIR(fn, uVars, mag, subset=subset, printUnique=printUnique)
    if not printUnique:
        """Enforce LOD"""
        out.loc[out.mag < 0.00025, 'mag'] = 0.00025
        out['mag'] = np.log10(out.mag)
    return out

def parseBAMA(fn, uVars=['isotype', 'antigen'], mag='delta', subset={}, printUnique=False):
    #cols = ['protocol','ptid','antigen','response','delta','rx_code','antigen_label','visitno']
    out = _parseIR(fn, uVars, mag, subset=subset, printUnique=printUnique)
    if not printUnique:
        """Enforce LOD"""
        #out.loc[out.mag < 1, 'mag'] = 1
        out['mag'] = np.log10(out.mag)
    return out

def parseNAB(fn, uVars=['celltype', 'virusdilution', 'isolate'], mag='titer_num', subset={}, printUnique=False):
    out = _parseIR(fn, uVars, mag, subset=subset, printUnique=printUnique)
    if not printUnique:
        out['mag'] = np.log10(out.mag)
    return out    

def parseRx(rxFn, demFn=None):
    trtCols = ['ptid', 'arm', 'grp', 'protocol', 'rx_code', 'rx']
    tmp = pd.read_csv(rxFn)
    tmp = tmp.rename_axis({'Ptid': 'ptid'}, axis=1)
    tmp.loc[:, 'ptid'] = tmp.ptid.str.replace('-', '')
    trtDf = tmp[trtCols].set_index('ptid')
    
    if not  demFn is None:
        demCols = ['ptid', 'site', 'sex']
        demDf = pd.read_csv(demFn)
        demDf = demDf.rename_axis({'Ptid': 'ptid'}, axis=1)
        demDf.loc[:, 'ptid'] = demDf.ptid.str.replace('-', '')
        
        siteLists = dict(US = [121, 125, 126, 123, 127, 128, 129, 132, 133, 134, 167],
                        ZA = [138, 156, 157],
                        Lausanne = [168],
                        Peru = [150, 621])
        siteTranslation = {}
        for k, v in list(siteLists.items()):
            siteTranslation.update({n:k for n in v})
        
        demDf['site'] = demDf.DEMsitei.map(siteTranslation.get)
        demDf['sex'] = demDf.DEMsex
        
        trtDf = trtDf.join(demDf[demCols].set_index('ptid'))
    return trtDf

def imputeNA(df, method='median', dropThresh=0.):
    """Impute missing values in a pd.DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        Data containing missing values.
    method : str
        Method fo imputation: median, mean, sample, regression
    dropThres : float
        Threshold for dropping rows: drop rows with fewer than 90% non-nan values

    Returns
    -------
    df : pd.DataFrame
        Copy of the input data with no missing values."""
        
    outDf = df.dropna(axis=0, thresh=np.round(df.shape[1] * dropThresh)).copy()
    if method == 'sample':
        for col in outDf.columns:
            naInd = outDf[col].isnull()
            outDf.loc[naInd, col] = outDf.loc[~naInd, col].sample(naInd.sum(), replace=True).values
    elif method == 'mean':
        for col in outDf.columns:
            naInd = outDf[col].isnull()
            outDf.loc[naInd, col] = outDf.loc[~naInd, col].mean()
    elif method == 'median':
        for col in outDf.columns:
            naInd = outDf[col].isnull()
            outDf.loc[naInd, col] = outDf.loc[~naInd, col].median()
    elif method == 'regression':
        naInds = []
        for col in outDf.columns:
            naInd = outDf[col].isnull()
            outDf.loc[naInd, col] = outDf.loc[~naInd, col].mean()
            naInds.append(naInd)
        for naInd,col in zip(naInds, outDf.columns):
            if naInd.sum() > 0:
                otherCols = [c for c in outDf.columns if not c == col]
                mod = sklearn.linear_model.LinearRegression().fit(outDf[otherCols], outDf[col])
                outDf.loc[naInd, col] = mod.predict(outDf.loc[naInd, otherCols])
    return outDf