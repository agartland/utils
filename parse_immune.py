import pandas as pd
import numpy as np

__all__ = ['parseICS',
           'parseBAMA',
           'parseNAB',
           'parseRx',
           'unstackIR',
           'irLabels']

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

def _parseIR(fn, uVars, mag, subset={}, printUnique=False):
    raw = pd.read_csv(fn, dtype = {'ptid':str,'Ptid':str}, skipinitialspace = True)
    raw = raw.rename_axis({'Ptid':'ptid'}, axis=1)
    if uVars is None:
        uVars = raw.columns.drop(['ptid','response',mag]).tolist()

    if printUnique:
        for v in uVars:
            u = raw[v].astype(str).unique()
            if raw[v].dtype == object or len(u) <= 20:
                print '%s: %s' % (v, ', '.join(u))
            else:
                print '%s: mean %1.2f' % (v, np.nanmean(raw[v]))
        return

    cols = ['ptid','response','mag'] + uVars
    raw['mag'] = raw[mag]

    for k,v in subset.items():
        raw = raw.loc[raw[k].isin(v)]
    
    tmp = raw.set_index(uVars)
    ptids = tmp['ptid'].unique().shape[0]
    total = tmp.shape[0]
    conditions = tmp.index.unique().shape[0]
    printTuple = (ptids*conditions - total, ptids, conditions, ptids*conditions, total)
    if total > (ptids * conditions):
        print 'uVars are not sufficiently unique (%d): expected %d PTIDs x %d conditions = %d assays, found %d' % printTuple
    elif tmp.shape[0] < (tmp['ptid'].unique().shape[0] * tmp.index.unique().shape[0]):
        print 'Missing %d assays: expected %d PTIDs x %d conditions = %d assays, found %d' % printTuple

    """What about dropping the negative controls?"""
    return raw[cols]

def parseICS(fn, uVars=['visitno','tcellsub','cytokine','antigen'], mag='pctpos_adj', subset={}, printUnique=False):
    """Parse a processed ICS file.
    Returns one row per response, subsetting on subset values."""
    out = _parseIR(fn, uVars, mag, subset=subset, printUnique=printUnique)
    if not printUnique:
        """Enforce LOD"""
        out.loc[out.mag < 0.00025, 'mag'] = 0.00025
        out['mag'] = np.log10(out.mag)
    return out

def parseBAMA(fn, uVars=['isotype','antigen'], mag='delta', subset={}, printUnique=False):
    #cols = ['protocol','ptid','antigen','response','delta','rx_code','antigen_label','visitno']
    out = _parseIR(fn, uVars, mag, subset=subset, printUnique=printUnique)
    if not printUnique:
        """Enforce LOD"""
        #out.loc[out.mag < 1, 'mag'] = 1
        out['mag'] = np.log10(out.mag)
    return out

def parseNAB(fn, uVars=['celltype','virusdilution','isolate'], mag='titer_num', subset={}, printUnique=False):
    out = _parseIR(fn, uVars, mag, subset=subset, printUnique=printUnique)
    if not printUnique:
        out['mag'] = np.log10(out.mag)
    return out    

def parseRx(rxFn, demFn=None):
    trtCols = ['ptid', 'arm', 'grp', 'protocol', 'rx_code', 'rx']
    tmp = pd.read_csv(rxFn)
    tmp['ptid'] = tmp.Ptid
    trtDf = tmp[trtCols].set_index('ptid')
    
    if not  demFn is None:
        demCols = ['ptid','site','sex']
        demDf = pd.read_csv(demFn)
        #demDf['ptid'] = demDf.Ptid
        siteLists = dict(US = [121, 125, 126, 123, 127, 128, 129, 132, 133, 134, 167],
                        ZA = [138, 156, 157],
                        Lausanne = [168],
                        Peru = [150,621])
        siteTranslation = {}
        for k,v in siteLists.items():
            siteTranslation.update({n:k for n in v})
        demDf['site'] = demDf.DEMsitei.map(siteTranslation.get)
        demDf['sex'] = demDf.DEMsex
        trtDf = trtDf.join(demDf[demCols].set_index('ptid'))
    return trtDf