"""Generate HVTN505 dataset for Michael on statsrv"""
import pandas as pd
import numpy as np
import re
import itertools

__all__ = ['parseProcessed',
           'parseRaw',
           'unstackIR',
           'compressSubsets',
           'subset2vec',
           'vec2subset']

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

def parseProcessed(fn, uVars=['visitno', 'tcellsub', 'cytokine', 'antigen'], mag='pctpos_adj', subset={}, printUnique=False):
    """Parse a processed ICS file.
    Returns one row per response, subsetting on subset values."""
    out = _parseIR(fn, uVars, mag, subset=subset, printUnique=printUnique)
    if not printUnique:
        """Enforce LOD"""
        out.loc[out.mag < 0.00025, 'mag'] = 0.00025
        out['mag'] = np.log10(out.mag)
    return out

def parseRaw(fn):
    """Process raw ICS file (all subsets all counts) and adjust for background"""
    ctrlCols = ['ptid', 'visitday', 'tcellsub', 'cytokine']
    indexCols = ctrlCols + ['antigen']

    rdf = pd.read_csv(fn, usecols=indexCols + ['nsub', 'cytnum', 'nrepl'],
                          dtype={'ptid':object,
                                 'visitday':np.int,
                                 'tcellsub':object,
                                 'cytokine':object,
                                 'antigen':object,
                                 'nsub':np.int,
                                 'cytnum':np.int,
                                 'nrepl':np.int},
                          converters={'ptid':_parsePTID},
                          index_col=indexCols).sort_index()

    uAg = [ag for ag in rdf.index.levels[-1] if not np.any([ag.find(s) >= 0 for s in ['negctrl', 'phactrl']])]

    """Sum the negative control replicates"""
    ndf = rdf.xs('negctrl', level='antigen').reset_index().groupby(ctrlCols)[['nsub', 'cytnum']].agg(np.sum)
    ndf.loc[:, 'bg'] = ndf['cytnum'] / ndf['nsub']

    """Define the magnitude as the fraction of cytokine positive cells"""
    pdf = rdf.loc[(slice(None), slice(None), slice(None), slice(None), uAg), :]
    pdf.loc[:, 'mag'] = pdf['cytnum'] / pdf['nsub']

    """Subtract off the background/negative control"""
    df = pdf['mag'].reset_index().join(ndf['bg'], on=ctrlCols)
    return df

def compressSubsets(df, subset=['IFNg', '2', 'TNFa'], indexCols=['sample', 'visitno', 'testdt', 'tcellsub', 'nsub', 'antigen', 'nrepl'], groups=None, magCol='cytnum'):
    """Combine cell subsets into a smaller number of subsets before performing the analysis.
    Data will be summed-over cytokines not included in the subset list.

    Parameters
    ----------
    df : pd.DataFrame
        Raw stacked ICS data such that each row is a single data point defined by the columns in indexCols
    subset : list
        Cytokines whose combinations will define the compressed responses.
    indexCols : list
        List of columns in df that make each row unique.
    groups : dict of lists
        Alternatively, use groups of cytokine subsets to compress (e.g. ANY 1 marker)
        New column names will be the keys in groups.

    Returns
    -------
    aggDf : pd.DataFrame
        A stacked dataset that has fewer unique cytokine subsets after marginalizing over cytokines not in subset."""
        
    cytokineSubsets = df.cytokine.unique()
    cytokines = cytokineSubsets[0].replace('-', '+').split('+')[:-1]
    L = np.array([len(c) for c in cytokines])
    boolSubsets = np.array([[x[sum(L[:i+1])+i] == '+' for i in range(len(L))] for x in cytokineSubsets], dtype=bool)
    symLookup = {True:'+', False:'-'}

    if not subset is None:
        convertLookup = {}
        for cys in itertools.product(*((True, False),)*len(subset)):
            """Loop over all combinations of +/- cytokines for the selected subset"""
            subsInd = np.ones(cytokineSubsets.shape[0], dtype=bool)
            for cy, include in zip(subset, cys):
                """Build up an index based on all possible subsets in df"""
                if include:
                    subsInd = subsInd & boolSubsets[:, cytokines.index(cy)]
                else:
                    subsInd = subsInd & (~boolSubsets[:, cytokines.index(cy)])
            name = ''.join([cy + symLookup[include] for cy, include in zip(subset, cys)])
            """Rename the longer column name to the subset name so that it can be aggregated over"""
            convertLookup.update({sub:name for sub in cytokineSubsets[subsInd]})
    elif not groups is None:
        convertLookup = {}
        for g in groups:
            convertLookup.update({v:g for v in groups[g]})
    else:
        print("Need to specify cytokine subsets!")
        return df

    aggDf = df.copy()
    aggDf['cytokine'] = aggDf.cytokine.map(convertLookup.get)
    """Groupby unique columns and agg-sum across cytokine subsets, then reset index"""
    aggDf = aggDf[indexCols + [magCol, 'cytokine']].groupby(indexCols + ['cytokine']).agg(sum)
    aggDf = aggDf.reset_index()
    return aggDf

def subset2vec(cy, nsubsets=4):
    m = re.match(r'.*([\+-])'*nsubsets, cy)
    if not m is None:
        vec = np.zeros(len(m.groups()))
        for i,g in enumerate(m.groups()):
            vec[i] = 1 if g == '+' else 0
    return vec

def vec2subset(vec, cytokines=['IFNg', 'IL2', 'TNFa', 'IL4']):
    s = ''
    for i,cy in enumerate(cytokines):
        s += cy
        s += '+' if vec[i] == 1 else '-'
    return s







