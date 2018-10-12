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
           'vec2subset',
           'itersubsets',
           'subset2label',
           'subsetdf',
           'applyResponseCriteria',
           'computeMarginals',
           'generateGzAPerfExceptions']

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
    raw = raw.rename({'Ptid':'ptid'}, axis=1)
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
                          dtype={'visitday':np.int,
                                 'tcellsub':object,
                                 'cytokine':object,
                                 'antigen':object,
                                 'nsub':np.int,
                                 'cytnum':np.int,
                                 'nrepl':np.int},
                          converters={'ptid':_parsePTID})
    rdf = rdf.set_index(indexCols).sort_index()

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

def generateGzAPerfExceptions(cytokines):
    """Exclude subsets as positive if they include only GrzA or Perf and no other cytokines"""
    exceptions = []
    symLookup = {True:'+', False:'-'}
    for cys in itertools.product(*((True, False),)*len(cytokines)):
        """Loop over all combinations of +/- cytokines"""
        name = ''.join([cy + symLookup[include] for cy, include in zip(cytokines, cys)])
        if 'GzA-' in name and 'Perf+' in name and name.count('+') == 1:
            exceptions.append(name)
        elif 'GzA+' in name and 'Perf-' in name and name.count('+') == 1:
            exceptions.append(name)
        elif 'GzA+' in name and 'Perf+' in name and name.count('+') == 2:
            exceptions.append(name)
    return exceptions


def applyResponseCriteria(df, subset=['IFNg', 'IL2'], ANY=1, indexCols=None, magCols=['cytnum'], nsubCols=['nsub'], exceptions=[]):
    """Compress cytokine subsets into binary subsets (ie positive/negative)
    based on criteria such as ANY 1 of IFNg or IL2 (ie IFNg and/or IL2)

    Parameters
    ----------
    df : pd.DataFrame
        Raw stacked ICS data such that each row is a single data point defined by the columns in indexCols
    subset : list
        Cytokines considered for positivity.
    ANY : int
        Number of cytokines from subset required to be positive.
    magCols : list
        Columns that will be summed in the summary.
    nsubCols : list
        Columns that will be median'ed over in the compression.
        (they should all be identical though, since nsub is same for all cytokine subsets)

    Returns
    -------
    aggDf : pd.DataFrame
        A stacked dataset that has fewer unique cytokine subsets after marginalizing over cytokines not in subset."""
    
    """First compress to the relevant cytokines"""
    cdf = compressSubsets(df, subset=subset, indexCols=indexCols, groups=None, magCols=magCols, nsubCols=nsubCols)
    
    cytokineSubsets = cdf.cytokine.unique()
    cytokines = cytokineSubsets[0].replace('-', '+').split('+')[:-1]
    L = np.array([len(c) for c in cytokines])
    boolSubsets = np.array([[x[sum(L[:i+1])+i] == '+' for i in range(len(L))] for x in cytokineSubsets], dtype=bool)
    symLookup = {True:'+', False:'-'}
    
    base = '{} of {}'.format(ANY, len(subset)) + ' (' + '/'.join([subset2label(ss) for ss in subset]) + ')'
    pos = base
    neg = 'NOT ' + base
    convertLookup = {}
    for cys in itertools.product(*((True, False),)*len(subset)):
        """Loop over all combinations of +/- cytokines for the selected subset"""
        name = ''.join([cy + symLookup[include] for cy, include in zip(subset, cys)])
        if np.sum(cys) >= ANY and not name in exceptions:
            convertLookup[name] = pos
        else:
            convertLookup[name] = neg

    cdf['cytokine'] = cdf.cytokine.map(convertLookup.get)
    """Only keep the positive subset"""
    cdf = cdf.loc[cdf['cytokine'] == pos]
    """Groupby unique columns and agg-sum across cytokine subsets, then reset index"""
    out = cdf[indexCols + magCols + ['cytokine']].groupby(indexCols + ['cytokine']).agg(np.sum)

    if not nsubCols is None:
        nsubDf = cdf[indexCols + nsubCols + ['cytokine']].groupby(indexCols + ['cytokine']).agg(np.median)
        out = out.join(nsubDf)
   
    out = out.reset_index()
    return out

def compressSubsets(df, subset=['IFNg', '2', 'TNFa'], markerCol='cytokine', indexCols=None, groups=None, magCols=['cytnum'], nsubCols=['nsub']):
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
    magCols : list
        Columns that will be summed over in the compression.
    nsubCols : list
        Columns that will be median'ed over in the compression.
        (they should all be identical though, since nsub is same for all cytokine subsets)

    Returns
    -------
    aggDf : pd.DataFrame
        A stacked dataset that has fewer unique cytokine subsets after marginalizing over cytokines not in subset."""

    if indexCols is None:
        indexCols = ['sample', 'visitno', 'testdt', 'tcellsub', 'nsub', 'antigen', 'nrepl']
        
    cytokineSubsets = df[markerCol].unique()
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
        print("Need to specify cytokine/marker subsets!")
        return df

    preDf = df.copy()
    preDf[markerCol] = preDf[markerCol].map(convertLookup.get)
    """Groupby unique columns and agg-sum across cytokine subsets, then reset index"""
    #print('idx', indexCols)
    #print('m', markerCol, magCols)
    #print(preDf[markerCol].unique().shape, preDf[markerCol].unique())
    #print(preDf.groupby(markerCol)['ptid'].count())
    if preDf[markerCol].unique().shape[0] == df[markerCol].unique().shape[0]:
        """If subset == all cytokines then no need to groupby, just use the re-mapped column as an index"""
        aggDf = preDf[indexCols + magCols + [markerCol]].set_index(indexCols + [markerCol])
    else:
        """This groupby had problems when grouping by markers with too many unique values: Segmentation fault
        I never figured it out, but avoided it by skipping the groupby for the case when all markers are compressed.
        There's still a problem for the all but one case (though I don't need that ATM)
        This may be a memory constraint?"""
        aggDf = preDf[indexCols + magCols + [markerCol]].groupby(indexCols + [markerCol]).agg(np.sum)
    #print('OK')
    if not nsubCols is None:
        if preDf[markerCol].unique().shape[0] == df[markerCol].unique().shape[0]:
            nsubDf = preDf[indexCols + nsubCols + [markerCol]].set_index(indexCols + [markerCol])
        else:  
            nsubDf = preDf[indexCols + nsubCols + [markerCol]].groupby(indexCols + [markerCol]).agg(np.median)
        aggDf = aggDf.join(nsubDf)
    aggDf = aggDf.reset_index()
    return aggDf

def computeMarginals(df, indexCols, markerCol='cytokine', magCols=['cytnum'], nsubCols=['nsub']):
    """Compress df cytokine subsets to a single subset for each cytokine.

    Parameters
    ----------
    df : pd.DataFrame no index
        Raw or background subtracted ICS data
    indexCols : list
        Columns that make each sample unique
    magCol : str
        Typically "mag" or "bg"

    Returns
    -------
    df : pd.DataFrame
        Rows for each of the samples that were indexed,
        and for each cytokine"""

    cytokines = df[markerCol].iloc[0].replace('-', '+').split('+')[:-1]
    out = []
    for cy in cytokines:
        marg = compressSubsets(df, indexCols=indexCols, subset=[cy], magCols=magCols, markerCol=markerCol, nsubCols=nsubCols)
        marg = marg.loc[marg[markerCol] == cy + '+']
        out.append(marg)
    out = pd.concat(out, axis=0)
    #out.loc[:, magCols] = out
    return out

def subset2vec(cy):
    vec = np.array([1 if i == '+' else 0 for i in re.findall(r'[\+-]', cy)])
    return vec

def vec2subset(vec, cytokines=['IFNg', 'IL2', 'TNFa', 'IL4']):
    s = ''
    for i,cy in enumerate(cytokines):
        s += cy
        s += '+' if vec[i] == 1 else '-'
    return s

def subset2label(s, excludeNeg=False):
    def _convertGreek(s):
        
        conv = {'a':r'$\alpha$',
                'b': r'$\beta$',
                'g': r'$\gamma$'}
        return s[:-1] + conv.get(s[-1], s[-1])
        return s
    lookup = {'154': 'CD154',
              'Th2': 'IL4/IL13+',
              'DR': 'HLA-DR',
              'GzA+Perf+':'Granzyme A+\nPerforin+',
              '2':'IL2',
              '4':'IL4',
              '17':'IL17',
              '22':'IL22',
              'IFNg':'IFNg',
              'TNFa':'TNFa',
              'TN':'TNFa',
              'IF':'IFNg'}

    vec = re.findall(r'\w*[\+-]', s)
    if len(vec) == 0:
        out = _convertGreek(lookup.get(s, s))
    else:
        out = ''
        for cyt in vec:
            if not excludeNeg or cyt[-1] == '+':
                cy = lookup.get(cyt[:-1], cyt[:-1])
                # print(len(cy), '"{:>5s}"'.format(cy))
                tmp = _convertGreek('{:>5s}'.format(cy)) + '{}\n'.format(cyt[-1])
                out += tmp
        out = out[:-1]
    return out

def itersubsets(cytokines):
    vectors = [v for v in itertools.product(*((True, False),)*len(cytokines))]
    nfunctions = [sum(v) for v in vectors]
    sorti = np.argsort(nfunctions)
    return [vec2subset(vectors[i], cytokines) for i in sorti]

def subsetdf(df, **ss):
    tmp = df.copy()
    for k in ss:
         tmp = tmp.loc[tmp[k] == ss[k]]
    return tmp
