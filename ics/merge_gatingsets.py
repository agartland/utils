import itertools
import pandas as pd
import numpy as np
import re
import feather
from os.path import join as opj
import os
from glob import glob
from functools import partial
import time
import sys

from .loading import applyResponseCriteria, generateGzAPerfExceptions

__all__ = ['matchSamples',
           'mergeSamples',
           'extractFunctions',
           'extractFunctionsMarkers',
           'extractFunctionsGBY',
           'extractFunctionsMarkersGBY',
           'parseSubsets']

"""
if sys.platform == 'win32':
    _dataPrefix = 'X:/'
    _homePrefix = 'A:/'
    GIT_PATH = 'A:/gitrepo/'
else:
    _dataPrefix = '/fh'
    _homePrefix = '/home/agartlan'
    GIT_PATH = '/home/agartlan/gitrepo/'

sys.path.append(opj(GIT_PATH, 'utils'))

from ics import *

dataFolder = opj(_dataPrefix, 'fast/gilbert_p/grp/compass_hvtn602_aw/tmpdata')
batchFolder = opj(dataFolder, '773-1')
metaFn = 'metadata.csv'
featherFn = 'gs_1_sample_61918.fcs_333821.feather'
f = feather.read_dataframe(opj(batchFolder, featherFn))
mDf = pd.read_csv(opj(batchFolder, metaFn))

subsetsFn = opj(_homePrefix, 'gitrepo/utils/ics/sample_subsets3.csv')
subsets, markers, functions, exclude = parseSubsets(subsetsFn)

cdf = extractFunctions(f, subsets, functions, compressions=[('ALL', 2)])
cmdf = extractFunctionsMarkers(f, subsets, functions, markers, compressions=[('ALL', 2)])

mDf = pd.read_csv(opj(batchFolder, 'metadata.csv'))
featherList = glob(opj(batchFolder, '*.feather'))

exKwargs = dict(subsets=subsets, functions=functions, compressions=[('ALL', 1),
                                                                    ('ALL', 2),
                                                                    (['IFNg','IL2', 'TNFa'], 1),
                                                                    (['IFNg','IL2', 'TNFa'], 2),
                                                                    (['IFNg','IL2'], 1)])

outDf = mergeSamples(batchFolder, extractionFunc=extractFunctions, extractionKwargs=exKwargs)
"""

def matchSamples(batchFolder, test=False):
    """Match each row of the metadata with each feather file (sample)
    in the batch folder"""
    mDf = pd.read_csv(opj(batchFolder, 'metadata.csv'))
    featherList = glob(opj(batchFolder, '*.feather'))
    featherLU = {sample_name:[fn for fn in featherList if sample_name in fn] for sample_name in mDf.sample_name}
    fallback = False
    if not len(featherLU) == mDf.shape[0]:
        print('Could not match all samples in the metadata.')
        fallback = True

    L = pd.Series({k:len(v) for k,v in featherLU.items()})
    if not (L == 1).all():
        print('Some samples in metadata matched to >1 feather file:')
        for k,v in featherLU.items():
            if len(v) > 1:
                print('\t%s: %s' % (k, v[:2]))
        fallback = True

    if fallback:
        featherLU = {}
        print('Attempting to use sample order with check on total event count.')
        for i,sample_name in enumerate(mDf.sample_name):
            events = int(sample_name.split('_')[-1])
            fn = [f for f in featherList if 'gs_%d_' % (i + 1) in f][0]
            f = feather.read_dataframe(opj(batchFolder, fn))
            if events == f.shape[0]:
                featherLU.update({sample_name:fn})
                print('Matched %s to %s. (%d of %d)' % (sample_name, fn, i+1, mDf.shape[0]))
                if test and (i + 1) >= 2:
                    break
            else:
                print('Sample order strategy not working.')
                break
    else:
        featherLU = {k:v[0] for k,v in featherLU.items()}

    if not len(featherLU) == mDf.shape[0]:
        print('Could not match all samples in the metadata.')
    if test:
        out = {}
        i = 0
        for k,v in featherLU.items():
            out.update({k:v})
            i += 1
            if i >= 2:
                break
        featherLU = out
    return featherLU

def mergeSamples(batchFolder, extractionFunc, extractionKwargs, test=False):
    """Go through each feather file (sample) in a batch folder,
    apply the analysis function, and merge together."""
    mDf = pd.read_csv(opj(batchFolder, 'metadata.csv'))
    featherList = glob(opj(batchFolder, '*.feather'))
    featherLU = matchSamples(batchFolder, test=test)

    mDf = pd.read_csv(opj(batchFolder, 'metadata.csv'))
    feathers = []
    i = 1
    print('Extracting from batch %s (%s)' % (batchFolder, time.ctime()))
    sttime = time.time()
    for sample_name, fn in featherLU.items():
        f = feather.read_dataframe(fn)
        print('Extracting from sample %s (%d of %d)' % (sample_name, i, len(featherLU)))
        x = extractionFunc(f, **extractionKwargs)
        x.loc[:, 'sample_name'] = sample_name
        feathers.append(x)
        i += 1

    outDf = pd.merge(pd.concat(feathers, axis=0), mDf, how='left', left_on='sample_name', right_on='sample_name')
    print('Finished batch %s (%1.0f minutes)' % (batchFolder, (time.time() - sttime) / 60), flush=True)
    return outDf

def subset2vec(cy):
    """Convert: "IFNg+IL2-TNFa+"
       To: (1, 0, 1)"""
    vec = np.array([1 if i == '+' else 0 for i in re.findall(r'[\+-]', cy)])
    return vec

def vec2subset(vec, cytokines):
    """Convert: (1, 0, 1)
        To: "IFNg+IL2-TNFa+" """
    s = ''
    for i,cy in enumerate(cytokines):
        s += cy
        s += '+' if vec[i] == 1 else '-'
    return s

def parseSubsets(subsetsFn):
    """Read on lists of subsets and functions from a config file"""
    df = pd.read_csv(subsetsFn)
    emptyName = df.name.isnull()
    df.loc[emptyName, 'name'] = df.loc[emptyName, 'value']
    subsets = df.loc[df.type == 'subset'].set_index('name')['value'].to_dict()
    markers = df.loc[df.type == 'marker'].set_index('name')['value'].to_dict()
    functions = df.loc[df.type == 'function'].set_index('name')['value'].to_dict()
    exclude = df.loc[df.type == 'exclude'].set_index('name')['value'].to_dict()
    return subsets, markers, functions, exclude

def extractFunctions(f, subsets, functions, compressions=None):
    """Extract functions from the GatingSet DataFrame and
    optionally apply response criteria.

    Parameters
    ----------
    subsets : dict
        From a config file, with names of subsets (keys) and column names (values)
    functions : dict
        From a config file, with names of functions (keys) and column name subset post-fixes (values)
    compression : list
        Optionally, provide list of cytokine subsets and ANYs for response calls. Use 'ALL' for all functions.
    
    Returns
    -------
    df : pd.DataFrame
        Data with columns: subset, cytokine, cytnum, nsub"""

    j = '/'
    cols = f.columns.tolist()
    newCols = {}
    """Subsets with functions"""
    for ssName, ssVal in subsets.items():
        fdict = {fk:fv for fk,fv in functions.items() if j.join([ssVal, fv]) in cols}
        fkeys = [fk for fk in fdict.keys()]
        fvals = [fdict[fk] for fk in fkeys]
        fkeys_stripped = [fk.replace('+','').replace('-','') for fk in fkeys]
        for vals in itertools.product(*[(0,1)]*len(fkeys)):
            """Go through each column name thats part of the subset function and
            parse into positive vs neg boolean comb."""
            neg = []
            pos = []
            for v,fv in zip(vals, fvals):
                nc = j.join([ssVal, fv])
                if v:
                    pos.append(nc)
                else:
                    neg.append(nc)
            ncName = (ssName, vec2subset(vals, fkeys_stripped))
            newCols[ncName] = {'pos':pos, 'neg':neg, 'subset':ssVal}

    out = []
    for ssName, cytokine in newCols:
        """Create the composite variable for each combination of variables/columns"""
        tmpk = (ssName, cytokine)
        nsub = f[newCols[tmpk]['subset']].values.sum(axis=0)
        """Start with the boolean index that is True for all cells in the subset"""
        posCols = [cols.index(c) for c in [newCols[tmpk]['subset']] + newCols[tmpk]['pos']]
        negCols = [cols.index(c) for c in newCols[tmpk]['neg']]

        """By doing this on two large matrices in numpy it is much faster"""
        ind = f.values[:, posCols].all(axis=1) & (~(f.values[:, negCols])).all(axis=1)
        cytnum = ind.sum(axis=0)
        
        tmp = {'subset':ssName,
                'cytokine':cytokine,
                'nsub':nsub,
                'cytnum':cytnum}
        out.append(tmp)
    cdf = pd.DataFrame(out)
    
    out = []
    if not compressions is None:
        """Apply response criteria to in "compressions"""
        for cytList, ANY in compressions:
            for ss in cdf.subset.unique():
                ssdf = cdf.loc[cdf.subset == ss]
                if not type(cytList) == list and cytList == 'ALL':
                    cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]
                else:
                    cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]
                    cytokines = [c for c in cytokines if c in cytList]
                    if len(cytokines) == 0:
                        cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]

                ssdf = applyResponseCriteria(ssdf,
                                             subset=cytokines,
                                             ANY=ANY,
                                             indexCols=['subset'],
                                             exceptions=generateGzAPerfExceptions(cytokines))
                out.append(ssdf)
        cdf = pd.concat(out, axis=0)
        cdf.index = np.arange(cdf.shape[0])

    return cdf

def extractFunctionsMarkers(f, subsets, functions, markers, compressions=[('ALL', 2)]):
    """Extract functions from the GatingSet DataFrame, then apply a response criteria
    before analyzing the proportion of positive/negative cells that
    express a combination of activation/phenotypic markers

    Parameters
    ----------
    subsets : dict
        From a config file, with names of subsets (keys) and column names (values)
    functions : dict
        From a config file, with names of functions (keys) and column name subset post-fixes (values)
    markers : dict
        From a config file, with names of functions (keys) and column name subset post-fixes (values)
    compression : list
        Optionally, provide list of cytokine subsets and ANYs for response calls. Use 'ALL' for all functions.
    
    Returns
    -------
    df : pd.DataFrame
        Data with columns: subset, cytokine, marker, cytnum, nsub"""
    
    def _prepKeys(d):
        fdict = {fk:fv for fk,fv in d.items() if j.join([ssVal, fv]) in cols}
        fkeys = [fk for fk in fdict.keys()]
        fvals = [fdict[fk] for fk in fkeys]
        fkeys_stripped = [fk.replace('+','').replace('-','') for fk in fkeys]
        return fkeys, fvals, fkeys_stripped

    j = '/'
    cols = f.columns.tolist()
    newCols = {}
    """Subsets with each function:marker combination"""
    for ssName, ssVal in subsets.items():
        fkeys, fvals, fkeys_stripped = _prepKeys(functions)
        mkeys, mvals, mkeys_stripped = _prepKeys(markers)
        for vals in itertools.product(*[(0,1)]*len(fkeys + mkeys)):
            """Go through each column name thats part of the subset function and
            parse into positive vs neg boolean comb."""
            fneg, mneg = [], []
            fpos, mpos = [], []
            for v,mfv in zip(vals, fvals + mvals):
                nc = j.join([ssVal, mfv])
                if v:
                    if mfv in fvals:
                        fpos.append(nc)
                    else:
                        mpos.append(nc)
                else:
                    if mfv in fvals:
                        fneg.append(nc)
                    else:
                        mneg.append(nc)
            ncName = (ssName, vec2subset(vals[:len(fkeys)], fkeys_stripped), vec2subset(vals[len(fkeys):], mkeys_stripped))
            newCols[ncName] = {'fpos':fpos, 'fneg':fneg,
                                'mpos':mpos, 'mneg':mneg,
                                'subset':ssVal}

    out = []
    for ssName, cytokine, marker in newCols:
        """Create the composite variable for each combination of variables/columns"""
        tmpk = (ssName, cytokine, marker)
        nsub = f[newCols[tmpk]['subset']].values.sum(axis=0)
        """Start with the boolean index that is True for all cells in the subset"""
        posCols = [newCols[tmpk]['subset']] + newCols[tmpk]['fpos'] + newCols[tmpk]['mpos']
        posCols = [cols.index(c) for c in posCols]

        negCols = newCols[tmpk]['fneg'] + newCols[tmpk]['mneg']
        negCols = [cols.index(c) for c in negCols]
        """By doing this on two large matrices in numpy it is much faster"""
        ind = f.values[:, posCols].all(axis=1) & (~(f.values[:, negCols])).all(axis=1)
        cytnum = ind.sum(axis=0)
        tmp = {'subset':ssName,
                'cytokine':cytokine,
                'marker':marker,
                'nsub':nsub,
                'cytnum':cytnum}
        out.append(tmp)
    
    cdf = pd.DataFrame(out)
    
    out = []
    """Apply response criteria to in "compressions"""
    for cytList, ANY in compressions:
        for ss in cdf.subset.unique():
            ssdf = cdf.loc[cdf.subset == ss]
            if not type(cytList) == list and cytList == 'ALL':
                cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]
            else:
                cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]
                cytokines = [c for c in cytokines if c in cytList]
                if len(cytokines) == 0:
                    cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]


            ssdf = applyResponseCriteria(ssdf,
                                         subset=cytokines,
                                         ANY=ANY,
                                         indexCols=['subset', 'marker'],
                                         exceptions=generateGzAPerfExceptions(cytokines))
            """Now marginalize across markers to get the nsub_cyt column for function positive cells"""
            tmp = ssdf.groupby(['subset', 'cytokine'])['cytnum'].agg(np.sum)
            tmp.name = 'nsub_cyt'
            ssdf = ssdf.set_index(['subset', 'cytokine']).join(tmp).reset_index()
            out.append(ssdf)
    cdf = pd.concat(out, axis=0)
    cdf.index = np.arange(cdf.shape[0])
    return cdf

def extractFunctionsGBY(f, subsets, functions, compressions=None, mincells=0):
    """Extract functions from the GatingSet DataFrame and
    optionally apply response criteria.

    Parameters
    ----------
    subsets : dict
        From a config file, with names of subsets (keys) and column names (values)
    functions : dict
        From a config file, with names of functions (keys) and column name subset post-fixes (values)
    compression : list
        Optionally, provide list of cytokine subsets and ANYs for response calls. Use 'ALL' for all functions.
    mincells : int
        Do not include function combinations with less than mincells.
    
    Returns
    -------
    df : pd.DataFrame
        Data with columns: subset, cytokine, cytnum, nsub"""
    def _prepKeys(d):
        fdict = {fk:fv for fk,fv in d.items() if j.join([ssVal, fv]) in f.columns}
        fkeys = [fk for fk in fdict.keys()]
        fvals = [fdict[fk] for fk in fkeys]
        fkeys_stripped = [fk.replace('+','').replace('-','') for fk in fkeys]
        return fkeys, fvals, fkeys_stripped

    j = '/'
    out = []
    for ssName, ssVal in subsets.items():
        fkeys, fvals, fkeys_stripped = _prepKeys(functions)
        ssDf = f.loc[f[ssVal]]
        gbyCols = [j.join([ssVal, v]) for v in fvals]
        if len(gbyCols) == 0:
            continue
        
        nsub = ssDf.shape[0]
        cytnums = ssDf.groupby(gbyCols, sort=False)[ssVal].count()
        cytnums.index.names = [n.split('/')[-1][:-1] for n in cytnums.index.names]

        for vals in itertools.product(*[(0,1)]*len(fkeys)):
            if vals in cytnums.index:
                cytnum = cytnums.loc[vals]
                if cytnum >= mincells:
                    tmp = {'subset':ssName,
                            'cytokine':vec2subset(vals[:len(fkeys)], fkeys_stripped),
                            'nsub':nsub,
                            'cytnum':cytnum}
            elif mincells > 0:
                tmp = {'subset':ssName,
                        'cytokine':vec2subset(vals[:len(fkeys)], fkeys_stripped),
                        'nsub':nsub,
                        'cytnum':0}
            out.append(tmp)
    cdf = pd.DataFrame(out)

    out = []
    if not compressions is None:
        """Apply response criteria to in "compressions"""
        for cytList, ANY in compressions:
            for ss in cdf.subset.unique():
                ssdf = cdf.loc[cdf.subset == ss]
                if not type(cytList) == list and cytList == 'ALL':
                    cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]
                else:
                    cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]
                    cytokines = [c for c in cytokines if c in cytList]
                    if len(cytokines) == 0:
                        cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]

                ssdf = applyResponseCriteria(ssdf,
                                             subset=cytokines,
                                             ANY=ANY,
                                             indexCols=['subset'],
                                             exceptions=generateGzAPerfExceptions(cytokines))
                out.append(ssdf)
        cdf = pd.concat(out, axis=0)
        cdf.index = np.arange(cdf.shape[0])

    return cdf

def extractFunctionsMarkersGBY(f, subsets, functions, markers, compressions=[('ALL', 2)]):
    """Extract functions from the GatingSet DataFrame, then apply a response criteria
    before analyzing the proportion of positive/negative cells that
    express a combination of activation/phenotypic markers

    Parameters
    ----------
    subsets : dict
        From a config file, with names of subsets (keys) and column names (values)
    functions : dict
        From a config file, with names of functions (keys) and column name subset post-fixes (values)
    markers : dict
        From a config file, with names of functions (keys) and column name subset post-fixes (values)
    compression : list
        Optionally, provide list of cytokine subsets and ANYs for response calls. Use 'ALL' for all functions.
    
    Returns
    -------
    df : pd.DataFrame
        Data with columns: subset, cytokine, marker, cytnum, nsub"""
    
    def _prepKeys(d):
        fdict = {fk:fv for fk,fv in d.items() if j.join([ssVal, fv]) in f.columns}
        fkeys = [fk for fk in fdict.keys()]
        fvals = [fdict[fk] for fk in fkeys]
        fkeys_stripped = [fk.replace('+','').replace('-','') for fk in fkeys]
        return fkeys, fvals, fkeys_stripped

    j = '/'
    """Subsets with each function:marker combination"""
    out = []
    for ssName, ssVal in subsets.items():
        fkeys, fvals, fkeys_stripped = _prepKeys(functions)
        mkeys, mvals, mkeys_stripped = _prepKeys(markers)
        ssDf = f.loc[f[ssVal]]
        gbyCols = [j.join([ssVal, v]) for v in fvals+mvals]
        if len(gbyCols) == 0:
            continue

        nsub = ssDf.shape[0]
        cytnums = ssDf.groupby(gbyCols, sort=False)[ssVal].count()
        # cytnum.index = cytnum.index.reorder_levels(gbyCols)
        cytnums.index.names = [n.split('/')[-1][:-1] for n in cytnums.index.names]

        for vals in itertools.product(*[(0,1)]*len(fkeys + mkeys)):
            if vals in cytnums.index:
                tmp = {'subset':ssName,
                        'cytokine':vec2subset(vals[:len(fkeys)], fkeys_stripped),
                        'marker':vec2subset(vals[len(fkeys):], mkeys_stripped),
                        'nsub':nsub,
                        'cytnum':cytnums.loc[vals]}
            else:
                tmp = {'subset':ssName,
                        'cytokine':vec2subset(vals[:len(fkeys)], fkeys_stripped),
                        'marker':vec2subset(vals[len(fkeys):], mkeys_stripped),
                        'nsub':nsub,
                        'cytnum':0}
            out.append(tmp)
    cdf = pd.DataFrame(out)
    
    out = []
    """Apply response criteria to in "compressions"""
    for cytList, ANY in compressions:
        for ss in cdf.subset.unique():
            ssdf = cdf.loc[cdf.subset == ss]
            if not type(cytList) == list and cytList == 'ALL':
                cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]
            else:
                cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]
                cytokines = [c for c in cytokines if c in cytList]
                if len(cytokines) == 0:
                    cytokines = ssdf.cytokine.iloc[0].replace('-','+').split('+')[:-1]


            ssdf = applyResponseCriteria(ssdf,
                                         subset=cytokines,
                                         ANY=ANY,
                                         indexCols=['subset', 'marker'],
                                         exceptions=generateGzAPerfExceptions(cytokines))
            """Now marginalize across markers to get the nsub_cyt column for function positive cells"""
            tmp = ssdf.groupby(['subset', 'cytokine'])['cytnum'].agg(np.sum)
            tmp.name = 'nsub_cyt'
            ssdf = ssdf.set_index(['subset', 'cytokine']).join(tmp).reset_index()
            out.append(ssdf)
    cdf = pd.concat(out, axis=0)
    cdf.index = np.arange(cdf.shape[0])
    return cdf

