import pandas as pd
import numpy as np
import feather
import itertools
import time

from fg_shared import _fast, _git

import sys
from os.path import join as opj

sys.path.append(opj(_git, 'utils'))
from ics import parseSubsets, vec2subset

def gby_index2subsets(index):
    new_index = []
    for vals in index:
        s = ''
        for i, cy in enumerate(index.names):
            s += cy
            s += '+' if vals[i] == 1 else '-'
        new_index.append(s)
    return new_index

def extract_exhaustive(f, subsets, functions, markers, mincells=10):
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
        # cytnums.index.names = [n.split('/')[-1][:-1] for n in cytnums.index.names]
        cytnums.index.names = fkeys_stripped + mkeys_stripped

        levels = cytnums.index.names
        for k in range(2, len(levels) + 1):
            print(ssName, k)
            test = 0
            for cur_levels in itertools.combinations(levels, k):
                re_gby = cytnums.groupby(list(cur_levels), sort=False).sum()
                re_gby = re_gby.loc[re_gby >= mincells]
                new_index = gby_index2subsets(re_gby.index)
                out.append(pd.DataFrame({'subset':ssName,
                                         'marker':new_index,
                                         'nsub':nsub,
                                         'cytnum':re_gby.values}))
                '''
                test += 1
                if test > 2:
                    break
                '''
        cdf = pd.concat(out, axis=0)
    return cdf

def main():
    fn = opj(_fast, 'gilbert_p', 'grp', 'hvtn602_compass', 'tmpdata', '773-1', 'gs_1_sample_61918.fcs_333821.feather')
    d = feather.read_dataframe(fn)

    #gates_fn = opj(_fast, 'gilbert_p', 'grp', 'hvtn602_compass', 'tmpdata', 'config_files', 'flow_config_v2_CD4Tcells.csv')
    #gates_df = pd.read_csv(gates_fn)

    subsets_fn = opj(_git, 'utils', 'ics', 'subsets_CD4_gd_Tcells.csv')
    subsets, functions, markers, exclude = parseSubsets(subsets_fn)

    st = time.time()

    cdf = extract_exhaustive(d, subsets, functions, markers, mincells=10)
    
    et = (time.time() - st) / 60.
    print('Elapsed time: %1.2f min' % et)
    feather.write_dataframe(cdf, opj(_fast, 'gilbert_p', 'grp', 'hvtn602_compass', 'tmpdata', 'testextract.feather'))
    

if __name__ == '__main__':
    main()