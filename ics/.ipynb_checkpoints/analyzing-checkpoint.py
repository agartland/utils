import pandas as pd
import numpy as np
from myfisher import fisherTestVec
import statsmodels.api as sm

__all__ = ['computePvalues',
           'pivotPvalues']

def computePvalues(df, indexCols=['sample', 'TreatmentGroupID', 'visitno', 'testdt', 'tcellsub', 'cytokine'], replicateCol='nrepl', bgCutoff=0.1):
    """Match negative control wells with antigen wells and compute pvalues,
    first discounting participants with high background for any one single cytokine.
    Fisher's test uses a one-sided alternative: pos > neg   

    Combines negative control replicates using replicate column.

    All columns not in indexCols will be discarded.

    Uses a C implementation of a Fisher's exact test that is fast.

    Parameters
    ----------
    df : pd.DataFrame
        Contains stacked data that can be uniquely identified by indexCols
    indexCols : list
        Columns that make each row/observation unique.
        All other rows will get summed over.
        (TODO: check to see that indexCols work and that there aren't duplicate rows
    replicateCol : str
        Column in df that will be summed over for combining replicates.
    bgCutoff : float or None
        Percentage of cells positive in the negative control that is
        the exclusion threshold for high-background. None for no exclusion.

    Returns
    -------
    jDf : pd.DataFrame
        Columns are indexCols plus a pvalue and OR column for Fisher's exact test"""

    dataCols = ['antigen', 'cytnum', 'nsub']
    negInd = df.antigen == 'negctrl'
    """Isolate the negative controls and add the replicates"""
    negDf = df.loc[negInd, indexCols + dataCols + [replicateCol]]
    negDf = negDf.groupby(indexCols).agg(sum)
    """Isolate the counts for each antigen and join with the negative controls"""
    posDf = df.loc[~negInd, indexCols + dataCols + [replicateCol]]
    """TODO: may need to sum across replicates at some point in the future"""
    posDf = posDf.set_index(indexCols)
    jDf = posDf.join(negDf, how='left', rsuffix='_neg', lsuffix='_pos')
    """Run a Fisher's exact test for every row using a vectorized test written in C"""
    aVec = jDf.cytnum_pos.values
    bVec = jDf.nsub_pos.values - jDf.cytnum_pos.values
    cVec = jDf.cytnum_neg.values
    dVec = jDf.nsub_neg.values - jDf.cytnum_neg.values
    jDf['OR'], jDf['pvalue'] = fisherTestVec(aVec, bVec, cVec, dVec, alternative='greater')
    jDf['bg'] = jDf.cytnum_neg/jDf.nsub_neg
    jDf['mag'] = jDf.cytnum_pos/jDf.nsub_pos
    jDf['mag_adj'] = jDf.mag - jDf.bg

    """Mark subsets with high-background (>0.1%) by setting p-value = nan"""
    if not bgCutoff is None:
        jDf.loc[(100*jDf['mag']) > bgCutoff, 'pvalue'] = np.nan
    return jDf.reset_index()

def pivotPvalues(jDf, adjust=False, subsets=None):
    """Pivot stacked p-values into a df with one row per participant.
    
    Optionally compute adjusted pvalues for the specified cytokine subsets.

    Uses statsmodels.stats.multipletests 'holm' method,
    a step-down method using Bonferroni adjustments.

    Parameters
    ----------
    jDf : pd.DataFrame
        Stacked data with one row/pvalue per participant per cytokine subset.
    adjust : bool
        Specifies whether Holm-Bonferroni adjustment should be applied to the p-values
    subsets : list
        Cytokine subsets in the "cytokine" column over which adjustment will be performed.
        By default it will adjust over all but the all-negative subset (e.g. excluding IFNg-2-TNFa-)"""

    pivotDf = jDf.pivot(index='sample', columns='cytokine', values='pvalue')
    if adjust:
        cytokineSubsets = jDf.cytokine.unique()
        if subsets is None:
            cytokines = cytokineSubsets[0].replace('-', '+').split('+')[:-1]
            subsets = [c for c in cytokineSubsets if not c == '-'.join(cytokines)+'-']
        notAdjusted = [s for s in cytokineSubsets if s not in subsets]
        
        pvalues = pivotDf[subsets].values
        adj_pvalues = np.zeros(pvalues.shape)
        for rowi in range(pvalues.shape[0]):
            H, adj_pvalues[rowi,:], _, _ = sm.stats.multipletests(pvalues[rowi,:], method='holm')
        pivotDf.loc[:, subsets] = adj_pvalues
        pivotDf.loc[:, notAdjusted] = np.nan
    return pivotDf