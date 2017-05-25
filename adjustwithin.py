import numpy as np
import pandas as pd
import statsmodels.api as sm
from functools import partial
from copy import deepcopy

__all__ = ['adjustwithin', 'adjustnonnan']

def adjustnonnan(pvalues, method='holm'):
    """Convenient function for doing p-value adjustment.
    Accepts any matrix shape and adjusts across the entire matrix.
    Ignores nans appropriately.

    Parameters
    ----------
    pvalues : list, pd.DataFrame, pd.Series or np.ndarray
        Contains pvalues and optionally nans for adjustment.
    method : str
        An adjustment method for sm.stats.multipletests.
        Use 'holm' for Holm-Bonferroni FWER-adj and
        'fdr_bh' for Benjamini and Hochberg FDR-adj

    Returns
    -------
    adjpvalues : same as pvalues in type and shape"""

    """Turn it into a one-dimensional vector"""

    p = np.asarray(pvalues).flatten()

    """adjpvalues intialized with p to copy nans in the right places"""
    adjpvalues = deepcopy(p)
    
    nanInd = np.isnan(p)
    
    """Drop the nans, calculate adjpvalues, copy to adjpvalues vector"""
    rej, q, alphasidak, alphabon = sm.stats.multipletests(p[~nanInd], alpha=0.05, method=method)
    adjpvalues[~nanInd] = q
    
    """Reshape adjpvalues"""
    if not isinstance(pvalues, list):
        adjpvalues = adjpvalues.reshape(pvalues.shape)

    """Return same type as pvalues"""
    if isinstance(pvalues, list):
        return [pv for pv in adjpvalues]
    elif isinstance(pvalues, pd.core.frame.DataFrame):
        return pd.DataFrame(adjpvalues, columns=pvalues.columns, index=pvalues.index)
    elif isinstance(pvalues, pd.core.series.Series):
        return pd.Series(adjpvalues, name=pvalues.name, index=pvalues.index)
    else:
        return adjpvalues

def adjustwithin(df, pCol, withinCols, method='holm'):
    """Apply multiplicity adjustment to a "stacked"
    pd.DataFrame, adjusting within groups defined by
    combinations of unique values in withinCols

    Parameters
    ----------
    df : pd.DataFrame
        Stacked DataFrame with one column of pvalues
        and other columns to define groups for adjustment.
    pCol : str
        Column containing pvalues.
    withinCols : list
        Columns used to define subgroups/families for adjustment.
    method : str
        An adjustment method for sm.stats.multipletests.
        Use 'holm' for Holm-Bonferroni FWER-adj and
        'fdr_bh' for Benjamini and Hochberg FDR-adj

    Returns
    -------
    adjDf : pd.DataFrame
        Same shape as df, but with adjusted pvalues/adjpvalues."""

    def _transformFunc(ser, method):
        nonNan = ~ser.isnull()
        if nonNan.sum() >= 1:
            rej, adjp, alphas, alphab = sm.stats.multipletests(ser.loc[nonNan].values, method=method)
            out = ser.copy(deep=True)
            out.loc[nonNan] = adjp
            return out
        else:
            return ser
    
    if not len(withinCols) == 0:
        gby = df[[pCol] + withinCols].groupby(withinCols)

        adjDf = gby.transform(partial(_transformFunc, method=method))
        adjDf = df.drop(pCol, axis=1).join(adjDf)
    else:
        adjDf = df.copy()
        adjDf.loc[:, pCol] = adjustnonnan(adjDf.loc[:, pCol], method=method)
    return adjDf
