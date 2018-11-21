import pandas as pd
import numpy as np
import multiprocessing

__all__ = ['parallel_groupby']

def parallel_groupby(gb, func, ncpus=4, concat=True):
    """Performs a Pandas groupby operation in parallel.
    Results is equivalent to the following:
    
    res = []
    for (name, group) in gb:
        res.append(func((name, group)))
    df =  pd.concat(got)

    OR

    df = gb.apply(func)

    
    Parameters
    ----------
    gb : pandas.core.groupby.DataFrameGroupBy
        Generator from calling df.groupby(columns)
    func : function
        Function that is called on each group taking one argument as input:
        a tuple of (name, groupDf)
    ncpus : int
        Number of CPUs to use for processing.

    Returns
    -------
    df : pd.DataFrame


    Example
    -------
    ep = groupby_parallel(posDf.groupby(['ptid', 'IslandID'], partial(findEpitopes, minSharedAA=5, minOverlap=7))"""

    with multiprocessing.Pool(ncpus) as pool:
        queue = multiprocessing.Manager().Queue()
        result = pool.starmap_async(func, [(name, group) for name, group in gb])
        got = result.get()
    if concat:
        out = pd.concat(got)
    else:
        out = got
    return out