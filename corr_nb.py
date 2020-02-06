import numba
import numpy as np
import pandas as pd
import itertools

__all__ = ['crosscorr']

def _mat_stat(mat, indices, stat_func, min_n, method):
    results = np.zeros((indices.shape[0], 3))

    for pairi in range(indices.shape[0]):
        i, j = indices[pairi, :]
        notnan = (~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j]))
        n = np.sum(notnan)
        if n >= min_n:
            rho, pvalue = stat_func(mat[notnan, i], mat[notnan, j], method)
        else:
            rho, pvalue = np.nan, np.nan
        results[pairi, 0] = rho
        results[pairi, 1] = pvalue
        results[pairi, 2] = n
    return results

_mat_stat_nb = numba.jit(_mat_stat, nopython=True, parallel=True, error_model='numpy')

def crosscorr(data, left_cols=None, right_cols=None, method='spearman', min_n=5):
    """Compute correlation among columns of data, ignoring NaNs.
    Uses numba for speedup and parallelization.
    Results have been tested against scipy.stats.spearmanr and scipy.stats.pearsonr

    Parameters
    ----------
    left_cols, right_cols : list of columns in data
        If both are present then return correlations between columns in left vs. right.
        If only left is not None then limit correlations to left vs. left.
        If neither are not None then compute all correlations.
    method : str
        Method can be spearman or pearson.
    min_n : int
        Minimum number of observation required to compute the correlation
        (otherwise NaN is returned)

    Returns
    -------
    results : pd.DataFrame
        Results with columns: Left, Right, rho, pvalue, N"""
    if left_cols is None:
        left_cols = data.columns
    if right_cols is None:
        right_cols = left_cols
    
    columns = data.columns.tolist()
    l_coli = [columns.index(c) for c in left_cols]
    r_coli = [columns.index(c) for c in right_cols]
    
    column_pairs = np.array([ij for ij in itertools.product(left_cols, right_cols)])
    indices = np.array([ij for ij in itertools.product(l_coli, r_coli)])

    res = _mat_stat_nb(np.asarray(data), indices, _numba_corr, min_n, method)

    """SAS and pearsonr look the statistic up in a t distribution while R uses the normnal"""
    res[:, 1] = 2 * stats.distributions.t.sf(np.abs(res[:, 1]), res[:, 2])
    # res[:, 1] = 2 * stats.norm.cdf(-np.abs(res[:, 1]))

    results = pd.DataFrame(res, columns=['rho', 'pvalue', 'N'])
    results = results.assign(Left=column_pairs[:, 0], Right=column_pairs[:, 1])
    return results

@numba.jit(nopython=True, parallel=False, error_model='numpy')
def _rankvector(v):
    n = len(v)
    rk = np.empty((n,))
    idx = np.argsort(v)
    rk[idx[:n]] = np.arange(1, n+1)
    return rk

@numba.jit(nopython=True, parallel=False, error_model='numpy')
def _numba_corr(v1, v2, method):
    x = np.empty((v1.shape[0], 2))
    if method == 'spearman':
        x[:, 0] = _rankvector(v1)
        x[:, 1] = _rankvector(v2)
    else:
        x[:, 0] = v1
        x[:, 1] = v2

    n_obs = x.shape[0]
    dof = n_obs - 2.
    if dof < 0:
        raise ValueError("The input must have at least 3 entries!")

    rs = np.corrcoef(x, rowvar=False)

    # clip the small negative values possibly caused by rounding
    # errors before taking the square root
    tmp = (dof / ((rs + 1.0) * (1.0 - rs)))
    for i in np.nonzero(tmp < 0):
        tmp[i] = 0
    t = rs * np.sqrt(tmp)
    return rs[1, 0], t[1, 0]

def _scipy_corr(v1, v2, method):
    if method == 'spearman':
        rho, pvalue = stats.spearmanr(v1, v2)
    elif method == 'pearson':
        rho, pvalue = stats.pearsonr(v1, v2)
    elif method == 'kendall':
        rho, pvalue = stats.kendalltau(v1, v2)
    return rho, pvalue

def crosscorr_scipy(data, left_cols=None, right_cols=None, method='spearman', min_n=5):
    if left_cols is None:
        left_cols = data.columns
    if right_cols is None:
        right_cols = left_cols
    
    columns = data.columns.tolist()
    l_coli = [columns.index(c) for c in left_cols]
    r_coli = [columns.index(c) for c in right_cols]
    
    column_pairs = np.array([ij for ij in itertools.product(left_cols, right_cols)])
    indices = np.array([ij for ij in itertools.product(l_coli, r_coli)])

    res = _mat_stat(np.asarray(data), indices, _scipy_corr, min_n, method)

    results = pd.DataFrame(res, columns=['rho', 'pvalue', 'N'])
    results = results.assign(Left=column_pairs[:, 0], Right=column_pairs[:, 1])
    return results

def test():
    """Add CIs"""
    from scipy import stats

    alphabet = np.array([a for a in 'ABCDEFGHIJKLMNOP'])
    np.random.seed(110820)
    dat = pd.DataFrame(np.random.rand(50, 20))
    dat.columns = [''.join(np.random.choice(alphabet, size=2)) for i in range(20)]

    for m in ['spearman', 'pearson']:
        res = crosscorr(dat, dat.columns, method=m)
        res_scipy = crosscorr_scipy(dat, dat.columns, method=m)
        
        if not np.allclose(res['rho'], res_scipy['rho']):
            print(np.abs(res['rho']-res_scipy['rho']).describe())
        if not np.allclose(res['pvalue'], res_scipy['pvalue']):
            print(np.abs(res['pvalue']-res_scipy['pvalue']).describe())
