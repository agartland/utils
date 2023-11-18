import numba
import numpy as np
import pandas as pd
import itertools
from scipy import stats

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

def _mat_stat_rho_only(mat, indices, stat_func, min_n, method):
    results = np.zeros(indices.shape[0])
    for pairi in numba.prange(indices.shape[0]):
        i, j = indices[pairi, :]
        notnan = (~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j]))
        n = np.sum(notnan)
        if n >= min_n:
            rho = stat_func(mat[notnan, i], mat[notnan, j], method)
        else:
            rho = np.nan
        results[pairi] = rho
    return results

_mat_stat_nb = numba.jit(_mat_stat, nopython=True, parallel=True, error_model='numpy')
_mat_stat_rho_only_nb = numba.jit(_mat_stat_rho_only, nopython=True, parallel=True, error_model='numpy')

def crosscorr(data, left_cols=None, right_cols=None, method='spearman', min_n=5, rho_only=False):
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
    sym = False
    if left_cols is None:
        left_cols = data.columns
        sym = True
    if right_cols is None:
        right_cols = left_cols
        sym = True
    
    columns = data.columns.tolist()
    l_coli = [columns.index(c) for c in left_cols]
    r_coli = [columns.index(c) for c in right_cols]
    
    column_pairs = np.array([ij for ij in itertools.product(left_cols, right_cols)])
    if sym:
        """Only compute the upper triangle of the matrix if left and right indices are identical.
        This can later be transformed into a square matrix using squareform which will assume zeros on the diagonal
        (though for correlations they should be 1's)"""
        indices = np.array([ij for ij in itertools.combinations(l_coli, 2)])
    else:
        indices = np.array([ij for ij in itertools.product(l_coli, r_coli)])

    if method == 'spearman-preranked':
        rdata = np.empty(data.values.shape)
        for i in set(l_coli + r_coli):
            idx = np.argsort(data.values[:, i])
            rdata[idx, i] = np.arange(1, data.values.shape[0] + 1)
        
        if rho_only:
            res = _mat_stat_rho_only_nb(rdata, indices, _numba_corr_only, min_n, 'spearman-preranked')
        else:
            res = _mat_stat_nb(rdata, indices, _numba_corr, min_n, 'spearman-preranked')
    else:
        if rho_only:
            res = _mat_stat_rho_only_nb(np.asarray(data), indices, _numba_corr_only, min_n, method)
        else:
            res = _mat_stat_nb(np.asarray(data), indices, _numba_corr, min_n, method)

    if rho_only:
        """Return the correlation estimates only in scipy condensed vector format"""
        return res 
    else:   
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

@numba.jit(nopython=True, parallel=False, error_model='numpy')
def _numba_corr_only(v1, v2, method):
    x = np.empty((v1.shape[0], 2))
    if method == 'spearman':
        x[:, 0] = _rankvector(v1)
        x[:, 1] = _rankvector(v2)
    else:
        x[:, 0] = v1
        x[:, 1] = v2

    n_obs = x.shape[0]

    rs = np.corrcoef(x, rowvar=False)

    return rs[1, 0]

def _scipy_corr(v1, v2, method):
    if method == 'spearman':
        rho, pvalue = stats.spearmanr(v1, v2)
    elif method == 'pearson':
        rho, pvalue = stats.pearsonr(v1, v2)
    elif method == 'kendall':
        rho, pvalue = stats.kendalltau(v1, v2)
    return rho, pvalue

def crosscorr_scipy(data, left_cols=None, right_cols=None, method='spearman', min_n=5, alpha=0.05):
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
    if method == 'pearson':
        r_z = np.arctanh(results['rho'])
        se = 1 / np.sqrt(results['N'] - 3)
        z = stats.norm.ppf(1 - alpha / 2)
        lo_z, hi_z = r_z-z*se, r_z+z*se
        lo, hi = np.tanh((lo_z, hi_z))
        results = results.assign(LCL=lo, UCL=hi)
    return results


def _square_stat(mat, stat_func, min_n, method):
    n = mat.shape[1]
    m = int((n * n - n) / 2)
    results = np.empty(m)
    pairi = 0
    """NOTE: this attempt at paralleization did not work. It gives incorrect values and doesn't throw an error"""
    # for i in numba.prange(n):
    for i in range(n):
        for j in range(n):
            if i < j:
                notnan = ~np.isnan(mat[:, i]) & ~np.isnan(mat[:, j])
                if np.sum(notnan) >= min_n:
                    rho = stat_func(mat[notnan, i], mat[notnan, j], method)
                else:
                    rho = np.nan
                results[pairi] = rho
                pairi += 1
    return results

_square_stat_nb = numba.jit(_square_stat, nopython=True, parallel=False, error_model='numpy')

def squarecorr(mat, method='spearman', min_n=5):
    """Compute correlations for all columns of numpy array mat, ignoring NaNs.
    Uses numba for speedup and parallelization.

    Parameters
    ----------
    mat : np.ndarray [m observations, n features]
    method : str
        Method can be spearman or pearson.
    min_n : int
        Minimum number of observation required to compute the correlation
        (otherwise NaN is returned)

    Returns
    -------
    rho : np.ndarray
        Scipy condensed vector representation of all pairwise correlations"""

    res = _square_stat_nb(mat, _numba_corr_only, min_n, method)
    return res

def test():
    """Add CIs"""
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
