import numpy as np
import pandas as pd

import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar

"""USELESS: I found dask.dataframe.corr which handles nans like pandas (but not rank corr!!)"""

def np_corrcoef(dat):
    """This should match exactly the output of np.ma.corrcoef when masking nans
    The cov part matches but something about stdev and after that adds an error"""
    
    """Subtract off the mean of each column (assumes features along the columns)"""
    X = dat - np.nanmean(dat, axis=0, keepdims=True)

    """Compute the correct N for each pairwise correlation using the nan mask"""
    xmask = np.isnan(X)
    xnotmask = (~xmask).astype(int)
    ddof = 1 #np default
    fact = np.dot(xnotmask.T, xnotmask).astype(float) - ddof

    """Fill nans with 0 before dot product"""
    Xfilled = X.copy()
    Xfilled[xmask] = 0

    d = np.dot(Xfilled.T, Xfilled) / fact

    """d is the covariance matrix and needs to be divided by the variances along the diagonal"""
    stdev = np.sqrt(np.diag(d))

    r = d / stdev[:, None]
    r /= stdev[None, :]
    return r

def da_corrcoef(dat):
    X = dat - da.nanmean(dat, axis=0, keepdims=True)

    xmask = da.isnan(X)
    xnotmask = (~xmask).astype(int)
    ddof = 1 #np default
    fact = da.dot(xnotmask.T, xnotmask).astype(float) - ddof
    
    X[xmask] = 0
    d = da.dot(X.T, X) / fact
    stdev = da.sqrt(da.diag(d))
    d /= stdev[:, None]
    d /= stdev[None, :]
    return d


def _test(n=10):
    dat = np.random.rand(52, n)
    dat[np.random.rand(52, n) > 0.9] = np.nan
    dat[:, :2] = np.random.rand(52, 2)
    #df = pd.DataFrame(dat)

    #res_pd = df.corr()

    #dat_ma = np.ma.masked_array(dat, np.isnan(dat))
    #res_np_ma = pd.DataFrame(np.ma.corrcoef(dat_ma, rowvar=False))
    #covar_np_ma = pd.DataFrame(np.ma.cov(dat_ma, rowvar=False))

    r_da = da_corrcoef(da.from_array(dat, chunks='auto'))
    #r_np = np_corrcoef(dat)
    return r_da


#client = Client(n_workers=2, threads_per_worker=1)
#res = _test(n=20000).compute()
#print(pd.DataFrame(res[:50, :50]))
#print(res.shape)