import argparse
import dask.array as da
import dask.dataframe as dd
import feather
import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smt
from dask.diagnostics import ProgressBar
from scipy.stats import norm

from dask.distributed import Client

"""USAGE:
python ~/gitrepo/utils/correlation_diff.py  PASC_rank_cor_mat.parquet  nonPASC_rank_cor_mat.parquet rank_zdiff.parquet rank_pval.parquet 52 104 --cores 6
"""

def compute_z_diff(corr_a, corr_b):
    z_a = np.arctanh(corr_a)
    z_b = np.arctanh(corr_b)
    #se = np.sqrt(1/(n_a-3) + 1/(n_b-3))
    zdiff = z_a - z_b
    return zdiff

def compute_pvalue(z_diff, n1, n2):
    se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
    p_diff = norm.sf(abs(z_diff/se_diff)) * 2
    #sh = p_diff.shape
    #adj_p_diff = smt.multipletests(p_diff.ravel(), alpha=0.05, method='fdr_bh')[1]
    return p_diff#.reshape(sh)

def main(file_a, file_b, out_zdiff, out_pvals, n1, n2, cores):
    a = dd.read_parquet(file_a, npartitions=20).to_dask_array()
    b = dd.read_parquet(file_b, npartitions=20).to_dask_array()
    #a = dd.from_pandas(feather.read_dataframe(file_a), npartitions=cores)
    #b = dd.from_pandas(feather.read_dataframe(file_b), npartitions=cores)
    #z_diffs = da.map_blocks(compute_z_diff, a.to_dask_array(), b.to_dask_array())

    with ProgressBar():
        z_diffs = (da.arctanh(a) - da.arctanh(b))
        dd.from_dask_array(z_diffs).to_parquet(out_zdiff)
        # np.savez(out_zdiff, z_diffs=z_diffs.compute())
    adj_p = da.map_blocks(compute_pvalue, z_diffs, n1, n2)
    with ProgressBar():
        dd.from_dask_array(adj_p).to_parquet(out_pvals)
        # np.savez(out_adj_pvals, adj_p=adj_p.compute())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Fisher\'s Z transformation and p-values for two sets of correlation coefficients.')
    parser.add_argument('file_a', type=str, help='File path for set A of correlation coefficients in parquet format.')
    parser.add_argument('file_b', type=str, help='File path for set B of correlation coefficients in parquet format.')
    parser.add_argument('out_zdiff', type=str, help='File path for output z-transformed differences in parquet format.')
    parser.add_argument('out_pvals', type=str, help='File path for output p-values in parquet format.')
    parser.add_argument('n1', type=int, help='Sample size for set A.')
    parser.add_argument('n2', type=int, help='Sample size for set B.')
    parser.add_argument('--cores', type=int, default=1, help='Number of CPU cores to use. Default is to use one core.')
    args = parser.parse_args()

    client = Client(n_workers=args.cores, threads_per_worker=1)

    main(args.file_a, args.file_b, args.out_zdiff, args.out_pvals, args.n1, args.n2, args.cores)
