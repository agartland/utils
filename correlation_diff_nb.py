import argparse
import numpy as np
import pandas as pd
import joblib 
import statsmodels.stats.multitest as smt
from scipy.stats import norm

def compute_z_diff(corr_a, corr_b):
    z_a = np.arctanh(corr_a)
    z_b = np.arctanh(corr_b)
    #se = np.sqrt(1/(n_a-3) + 1/(n_b-3))
    zdiff = z_a - z_b
    return zdiff

def compute_pvalue(z_diff, n1, n2):
    se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
    p_diff = norm.sf(abs(z_diff/se_diff)) * 2
    # adj_p_diff = smt.multipletests(p_diff.ravel(), alpha=0.05, method='fdr_bh')[1]
    return adj_p_diff

def main(file_a, file_b, out_zdiff, out_adj_pvals, n1, n2, cores):
    a = pd.read_parquet(file_a).values
    b = pd.read_parquet(file_b).values

    sh = a.shape

    a = a.ravel()
    b = b.ravel()

    chunksz = max(len(a) // cores, 1)
    remainder = len(a) % cores
    idx = [0] + [chunksz * i + min(i, remainder) for i in range(1, cores)] + [len(a)]

    with joblib.Parallel(n_jobs=args.cores) as parallel:
        z_diffs = parallel(joblib.delayed(compute_z_diff)(a[idx[i]:idx[i+1]], b[idx[i]:idx[i+1]])
                           for i in range(cores))
    
    z_diffs = np.concatenate(z_diffs)
    np.savez(out_zdiff, z_diffs=z_diffs.reshape(sh))

    with joblib.Parallel(n_jobs=args.cores) as parallel:
        pvalues = parallel(joblib.delayed(compute_pvalue)(z_diffs[idx[i]:idx[i+1]], args.sample_size_a, args.sample_size_b)
                           for i in range(cores))
    
    pvalues = np.concatenate(pvalues)
    np.savez(out_adj_pvals, pvalues=pvalues.reshape(sh))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute Fisher\'s Z transformation and adjusted p-values for two sets of correlation coefficients.')
    parser.add_argument('file_a', type=str, help='File path for set A of correlation coefficients in feather format.')
    parser.add_argument('file_b', type=str, help='File path for set B of correlation coefficients in feather format.')
    parser.add_argument('out_zdiff', type=str, help='File path for output z-transformed differences in feather format.')
    parser.add_argument('out_adj_pvals', type=str, help='File path for output adjusted p-values in feather format.')
    parser.add_argument('n1', type=int, help='Sample size for set A.')
    parser.add_argument('n2', type=int, help='Sample size for set B.')
    parser.add_argument('--cores', type=int, default=1, help='Number of CPU cores to use. Default is to use one core.')
    args = parser.parse_args()
    main(args.file_a, args.file_b, args.out_zdiff, args.out_adj_pvals, args.n1, args.n2, args.cores)
