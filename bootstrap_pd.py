import numpy as np
import pandas as pd


def bootstrap_pd(df, statfunction, alpha=0.05, n_samples=10000, method='bca'):
    alphas = np.array([alpha/2, 1-alpha/2])
    
    boot_res = []
    for i in range(n_samples):
        boot_res.append(statfunction(df.sample(frac=1, replace=True)))
    boot_res = pd.DataFrame(boot_res)

    # Percentile Interval Method
    if method == 'pi':
        avals = alphas
    # Bias-Corrected Accelerated Method
    elif method == 'bca':
        # The value of the statistic function applied just to the actual data.
        res = statfunction(df)

        ind = np.ones(df.shape[0], dtype=bool)
        jack_res = []
        for i in df.shape[0]:
            ind[i] = False
            jack_res.append(statfunction(df.loc[ind]))
            ind[i] = True

        jack_res = pd.DataFrame(jack_res)
        jmean = jack_res.mean()
        #bca_accel = np.nansum((jmean - jstats)**3) / (6.0 * np.nansum((jmean - jstats)**2)**1.5)
        

        """The bias correction value"""
        z0 = stats.distributions.norm.ppf( (np.sum(stat < ostat)) / np.sum(~np.isnan(stat)) )
        zs = z0 + stats.distributions.norm.ppf(alphas).reshape(alphas.shape + (1,) * z0.ndim)
        avals = stats.distributions.norm.cdf(z0 + zs / (1 - bca_accel * zs))


    non_nan_ind = ~np.isnan(stat)
    nvals = np.round((non_nan_ind.sum() - 1) * avals).astype(int)
    auc_ci = stat[non_nan_ind][nvals]
    
    if np.any(nvals < 10) or np.any(nvals > n_samples-10):
        print('Extreme samples used for AUC, results unstable')
    return ostat, auc_ci