import numpy as np
from scipy import stats
import numba
from numba import jit, prange

from roc_numba import roc_auc, twobytwo_jit, twobytwo_stats_arr_jit, twobytwo_stats_jit, predictor_stats, roc_stats_jit

__all__ = ['bootstrap_roc', 'bootstrap_twobytwo', 'bootstrap_auc']

@jit(nopython=True, parallel=True, error_model='numpy')
def bootstrap_twobytwo_jit(pred, obs, nstraps):
    n = pred.shape[0]

    a = np.zeros(nstraps)
    b = np.zeros(nstraps)
    c = np.zeros(nstraps)
    d = np.zeros(nstraps)
    for booti in prange(nstraps):
        rind = np.random.choice(np.arange(n), n)
        a[booti], b[booti], c[booti], d[booti] = twobytwo_jit(pred[rind], obs[rind])
    out = twobytwo_stats_arr_jit(a, b, c, d)

    """Sort each statistic independent, prep for bootstrap"""
    for k in out.keys():
        out[k].sort()
    return out

@jit(nopython=True, parallel=True, error_model='numpy')
def jackknife_twobytwo_jit(pred, obs):
    a, b, c, d = twobytwo_jit(pred, obs)
    ostat_d = twobytwo_stats_jit(a, b, c, d)

    n = pred.shape[0]

    a_vec = a * np.ones(n)
    b_vec = b * np.ones(n)
    c_vec = c * np.ones(n)
    d_vec = d * np.ones(n)
    for i in range(n):
        """Jackknife sample for pred/obs is subtracting 1
        from a, b, c, or d"""
        if pred[i] == 1 and obs[i] == 1:
            a_vec[i] = a - 1
        elif pred[i] == 1 and obs[i] == 0:
            b_vec[i] = b - 1
        elif pred[i] == 0 and obs[i] == 1:
            c_vec[i] = c - 1
        elif pred[i] == 0 and obs[i] == 0:
            d_vec[i] = d - 1
    jstats_d = twobytwo_stats_arr_jit(a_vec, b_vec, c_vec, d_vec)

    bca_accel_d = dict()
    for k in ostat_d.keys():
        jmean = np.nanmean(jstats_d[k])
        bca_accel_d[k] = np.nansum((jmean - jstats_d[k])**3) / (6.0 * np.nansum((jmean - jstats_d[k])**2)**1.5)
        """if k == 'Sensitivity':
            print(k, 'jmean', jmean, 'a', bca_accel_d[k])"""
    return bca_accel_d

def bootstrap_twobytwo(pred, obs, alpha=0.05, n_samples=10000, method='bca'):
    """Compute stats for a 2x2 table derived from
    observed and predicted data vectors.
    
    Returns two dict of parameters below: one contains point-estimates and one
    contains upper and lower confidence bounds estimated from bootstrap samples.

    Parameters
    ----------
    obs,pred : np.ndarray or pd.Series of shape (n,)
    alpha : float [0, 1]
        Specify CI: [alpha/2, 1-alpha/2]
    n_samples : int
        Number of bootstrap samples.
    method : str
        Specify bias-corrected and accelerated ("bca") or percentile ("pi")
        bootstrap.
    

    Returns
    -------
    sens : float
        Sensitivity (1 - false-negative rate)
    spec : float
        Specificity (1 - false-positive rate)
    ppv : float
        Positive predictive value (1 - false-discovery rate)
    npv : float
        Negative predictive value
    acc : float
        Accuracy
    OR : float
        Odds-ratio of the observed event in the two predicted groups.
    rr : float
        Relative rate of the observed event in the two predicted groups.
    nnt : float
        Number needed to treat, to prevent one case.
        (assuming all predicted positives were "treated")"""

    alphas = np.array([alpha/2, 1-alpha/2])
    stat_d = bootstrap_twobytwo_jit(pred, obs, nstraps=n_samples)

    # Percentile Interval Method
    if method == 'pi':
        avals = {k:alphas for k in stat_d.keys()}
    # Bias-Corrected Accelerated Method
    elif method == 'bca':
        # The value of the statistic function applied just to the actual data.
        a, b, c, d = twobytwo_jit(pred, obs)
        ostat_d = twobytwo_stats_jit(a, b, c, d)
        bca_accel_d = jackknife_twobytwo_jit(pred, obs)

        avals = dict()
        for k in ostat_d.keys():
            """The bias correction value"""
            z0 = stats.distributions.norm.ppf( (np.sum(stat_d[k] < ostat_d[k])) / np.sum(~np.isnan(stat_d[k])) )
            zs = z0 + stats.distributions.norm.ppf(alphas).reshape(alphas.shape + (1,) * z0.ndim)
            avals[k] = stats.distributions.norm.cdf(z0 + zs / (1 - bca_accel_d[k] * zs))

    ci_d = dict()
    for k in ostat_d.keys():
        if np.all(np.isnan(avals[k])):
            print('No bootstrap variation in %s: LCL = UCL = observed stat' % (k))
            ci_d[k] = ostat_d[k] * np.ones(len(alphas))
        else:
            non_nan_ind = ~np.isnan(stat_d[k])
            if np.any(np.isnan(avals[k])):
                print('Unhandled NaNs for %s, results also NaN' % (k))
                ci_d[k] = np.ones(len(avals[k])) * np.nan
            else:
                nvals = np.round((non_nan_ind.sum() - 1) * avals[k]).astype(int)
                if np.any(nvals < 10) or np.any(nvals > n_samples-10):
                    pass
                    print('Extreme samples (%s) used for %s, results unstable' % (nvals, k))
                ci_d[k] = stat_d[k][non_nan_ind][nvals]
        
    return ostat_d, ci_d

@jit(nopython=True, parallel=True, error_model='numpy')
def bootstrap_twobytwo_roc_jit(pred_continuous, obs, thresholds, nstraps):
    n = pred_continuous.shape[0]

    nthresh = len(thresholds)
    a = np.zeros(nstraps)
    b = np.zeros(nstraps)
    c = np.zeros(nstraps)
    d = np.zeros(nstraps)
    
    out = dict()
    for i in range(len(thresholds)):
        t = thresholds[i]
        pred = (pred_continuous >= t).astype(np.int_)
        for booti in prange(nstraps):
            rind = np.random.choice(np.arange(n), n)
            a[booti], b[booti], c[booti], d[booti] = twobytwo_jit(pred[rind], obs[rind])
        tmp = twobytwo_stats_arr_jit(a, b, c, d)

        """Sort each statistic independent, prep for bootstrap"""
        for k in tmp.keys():
            tmp[k].sort()
            if i == 0:
                out[k] = np.zeros((nthresh, nstraps))
            out[k][i, :] = tmp[k]
    return out

@jit(nopython=True, parallel=True, error_model='numpy')
def jackknife_twobytwo_roc_jit(pred_continuous, obs, thresholds):
    n = pred_continuous.shape[0]
    nthresh = len(thresholds)

    bca_accel_d = dict()
    for threshi in range(nthresh):
        t = thresholds[threshi]
        pred = (pred_continuous >= t).astype(np.int_)
        a, b, c, d = twobytwo_jit(pred, obs)
        ostat_d = twobytwo_stats_jit(a, b, c, d)

        a_vec = a * np.ones(n)
        b_vec = b * np.ones(n)
        c_vec = c * np.ones(n)
        d_vec = d * np.ones(n)
        for i in range(n):
            """Jackknife sample for pred/obs is subtracting 1
            from a, b, c, or d"""
            if pred[i] == 1 and obs[i] == 1:
                a_vec[i] = a - 1
            elif pred[i] == 1 and obs[i] == 0:
                b_vec[i] = b - 1
            elif pred[i] == 0 and obs[i] == 1:
                c_vec[i] = c - 1
            elif pred[i] == 0 and obs[i] == 0:
                d_vec[i] = d - 1
        jstats_d = twobytwo_stats_arr_jit(a_vec, b_vec, c_vec, d_vec)

        for k in ostat_d.keys():
            jmean = np.nanmean(jstats_d[k])
            if threshi == 0:
                bca_accel_d[k] = np.zeros(nthresh)
            bca_accel_d[k][threshi] = np.nansum((jmean - jstats_d[k])**3) / (6.0 * np.nansum((jmean - jstats_d[k])**2)**1.5)
            """if k == 'Sensitivity':
                print(k, 'jmean', jmean, 'a', bca_accel_d[k])"""
    return bca_accel_d

def bootstrap_roc(pred_continuous, obs, n_thresholds=50, alpha=0.05, n_samples=10000, method='bca'):
    """Compute ROC stats for a continuous predictor using n_thresholds
    from min(pred_continuous) to max(pred_continuous).
    
    Returns two dicts of the parameters below computed at every threshold:
        (1) point-estimates [n_thrsholds, 1]
        (2) upper and lower CL from bootstrap samples [n_thresholds, 2]

    Parameters
    ----------
    obs,pred : np.ndarray or pd.Series of shape (n,)
    alpha : float [0, 1]
        Specify CI: [alpha/2, 1-alpha/2]
    n_samples : int
        Number of bootstrap samples.
    method : str
        Specify bias-corrected and accelerated ("bca") or percentile ("pi")
        bootstrap.
    
    Returns
    -------
    sens : float
        Sensitivity (1 - false-negative rate)
    spec : float
        Specificity (1 - false-positive rate)
    ppv : float
        Positive predictive value (1 - false-discovery rate)
    npv : float
        Negative predictive value
    acc : float
        Accuracy
    OR : float
        Odds-ratio of the observed event in the two predicted groups.
    rr : float
        Relative rate of the observed event in the two predicted groups.
    nnt : float
        Number needed to treat, to prevent one case.
        (assuming all predicted positives were "treated")"""
    mn, mx = np.min(pred_continuous), np.max(pred_continuous)
    rng = mx - mn
    delta = rng / n_thresholds
    thresholds = np.linspace(mn + delta, mx - delta, n_thresholds - 1)

    alphas = np.array([alpha/2, 1-alpha/2])
    stat_d = bootstrap_twobytwo_roc_jit(pred_continuous, obs, thresholds, nstraps=n_samples)

    # Percentile Interval Method
    if method == 'pi':
        avals = {k:np.tile(alphas, (len(thresholds), 1)) for k in stat_d.keys()}
    # Bias-Corrected Accelerated Method
    elif method == 'bca':
        # The value of the statistic function applied just to the actual data.
        ostat_d, _ = roc_stats_jit(np.asarray(pred_continuous), np.asarray(obs), thresholds)
        bca_accel_d = jackknife_twobytwo_roc_jit(pred_continuous, obs, thresholds)

        avals = dict()
        for k in ostat_d.keys():
            avals[k] = np.zeros((len(thresholds), 2))
        for k in ostat_d.keys():
            for threshi in range(len(thresholds)):
                """The bias correction value"""
                z0 = stats.distributions.norm.ppf( (np.sum(stat_d[k][threshi, :] < ostat_d[k][threshi])) / np.sum(~np.isnan(stat_d[k][threshi, :])) )
                zs = z0 + stats.distributions.norm.ppf(alphas).reshape(alphas.shape + (1,) * z0.ndim)
                avals[k][threshi, :] = stats.distributions.norm.cdf(z0 + zs / (1 - bca_accel_d[k][threshi] * zs))
                """if k == 'Sensitivity':
                    print(k, ostat_d[k], z0, zs, avals)"""

    ci_d = dict()
    for k in ostat_d.keys():
        ci_d[k] = ostat_d[k][:, None] * np.ones((len(thresholds), len(alphas)))    

        for threshi in range(len(thresholds)):
            if np.all(np.isnan(avals[k][threshi, :])):
                print('No variation in stat %s, thresh %d (%1.2g): LCL = UCL = observed stat' % (k, threshi, thresholds[threshi]))
            else:
                non_nan_ind = ~np.isnan(stat_d[k][threshi, :])
                nvals = np.round((non_nan_ind.sum() - 1) * avals[k][threshi, :])

                if np.any(np.isnan(nvals)):
                    print('All nan samples for %s : %f, results are nan' % (k, thresholds[threshi]))
                    ci_d[k][threshi, :] = np.nan
                else:    
                    ci_d[k][threshi, :] = stat_d[k][threshi, non_nan_ind][nvals.astype(int)]

                if np.any(nvals < 10) or np.any(nvals > n_samples-10):
                    print('Extreme samples used for %s : %f, results unstable' % (k, thresholds[threshi]))
    return dict(ostat_d), ci_d


@jit(nopython=True, parallel=True, error_model='numpy')
def bootstrap_auc_jit(pred_continuous, obs, nstraps):
    n = pred_continuous.shape[0]
    auc = np.zeros(nstraps)
    for booti in prange(nstraps):
        rind = np.random.choice(np.arange(n), n)
        auc[booti] = roc_auc(obs[rind], pred_continuous[rind])
    auc.sort()        
    return auc

@jit(nopython=True, parallel=True, error_model='numpy')
def jackknife_auc_jit(pred_continuous, obs):
    oauc = roc_auc(obs, pred_continuous)
    n = pred_continuous.shape[0]

    jstats = np.zeros(n)
    #jind = np.ones(n, dtype=np.bool_)
    for i in prange(n):
        jind = np.ones(n, dtype=np.bool_)
        jind[i] = False
        jstats[i] = roc_auc(obs[jind], pred_continuous[jind])
        #jind[i] = True
    jmean = np.nanmean(jstats)
    bca_accel = np.nansum((jmean - jstats)**3) / (6.0 * np.nansum((jmean - jstats)**2)**1.5)
    return bca_accel

def bootstrap_auc(pred_continuous, obs, alpha=0.05, n_samples=10000, method='bca'):
    """Computes ROC AUC for a continuous predictor and provides bootstrap CI.
    
    Returns two dicts of the parameters below computed at every threshold:
        (1) point-estimates [n_thrsholds, 1]
        (2) upper and lower CL from bootstrap samples [n_thresholds, 2]

    Parameters
    ----------
    obs,pred : np.ndarray or pd.Series of shape (n,)
    alpha : float [0, 1]
        Specify CI: [alpha/2, 1-alpha/2]
    n_samples : int
        Number of bootstrap samples.
    method : str
        Specify bias-corrected and accelerated ("bca") or percentile ("pi")
        bootstrap.
    
    Returns
    -------
    auc : float
        Area under the receiver operator curve (AUC-ROC)
    ci : np.ndarray
        Lower and upper confidence limit obtained from the bootstrap samples."""

    alphas = np.array([alpha/2, 1-alpha/2])
    stat = bootstrap_auc_jit(pred_continuous, obs, nstraps=n_samples)

    # Percentile Interval Method
    if method == 'pi':
        avals = alphas
    # Bias-Corrected Accelerated Method
    elif method == 'bca':
        # The value of the statistic function applied just to the actual data.
        ostat = roc_auc(obs, pred_continuous)
        bca_accel = jackknife_auc_jit(pred_continuous, obs)

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


def _test_bca():
    from scikits.bootstrap import ci

    n = int(1000)
    np.random.seed(110820)
    pred = np.random.randint(2, size=n)
    obs = np.random.randint(2, size=n)

    def _sens_stat(pred, obs):
        return predictor_stats(pred, obs)['Sensitivity']
    with np.errstate(all='ignore'):
        lcl, ucl = ci((pred, obs), statfunction=_sens_stat)

    res = bootstrap_twobytwo(pred, obs)
    print(res)

def _bca_roc():
    n = int(100)
    pred_continuous = np.random.rand(n)
    obs = np.random.randint(2, size=n)

    res = bootstrap_roc(pred_continuous, obs, n_samples=100)
    auc, auc_ci = bootstrap_auc(pred_continuous, obs, n_samples=100)
    print(res)


