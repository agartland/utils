import numpy as np
from scipy import stats
import numba
from numba import jit

@jit(nopython=True, parallel=True, error_model='numpy')
def bootstrap_twobytwo_jit(pred, obs, nstraps):
    n = pred.shape[0]

    a = np.zeros(nstraps)
    b = np.zeros(nstraps)
    c = np.zeros(nstraps)
    d = np.zeros(nstraps)
    for booti in range(nstraps):
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
        if k == 'N':
            ci_d[k] = ostat_d[k] * np.ones(len(alphas))
        else:
            non_nan_ind = ~np.isnan(stat_d[k])
            nvals = np.round((non_nan_ind.sum() - 1) * avals[k]).astype(int)
            ci_d[k] = stat_d[k][non_nan_ind][nvals]
    
        if np.any(nvals < 10) or np.any(nvals > n_samples-10):
            print('Extreme samples used for %s, results unstable' % k)
    return ci_d

@jit(nopython=True, parallel=True, error_model='numpy')
def bootstrap_twobytwo_roc_jit(pred_continuous, obs, thresholds, nstraps):
    n = pred_continuous.shape[0]

    nthresh = len(thresholds)
    a = np.zeros(nstraps)
    b = np.zeros(nstraps)
    c = np.zeros(nstraps)
    d = np.zeros(nstraps)
    
    out = dict()
    for i, t in enumerate(thresholds):
        pred = (pred_continuous >= t).astype(np.int_)
        for booti in range(nstraps):
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
    for threshi, t in enumerate(thresholds):
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
        if k == 'N':
            ci_d[k] = ostat_d[k][:, None] * np.ones((len(thresholds), len(alphas)))
        else:
            ci_d[k] = np.zeros((len(thresholds), len(alphas)))
            for threshi in range(len(thresholds)):
                non_nan_ind = ~np.isnan(stat_d[k][threshi, :])
                nvals = np.round((non_nan_ind.sum() - 1) * avals[k][threshi, :])

                if np.any(np.isnan(nvals)):
                    print('All nan samples for %s : %f, results are nan' % (k, thresholds[threshi]))
                    ci_d[k][threshi, :] = np.nan
                else:    
                    ci_d[k][threshi, :] = stat_d[k][threshi, non_nan_ind][nvals.astype(int)]
    
                if np.any(nvals < 10) or np.any(nvals > n_samples-10):
                    print('Extreme samples used for %s : %f, results unstable' % (k, thresholds[threshi]))
    return ci_d



@jit(nopython=True, parallel=True, error_model='numpy')
def bootstrap_auc_jit(pred_continuous, obs, nstraps):
    n = pred.shape[0]

    auc = np.zeros(nstraps)
    for booti in range(nstraps):
        rind = np.random.choice(np.arange(n), n)
        auc[booti] = roc_auc(obs[rind], pred_continuous[rind])
    auc.sort()        
    return auc

@jit(nopython=True, parallel=True, error_model='numpy')
def jackknife_auc_jit(pred_continuous, obs):
    oauc = roc_auc(obs, pred_continuous)
    n = pred.shape[0]

    jstats = np.zeros(n)
    jind = np.ones(n, dtype=np.bool_)
    for i in range(n):
        jind[i] = False
        jstats[i] = roc_auc(obs[jind], pred_continuous[jind])
        jind[i] = True
    jmean = np.nanmean(jstats)
    bca_accel = np.nansum((jmean - jstats)**3) / (6.0 * np.nansum((jmean - jstats)**2)**1.5)
    return bca_accel

def bootstrap_auc(pred_continuous, obs, alpha=0.05, n_samples=10000, method='bca'):
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
    n = int(100)
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


