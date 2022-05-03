import numpy as np
import pandas as pd
from numba import jit, types
from numba.typed import Dict

__all__ = ['predictor_stats',
            'roc_stats',
            'twobytwo_stats']

@jit(nopython=True)
def twobytwo_jit(pred, obs):
    """Compute stats for a 2x2 table derived from
    observed and predicted data vectors

    Parameters
    ----------
    obs, pred : np.ndarray or pd.Series of shape (n,)

    Returns
    -------
    a : int
     True positives
    b : int
     False positives
    c : int
     False negatives
    d : True negatives"""
    n = obs.shape[0]
    a = np.sum(pred & obs)
    pred_sum = pred.sum()
    obs_sum = obs.sum()

    b = pred_sum - a
    c = obs_sum - a
    d = n - pred_sum - c
    return a, b, c, d

@jit(nopython=True, error_model='numpy')
def twobytwo_stats_jit(a, b, c, d):
    """
            OUTCOME
             +   -
           ---------
         + | a | b |
    PRED   |-------|
         - | c | d |
           ---------
    """
    n = a + b + c + d

    out = Dict.empty(key_type=types.unicode_type,
                     value_type=types.float64)

    out['Sensitivity'] = a / (a+c)
    out['Specificity'] = d / (b+d)
    out['PPV'] = a / (a+b)
    out['NPV'] = d / (c+d)
    out['NNT'] = 1 / (a/(a+b) - c/(c+d))
    out['ACC'] = (a + d)/n
    out['RR'] = (a / (a+b)) / (c / (c+d))
    out['OR'] = (a/c) / (b/d)
    out['PrevObs'] = (a + c) / n
    out['PrevPred'] = (a + b) / n
    out['RR_corrected'] = ((a + 0.5) / (a + b + 1)) / ((c + 0.5) / (c + d + 1))

    out['N'] = n
    out['A'] = a
    out['B'] = b
    out['C'] = c
    out['D'] = d
    return out

@jit(nopython=True, error_model='numpy', parallel=True)
def twobytwo_stats_arr_jit(a, b, c, d, correction=False):
    """
            OUTCOME
             +   -
           ---------
         + | a | b |
    PRED   |-------|
         - | c | d |
           ---------
    """
    n = a + b + c + d

    out = dict()
    ac = a + c
    bd = b + d
    ab = a + b
    cd = c + d
    
    out['Sensitivity'] = a / (ac)
    out['Specificity'] = d / (bd)
    out['PPV'] = a / (ab)
    out['NPV'] = d / (cd)
    out['NNT'] = 1 / (a/(ab) - c/(cd))
    out['OR'] = (a/c) / (b/d)
    out['ACC'] = (a + d)/n
    out['RR'] = (a / (ab)) / (c / (cd))
    out['PrevObs'] = (ac) / n
    out['PrevPred'] = (ab) / n
    out['RR_corrected'] = ((a + 0.5) / (ab + 1)) / ((c + 0.5) / (cd + 1))

    out['N'] = n.astype(np.float64)
    out['A'] = a.astype(np.float64)
    out['B'] = b.astype(np.float64)
    out['C'] = c.astype(np.float64)
    out['D'] = d.astype(np.float64)
    return out

@jit(nopython=True, parallel=True)
def roc_stats_jit(pred_continuous, obs, thresholds):
    nthresh = len(thresholds)
    a = np.zeros(nthresh)
    b = np.zeros(nthresh)
    c = np.zeros(nthresh)
    d = np.zeros(nthresh)
    
    for i, t in enumerate(thresholds):
        pred = (pred_continuous >= t).astype(np.int_)
        a[i], b[i], c[i], d[i] = twobytwo_jit(pred, obs)
    out = twobytwo_stats_arr_jit(a, b, c, d)
    auc = roc_auc(obs, pred_continuous)
    return out, auc

@jit(nopython=True, error_model='numpy')
def roc_auc(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

#@jit(nopython=True, error_model='numpy')
def roc_auc_np(y_true, y_prob):
    """NOTE: though this is a faster numpy implementation than the for-loop
    as a numba compiled function its a bit slower than the loop version and
    uses a bit more memory, so this is really only useful in a non-numba context.
    Importantly, its identical results to sklearn.metrics.roc_auc but 100sx faster."""
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    nfalse = np.cumsum(1 - y_true)
    auc = np.cumsum(y_true * nfalse)
    auc = auc[-1] / (nfalse[-1] * (n - nfalse[-1]))
    return auc

def predictor_stats(pred, obs):
    """Compute stats for a 2x2 table derived from
    observed and predicted data vectors.
    Returns a dict of parameters below.

    Parameters
    ----------
    obs,pred : np.ndarray or pd.Series of shape (n,)

    Optionally return a series with quantities labeled.

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

    assert obs.shape[0] == pred.shape[0]

    exclude = np.isnan(obs) | np.isnan(pred)

    a, b, c, d = twobytwo_jit(np.asarray(pred[~exclude]), np.asarray(obs[~exclude]))
    out = twobytwo_stats_jit(a, b, c, d)
    return dict(out)

def roc_stats(pred_continuous, obs, n_thresholds=50):
    """Compute ROC stats for a continuous predictor
    using n_thresholds from min(pred_continuous)
    to max(pred_continuous)

    Parameters
    ----------
    obs,pred : np.ndarray or pd.Series of shape (n,)

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

    assert obs.shape[0] == pred_continuous.shape[0]

    exclude = np.isnan(obs) | np.isnan(pred_continuous)

    mn, mx = np.nanmin(pred_continuous), np.nanmax(pred_continuous)
    rng = mx - mn
    delta = rng / n_thresholds
    thresholds = np.linspace(mn + delta, mx - delta, n_thresholds - 1)
    out, auc = roc_stats_jit(np.asarray(pred_continuous[~exclude]), np.asarray(obs[~exclude]), thresholds)

    out = pd.DataFrame(dict(out), index=thresholds)
    return out, auc

def twobytwo_stats(a, b, c, d):
    """Compute stats for many 2x2 tables:
    
            OUTCOME
             +   -
           ---------
         + | a | b |
    PRED   |-------|
         - | c | d |
           ---------
    

    Parameters
    ----------
    a, b, c, d : int
        Number of events in each bin.
        Will also work based on probabilities or
        vectors of counts or probabilities.
    
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
        (assuming all predicted positives were "treated")
    prevOut : float
        Marginal prevalence of the outcome.
    prevPred : float
        Marginal prevalence of the predictor."""
    n = a + b + c + d
    if np.isscalar(a):
        out = twobytwo_stats_jit(a, b, c, d)
        out = pd.Series(out)
    else:
        out = twobytwo_stats_arr_jit(a, b, c, d)
        out = pd.DataFrame(out)
    return out

def _test_2x2():
    n = int(1e7)
    pred = np.random.randint(2, size=n)
    obs = np.random.randint(2, size=n)
    print(predictor_stats(pred, obs))

def _test_2x2_stats():
    out = twobytwo_stats_jit(45, 70, 30, 1000)
    print(out)

def _test_2x2_stats_arr():
    out = twobytwo_stats_arr_jit(np.array([40,45]), np.array([70,70]), np.array([20,30]), np.array([500,1000]))
    print(out)

def _test_roc():
    from sklearn.metrics import roc_auc_score
    from scipy import stats
    n = int(100)
    pred_continuous = np.random.rand(n)
    obs = np.random.randint(2, size=n)
    # out, auc = roc_stats(pred_continuous, obs, n_thresholds=50)
    auc = roc_auc(obs, pred_continuous)
    auc_np = roc_auc_np(obs, pred_continuous)
    auc_sk = roc_auc_score(obs, pred_continuous)
    print(auc, auc_np, auc_sk)
    """Also close, but not exactly"""
    auc_mwu = stats.mannwhitneyu(obs, pred_continuous).statistic / y.shape[0]**2