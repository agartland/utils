import numpy as np
import pandas as pd
from numba import jit

def _twobytwo(dat, col1, col2):
    a = np.sum(dat[col1] & dat[col2])
    b = np.sum(dat[col1] & ~dat[col2])
    c = np.sum(~dat[col1] & dat[col2])
    d = np.sum(~dat[col1] & ~dat[col2])
    return a, b, c, d

@jit(nopython=True, parallel=True)
def bootstrap_twobytwo_jit(pred, obs, nstraps):
    pass

@jit(nopython=True, parallel=True)
def bootstrap_roc_jit(pred_continuous, obs, nstraps):
    pass

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

@jit(nopython=True, parallel=True)
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
    sens = a / (a+c)
    spec = d / (b+d)
    ppv = a / (a+b)
    npv = d / (c+d)
    nnt = 1 / (a/(a+b) - c/(c+d))
    acc = (a + d)/n
    rr = (a / (a+b)) / (c / (c+d))
    OR = (a/c) / (b/d)
    prevOut = (a + c) / n
    prevPred = (a + b) / n
    return sens, spec, ppv, npv, nnt, acc, rr, OR, prevOut, prevPred

@jit(nopython=True, parallel=True)
def roc_stats_jit(pred_continuous, obs, thresholds):
    """TODO: compute AUC using all values in pred_continuous as thresholds"""
    nthresh = len(thresholds)
    a = np.zeros(nthresh)
    b = np.zeros(nthresh)
    c = np.zeros(nthresh)
    d = np.zeros(nthresh)
    
    for i, t in enumerate(thresholds):
        pred = (pred_continuous >= t).astype(int)
        a[i], b[i], c[i], d[i] = twobytwo_jit(pred, obs)
    sens, spec, ppv, npv, nnt, acc, rr, OR, prevOut, prevPred = twobytwo_stats_jit(a, b, c, d)
    return sens, spec, ppv, npv, nnt, acc, rr, OR, prevOut, prevPred

def predictor_stats(pred, obs):
    """Compute stats for a 2x2 table derived from
    observed and predicted data vectors

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

    a, b, c, d = twobytwo_jit(np.asarray(pred), np.asarray(obs))

    sens, spec, ppv, npv, nnt, acc, rr, OR, prevOut, prevPred = twobytwo_stats_jit(a, b, c, d)

    vec = [sens, spec, ppv, npv, nnt, acc, rr, OR, prevOut, prevPred, a, b, c, d, n]
    labels = ['Sensitivity', 'Specificity',
                'PPV', 'NPV', 'NNT',
                'ACC', 'RR', 'OR',
                'prevOut', 'prevPred',
                'A', 'B', 'C', 'D', 'N']

    out = pd.Series(vec, index=labels)
    return out

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

    assert obs.shape[0] == pred.shape[0]

    mn, mx = np.min(pred_continuous), np.max(pred_continuous)
    rng = mx - mn
    delta = rng / n_thresholds
    thresholds = np.linspace(mn + delta, mx - delta, n_thresholds - 1)
    sens, spec, ppv, npv, nnt, acc, rr, OR, prevOut, prevPred = roc_stats_jit(np.asarray(pred_continuous), np.assarray(obs), thresholds)

    vec = [sens, spec, ppv, npv, nnt, acc, rr, OR, prevOut, prevPred, a, b, c, d, n]
    labels = ['Sensitivity', 'Specificity',
                'PPV', 'NPV', 'NNT',
                'ACC', 'RR', 'OR',
                'prevOut', 'prevPred',
                'A', 'B', 'C', 'D', 'N']

    out = pd.DataFrame({k:v for k,v in zip(labels, vec)}, index=thresholds)
    return out

def twobytwo_stats(a, b, c, d):
    """Compute stats for a 2x2 table:
    
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
    sens, spec, ppv, npv, nnt, acc, rr, OR, prevOut, prevPred = twobytwo_stats_jit(a, b, c, d)

    vec = [sens, spec, ppv, npv, nnt, acc, rr, OR, prevOut, prevPred, a, b, c, d, n]
    labels = ['Sensitivity', 'Specificity',
                'PPV', 'NPV', 'NNT',
                'ACC', 'RR', 'OR',
                'prevOut', 'prevPred',
                'A', 'B', 'C', 'D', 'N']
    if np.isscalar(a):
        out = pd.Series(vec, index=labels)
    else:
        out = pd.DataFrame({k:v for k,v in zip(labels, vec)})
    return out

def _test_2x2():
     n = int(1e8)
     dat = pd.DataFrame({'COR':np.random.randint(2, size=n).astype(bool),
                         'endpoint':np.random.randint(2, size=n).astype(bool)})
     a, b, c, d = twobytwo2(dat, 'COR', 'endpoint')
