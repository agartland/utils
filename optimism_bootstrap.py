import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.svm import SVC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from roc import *
from powercalc import *

__all__ = ['optimism_bootstrap',
           'percentile_bootstrap',
           'ob_roc_curve',
           'pb_roc_curve',
           'noboot_roc_curve',
           'no_boot_metric',
           'plotBootROC']

def optimism_bootstrap(X, y, model, fitMethod, predictMethod, metric, alpha=0.05, nstraps=1000, ignoreNan=True):
    """Compute optimism adjusted bootstrap of the performance metric for the classification model.

    "Multivariable prognostic models: issues in developing models,
    evaluating assumptions and adequacy, and measuring and reducing errors"
    Harrell FE Jr, Lee KL, Mark DB
    Stat Med. 1996 Feb 28;15(4):361-87.

    Parameters
    ----------
    X : np.ndarray [nsamples, nfeatures]
    y : np.ndarray [nsamples, ]
    model : class object
        Instantiation of a sklearn model with fit and predict_proba-like methods
    fitMethod : str
        Name of the method that fits the model with X and y
    predictMethod : str
        Name of the method that makes a continuous prediction of y
        Likely is predict_probability or decision_function, but predict will not work
    metric : function
        Any classification metric that takes y and y_score as inputs
    alpha : float
        For computing 1 - alpha % confidence intervals
    nstraps : int
        Number of bootstrap samples.
    ignoreNan : bool
        Discard Nan values before computing percentiles,
        otherwise a Nan will lead to Nan CI bounds.

    Returns
    -------
    estimate, lb, ub : floats
        Optimism adjusted estimate, lower bound and upper bound on the metric.

    Example
    -------
    auc, lb, ub = optimism_bootstrap(X, y, SVC(), 'fit', 'decision_function', sklearn.metrics.roc_auc_score)"""
    
    n = len(y)
    res = getattr(model, fitMethod)(X, y)
    y_score = getattr(res, predictMethod)(X)
    Capp = metric(y, y_score)

    Cboot = np.zeros(nstraps)
    Corig = np.zeros(nstraps)
    for i in range(nstraps):
        rind = np.random.randint(n, size=n)
        res = getattr(model, fitMethod)(X[rind, :], y[rind])
        y_score = getattr(res, predictMethod)(X[rind, :])
        Cboot[i] = metric(y[rind], y_score)

        y_score = getattr(res, predictMethod)(X)
        Corig[i] = metric(y, y_score)
    
    adjustedC = Capp - (Cboot - Corig)
    if ignoreNan:
        lb, ub = np.nanpercentile(adjustedC, [100 * alpha/2, 100 * (1 - alpha/2)])
    else:
        lb, ub = np.percentile(adjustedC, [100 * alpha/2, 100 * (1 - alpha/2)])

    return np.median(adjustedC), lb, ub

def percentile_bootstrap_metric(y, metric, X=None, y_score=None, model=None, fitMethod='', predictMethod='', alpha=0.05, nstraps=1000, ignoreNan=True):
    """Compute percentile bootstrap of the performance metric for the classification model.

    Parameters
    ----------
    X : np.ndarray [nsamples, nfeatures]
    y : np.ndarray [nsamples, ]
    y_score : np.ndarray [nsamples, ]
    model : class object
        Instantiation of a sklearn model with fit and predict_proba-like methods
    fitMethod : str
        Name of the method that fits the model with X and y
    predictMethod : str
        Name of the method that makes a continuous prediction of y
        Likely is predict_probability or decision_function, but predict will not work
    metric : function
        Any classification metric that takes y and y_score as inputs
    alpha : float
        For computing 1 - alpha % confidence intervals
    nstraps : int
        Number of bootstrap samples.
    ignoreNan : bool
        Discard Nan values before computing percentiles,
        otherwise a Nan will lead to Nan CI bounds.

    Returns
    -------
    estimate, lb, ub : floats
        Estimate, lower bound and upper bound on the metric.

    Example
    -------
    auc, lb, ub = percentile_bootstrap(X, y, SVC(), 'fit', 'decision_function', sklearn.metrics.roc_auc_score)"""
    
    n = len(y)
    if not model is None:
        res = getattr(model, fitMethod)(X, y)
        y_score = getattr(res, predictMethod)(X)
    C = metric(y, y_score)

    Cboot = np.nan * np.zeros(nstraps)
    for i in range(nstraps):
        rind = np.random.randint(n, size=n)
        try:
            if not model is None:
                res = getattr(model, fitMethod)(X[rind, :], y[rind])
                y_score = getattr(res, predictMethod)(X[rind, :])
                Cboot[i] = metric(y[rind], y_score)
            else:
                Cboot[i] = metric(y[rind], y_score[rind])
        except ValueError:
            Cboot[i] = np.nan
        
    if ignoreNan:
        lb, ub = np.nanpercentile(Cboot, [100 * alpha/2, 100 * (1 - alpha/2)])
    else:
        lb, ub = np.percentile(Cboot, [100 * alpha/2, 100 * (1 - alpha/2)])
    return C, lb, ub

def noboot_metric(y, metric, X=None, y_score=None, model=None, fitMethod='', predictMethod='', alpha=0.05, ignoreNan=True):
    """Compute percentile bootstrap of the performance metric for the classification model.

    Parameters
    ----------
    X : np.ndarray [nsamples, nfeatures]
    y : np.ndarray [nsamples, ]
    y_score : np.ndarray [nsamples, ]
    model : class object
        Instantiation of a sklearn model with fit and predict_proba-like methods
    fitMethod : str
        Name of the method that fits the model with X and y
    predictMethod : str
        Name of the method that makes a continuous prediction of y
        Likely is predict_probability or decision_function, but predict will not work
    metric : function
        Any classification metric that takes y and y_score as inputs
        Metric must return (est, lower-bound CI, upper-bound CI)
    alpha : float
        For computing 1 - alpha % confidence intervals
    nstraps : int
        Number of bootstrap samples.
    ignoreNan : bool
        Discard Nan values before computing percentiles,
        otherwise a Nan will lead to Nan CI bounds.

    Returns
    -------
    resDf : pd.DataFrame
        Estimate, lower bound and upper bound of the metric.
        Columns: est, lb, ub, thresholds"""
    
    pos_label = np.max(y)
    n = len(y)
    if not model is None:
        res = getattr(model, fitMethod)(X, y)
        y_score = getattr(res, predictMethod)(X)
    # thresholds = np.linspace(np.max(y_score), np.min(y_score), 100)
    thresholds = np.unique(y_score)[::-1]

    thresholds = np.concatenate(([thresholds[0] + (thresholds[0] - thresholds[1])],
                                  thresholds,
                                  [thresholds[-1] - (thresholds[-2] - thresholds[-1])]))
    
    est = np.zeros(len(thresholds))
    lb = np.zeros(len(thresholds))
    ub = np.zeros(len(thresholds))
    for threshi, thresh in enumerate(thresholds):
        y_pred = (y_score >= thresh).astype(int)
        est[threshi], lb[threshi], ub[threshi] = metric(y, y_pred)

    outDf = pd.DataFrame({'est':est,
                          'lb':lb,
                          'ub':ub,
                          'threshold':thresholds})
    return outDf

def ob_roc_curve(X, y, model, fitMethod, predictMethod, alpha=0.05, nstraps=1000, ignoreNan=True):
    """Compute optimism adjusted bootstrap of the ROC curve for the classification model.

    "Multivariable prognostic models: issues in developing models,
    evaluating assumptions and adequacy, and measuring and reducing errors"
    Harrell FE Jr, Lee KL, Mark DB
    Stat Med. 1996 Feb 28;15(4):361-87.

    Parameters
    ----------
    X : np.ndarray [nsamples, nfeatures]
    y : np.ndarray [nsamples, ]
    model : class object
        Instantiation of a sklearn model with fit and predict_proba-like methods
    fitMethod : str
        Name of the method that fits the model with X and y
    predictMethod : str
        Name of the method that makes a continuous prediction of y
        Likely is predict_probability or decision_function, but predict will not work
    alpha : float
        For computing 1 - alpha % confidence intervals
    nstraps : int
        Number of bootstrap samples.
    ignoreNan : bool
        Discard Nan values before computing percentiles,
        otherwise a Nan will lead to Nan CI bounds.

    Returns
    -------
    rocDf : pd.DataFrame
        Optimism-adjusted estimate, lower bound and upper bound of the ROC curve.
        Columns: fpr_est, tpr_est, fpr_lb, fpr_ub, thresholds"""
    pos_label = np.max(y)
    n = len(y)
    res = getattr(model, fitMethod)(X, y)
    y_score = getattr(res, predictMethod)(X)
    fpr_app, tpr_app, thresholds = roc_curve(y, y_score, pos_label=pos_label)
    fpr_app = np.concatenate(([0], fpr_app, [1]))
    tpr_app = np.concatenate(([0], tpr_app, [1]))
    thresholds = np.concatenate((thresholds[:1], thresholds, thresholds[-1:]))
    nthresholds = len(tpr_app)

    fpr_boot = np.zeros((nstraps, nthresholds))
    fpr_orig = np.zeros((nstraps, nthresholds))
    tpr_boot = np.zeros((nstraps, nthresholds))
    tpr_orig = np.zeros((nstraps, nthresholds))
    for i in range(nstraps):
        rind = np.random.randint(n, size=n)
        res = getattr(model, fitMethod)(X[rind, :], y[rind])
        y_score = getattr(res, predictMethod)(X[rind, :])
        fpr_tmp, tpr_tmp, thresh = roc_curve(y[rind], y_score)

        tpr_boot[i, :] = np.interp(fpr_app, fpr_tmp, tpr_tmp)
        fpr_boot[i, :] = np.interp(tpr_app, tpr_tmp, fpr_tmp)

        y_score = getattr(res, predictMethod)(X)
        fpr_tmp, tpr_tmp, thresh = roc_curve(y, y_score)

        tpr_orig[i, :] = np.interp(fpr_app, fpr_tmp, tpr_tmp)
        fpr_orig[i, :] = np.interp(tpr_app, tpr_tmp, fpr_tmp)

    
    fpr_adj = fpr_app[None, :] - (fpr_boot - fpr_orig)
    tpr_adj = tpr_app[None, :] - (tpr_boot - tpr_orig)
    if ignoreNan:
        fpr_est, fpr_lb, fpr_ub = np.nanpercentile(fpr_adj, [50, 100 * alpha/2, 100 * (1 - alpha/2)], axis=0)
        tpr_est, tpr_lb, tpr_ub = np.nanpercentile(tpr_adj, [50, 100 * alpha/2, 100 * (1 - alpha/2)], axis=0)
    else:
        fpr_est, fpr_lb, fpr_ub = np.percentile(fpr_adj, [50, 100 * alpha/2, 100 * (1 - alpha/2)], axis=0)
        tpr_est, tpr_lb, tpr_ub = np.percentile(tpr_adj, [50, 100 * alpha/2, 100 * (1 - alpha/2)], axis=0)

    fpr_est[0], fpr_lb[0], fpr_ub[0] = 0, 0, 0
    fpr_est[-1], fpr_lb[-1], fpr_ub[-1] = 1, 1, 1

    tpr_est[0], tpr_lb[0], tpr_ub[0] = 0, 0, 0
    tpr_est[-1], tpr_lb[-1], tpr_ub[-1] = 1, 1, 1
    outDf = pd.DataFrame({'fpr_est':np.clip(fpr_est, 0, 1),
                          'fpr_lb':np.clip(fpr_lb, 0, 1),
                          'fpr_ub':np.clip(fpr_ub, 0, 1),
                          'tpr_est':np.clip(tpr_app, 0, 1),
                          'tpr_lb':tpr_lb,
                          'tpr_ub':tpr_ub,
                          'trheshold':thresholds})
    return outDf

def pb_roc_curve(X=None, y=None, y_score=None, model=None, fitMethod='', predictMethod='', alpha=0.05, nstraps=1000, ignoreNan=True):
    """Compute percentile bootstrap of the ROC curve for the classification model.

    Examples
    --------
    resDf = pb_roc_curve(y=y_actual, y_score=y_predicted)

    resDf = pb_roc_curve(X=gram_matrix, y=y_actual,
                             model=SVC(kernel='precomputed'),
                             fitMethod='fit', predictMethod='decision_function', nstraps=5000)

    Parameters
    ----------
    X : np.ndarray [nsamples, nfeatures]
    y : np.ndarray [nsamples, ]
    y_score : np.ndarray [nsamples, ]
        Optionally provide pre-computed y_scores instead of a model to fit.
    model : class object
        Instantiation of a sklearn model with fit and predict_proba-like methods
    fitMethod : str
        Name of the method that fits the model with X and y
    predictMethod : str
        Name of the method that makes a continuous prediction of y
        Likely is predict_probability or decision_function, but predict will not work
    alpha : float
        For computing 1 - alpha % confidence intervals
    nstraps : int
        Number of bootstrap samples.

    Returns
    -------
    rocDf : pd.DataFrame
        Estimate, lower bound and upper bound of the ROC curve.
        Columns: fpr_est, tpr_est, fpr_lb, fpr_ub, thresholds"""
    pos_label = np.max(y)
    n = len(y)
    if not model is None:
        res = getattr(model, fitMethod)(X, y)
        y_score = getattr(res, predictMethod)(X)
    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=pos_label)
    fpr = np.concatenate(([0], fpr, [1]))
    tpr = np.concatenate(([0], tpr, [1]))
    thresholds = np.concatenate((thresholds[:1], thresholds, thresholds[-1:]))
    nthresholds = len(tpr)

    tpr_boot = np.zeros((nstraps, nthresholds))
    fpr_boot = np.zeros((nstraps, nthresholds))
    for i in range(nstraps):
        rind = np.random.randint(n, size=n)
        if not model is None:
            res = getattr(model, fitMethod)(X[rind, :], y[rind])
            y_score = getattr(res, predictMethod)(X[rind, :])
            fpr_tmp, tpr_tmp, thresh = roc_curve(y[rind], y_score)
        else:
            fpr_tmp, tpr_tmp, thresh = roc_curve(y[rind], y_score[rind])
        
        tpr_boot[i, :] = np.interp(fpr, fpr_tmp, tpr_tmp)
        fpr_boot[i, :] = np.interp(tpr, tpr_tmp, fpr_tmp)
    
    if ignoreNan:
        tpr_lb, tpr_ub = np.nanpercentile(tpr_boot, [100 * alpha/2, 100 * (1 - alpha/2)], axis=0)
        fpr_lb, fpr_ub = np.nanpercentile(fpr_boot, [100 * alpha/2, 100 * (1 - alpha/2)], axis=0)
    else:
        tpr_lb, tpr_ub = np.percentile(tpr_boot, [100 * alpha/2, 100 * (1 - alpha/2)], axis=0)
        fpr_lb, fpr_ub = np.percentile(fpr_boot, [100 * alpha/2, 100 * (1 - alpha/2)], axis=0)

    #fpr_lb[0], fpr_ub[0] = 0, 0
    #fpr_lb[-1], fpr_ub[-1] = 1, 1
    outDf = pd.DataFrame({'fpr_est':fpr,
                          'tpr_est':tpr,
                          'tpr_lb':tpr_lb,
                          'tpr_ub':tpr_ub,
                          'fpr_lb':fpr_lb,
                          'fpr_ub':fpr_ub,
                          'threshold':thresholds})
    return outDf

def noboot_roc_curve(X=None, y=None, y_score=None, model=None, fitMethod='', predictMethod='', alpha=0.05, method='score'):
    """Compute ROC curve with confidence intervals using score test and other methods, for the classification model.

    Code in powercalc.py has been checked against R binom package. "Score" was derived
    from the Agresti paper and is equivalent to Wilson (copied from the R package).
    From the paper this seems to be the best in most situations.

    A. Agresti, B. A. Coull, T. A. Statistician, N. May,
    Approximate Is Better than "Exact" for Interval Estimation of Binomial Proportions,
    52, 119–126 (2007).

    Examples
    --------
    resDf = pb_roc_curve(y=y_actual, y_score=y_predicted)

    resDf = pb_roc_curve(X=gram_matrix, y=y_actual,
                             model=SVC(kernel='precomputed'),
                             fitMethod='fit', predictMethod='decision_function', nstraps=5000)

    Parameters
    ----------
    X : np.ndarray [nsamples, nfeatures]
    y : np.ndarray [nsamples, ]
    y_score : np.ndarray [nsamples, ]
        Optionally provide pre-computed y_scores instead of a model to fit.
    model : class object
        Instantiation of a sklearn model with fit and predict_proba-like methods
    fitMethod : str
        Name of the method that fits the model with X and y
    predictMethod : str
        Name of the method that makes a continuous prediction of y
        Likely is predict_probability or decision_function, but predict will not work
    alpha : float
        For computing 1 - alpha % confidence intervals
    method : str
        Method for CI: score, wilson, wald, agresti-coull, exact

    Returns
    -------
    rocDf : pd.DataFrame
        Estimate, lower bound and upper bound of the ROC curve.
        Columns: fpr_est, tpr_est, fpr_lb, fpr_ub, thresholds"""
    pos_label = np.max(y)
    n = len(y)
    if not model is None:
        res = getattr(model, fitMethod)(X, y)
        y_score = getattr(res, predictMethod)(X)
    # thresholds = np.linspace(np.max(y_score), np.min(y_score), 100)
    thresholds = np.unique(y_score)[::-1]

    thresholds = np.concatenate(([thresholds[0] + (thresholds[0] - thresholds[1])],
                                  thresholds,
                                  [thresholds[-1] - (thresholds[-2] - thresholds[-1])]))
    
    fpr = np.zeros(len(thresholds))
    tpr = np.zeros(len(thresholds))
    fpr_lb = np.zeros(len(thresholds))
    tpr_lb = np.zeros(len(thresholds))
    fpr_ub = np.zeros(len(thresholds))
    tpr_ub = np.zeros(len(thresholds))

    for threshi, thresh in enumerate(thresholds):
        y_pred = (y_score >= thresh).astype(int)
        a, b, c, d = compute2x2(y, y_pred)
        tpr[threshi], tpr_lb[threshi], tpr_ub[threshi] = sensitivityCI(a, b, c, d, alpha=alpha, method=method)
        fpr[threshi], fpr_lb[threshi], fpr_ub[threshi] = specificityCI(a, b, c, d, alpha=alpha, method=method)
    fpr = 1 - fpr
    fpr_lb = 1 - fpr_lb
    fpr_ub = 1 - fpr_ub

    #fpr_lb[0], fpr_ub[0] = 0, 0
    #fpr_lb[-1], fpr_ub[-1] = 1, 1
    outDf = pd.DataFrame({'fpr_est':fpr,
                          'tpr_est':tpr,
                          'tpr_lb':tpr_lb,
                          'tpr_ub':tpr_ub,
                          'fpr_lb':fpr_lb,
                          'fpr_ub':fpr_ub,
                          'threshold':thresholds})
    return outDf

def plotBootROC(rocDfL, labelL=None, aucL=None, ciParam='fpr'):
    """Plot of ROC curves with confidence intervals.

    Parameters
    ----------
    rocDfL : list of pd.DataFrames
        Each DataFram is one model and must include columns
        fpr_est, tpr_est, fpr_lb, fpr_ub
    labelL : list of str
        Names of each model for legend
    aucL : list of floats
        AUC scores of each model for legend"""
    if labelL is None and aucL is None:
        labelL = ['Model %d' % i for i in range(len(rocDfL))]
    elif labelL is None:
        labelL = ['Model %d (AUC = %0.2f [%0.2f, %0.2f])' % (i, auc[0], auc[1], auc[2]) for i, auc in enumerate(aucL)]
    else:
        labelL = ['%s (AUC = %0.2f [%0.2f, %0.2f])' % (label, auc[0], auc[1], auc[2]) for label, auc in zip(labelL, aucL)]

    colors = sns.color_palette('Set1', n_colors=len(rocDfL))

    plt.cla()
    plt.gca().set_aspect('equal')
    for i, (rocDf, label) in enumerate(zip(rocDfL, labelL)):
        if ciParam == 'fpr':
            plt.fill_betweenx(rocDf['tpr_est'], rocDf['fpr_lb'], rocDf['fpr_ub'], alpha=0.3, color=colors[i])
        elif ciParam == 'tpr':
            plt.fill_between(rocDf['fpr_est'], rocDf['tpr_lb'], rocDf['tpr_ub'], alpha=0.3, color=colors[i])
        plt.plot(rocDf['fpr_est'], rocDf['tpr_est'],'-', color=colors[i], lw=2)
        # plt.plot(rocDf['fpr_est'], rocDf['tpr_lb'], '.--', color=colors[i], lw=1)
        # plt.plot(rocDf['fpr_est'], rocDf['tpr_ub'], '.--', color=colors[i], lw=1)
        # plt.plot(rocDf['fpr_lb'], rocDf['tpr_est'], '--', color=colors[i], lw=1)
        # plt.plot(rocDf['fpr_ub'], rocDf['tpr_est'], '--', color=colors[i], lw=1)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend([plt.Line2D([0, 1], [0, 1], color=c, lw=2) for c in colors], labelL, loc='lower right', fontsize=10)
    plt.show()

def plotBootSensSpec(rocDfL, labelL=None, param='tpr'):
    """Plot of sensitivity curves with confidence intervals.

    Parameters
    ----------
    rocDfL : list of pd.DataFrames
        Each DataFram is one model and must include columns
        fpr_est, tpr_est, fpr_lb, fpr_ub
    labelL : list of str
        Names of each model for legend"""
    if labelL is None:
        labelL = ['Model %d' % (i) for i, auc in enumerate(rocDfL)]

    colors = sns.color_palette('Set1', n_colors=len(rocDfL))

    plt.cla()
    plt.gca().set_aspect('equal')
    for i, (rocDf, label) in enumerate(zip(rocDfL, labelL)):
        plt.fill_between(rocDf['threshold'], rocDf['%s_lb' % param], rocDf['%s_ub' % param], alpha=0.3, color=colors[i])
        plt.plot(rocDf['threshold'], rocDf['%s_est' % param],'-', color=colors[i], lw=2)
    plt.ylim([0, 1])
    plt.xlabel('Threshold')
    if param == 'tpr':
        plt.ylabel('True Positive Rate')
    else:
        plt.ylabel('False positive rate')
    plt.legend([plt.Line2D([0, 1], [0, 1], color=c, lw=2) for c in colors], labelL, loc='lower right', fontsize=10)
    plt.show()

def _test_oboot(n=50):
    X = np.random.randn(n,2)
    data = X[:,0] + 10 * np.random.rand(n)
    #data = np.random.rand(n)
    split = np.median(data)
    y = np.zeros(n)
    y[data>split] = 1
    y[data<=split] = 0

    O = optimism_bootstrap(X, y, SVC(), 'fit', 'decision_function', roc_auc_score)
    B = percentile_bootstrap(X, y, SVC(), 'fit', 'decision_function', roc_auc_score)
    return O, B

def _test_oboot_roc(n=50):
    X = np.random.randn(n,2)
    data = X[:,0] + 10 * np.random.rand(n)
    #data = np.random.rand(n)
    split = np.median(data)
    y = np.zeros(n)
    y[data>split] = 1
    y[data<=split] = 0

    obDf = ob_roc_curve(X, y, SVC(), 'fit', 'decision_function')
    pbDf = pb_roc_curve(X, y, SVC(), 'fit', 'decision_function')
    return obDf, pbDf

def _test_oboot_roc_plot(n=50):
    X = np.random.randn(n,2)
    data = X[:,0] + 10 * np.random.rand(n)
    #data = np.random.rand(n)
    split = np.median(data)
    y = np.zeros(n)
    y[data>split] = 1
    y[data<=split] = 0

    O = optimism_bootstrap(X, y, SVC(), 'fit', 'decision_function', roc_auc_score)
    B = percentile_bootstrap(X, y, SVC(), 'fit', 'decision_function', roc_auc_score)

    obDf = ob_roc_curve(X, y, SVC(), 'fit', 'decision_function')
    pbDf = pb_roc_curve(X, y, SVC(), 'fit', 'decision_function')
    plotBootROC([obDf, pbDf], labelL=['Optimism adjusted', 'Percentile'], aucL=[O, B])