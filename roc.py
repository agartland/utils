import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import sklearn
import sklearn.ensemble
from sklearn.model_selection import StratifiedKFold, cross_val_score, LeaveOneOut, LeavePOut, GridSearchCV
import sklearn.linear_model

sns.set(style='darkgrid', palette='muted', font_scale=1.5)

__all__ = ['plotROC', 'plotROCObj',
           'plotProb',
           'plotLogisticL1Paths',
           'plotLogisticL1Vars',
           'logisticL1NestedCV',
           'plotLogisticL1NestedTuning',
           'nestedCVClassifier',
           'computeROC',
           'computeCVROC',
           'smLogisticRegression',
           'rocStats',
           'plotNestedCVParams',
           'plotNestedCVScores']

def plotROCObj(**objD):
    fprL = [o['fpr'] for o in objD.values()]
    tprL = [o['tpr'] for o in objD.values()]
    aucL = [o['AUC'].mean() for o in objD.values()]
    accL = [o['ACC'].mean() for o in objD.values()]
    labelL = objD.keys()
    outcomeVar = [o['Yvar'] for o in objD.values()][0]
    plotROC(fprL, tprL, aucL, accL, labelL, outcomeVar)

def plotROC(fprL, tprL, aucL=None, accL=None, labelL=None, outcomeVar=''):
    if labelL is None and aucL is None and accL is None:
        labelL = ['Model %d' % i for i in range(len(fprL))]
    else:
        labelL = ['%s (AUC = %0.2f; ACC = %0.2f)' % (label, auc, acc) for label, auc, acc in zip(labelL, aucL, accL)]

    colors = sns.color_palette('Set1', n_colors=len(fprL))

    plt.clf()
    plt.gca().set_aspect('equal')
    for i, (fpr, tpr, label) in enumerate(zip(fprL, tprL, labelL)):
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=label)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if outcomeVar == '':
        plt.title('ROC')
    else:
        plt.title('ROC for %s' % outcomeVar)
    plt.legend(loc="lower right", fontsize=10)
    plt.show()

def plotProb(outcome, prob, **kwargs):
    """Scatter plot of probabilities for one outcome.

    Parameters
    ----------
    outcome : pd.Series
    prob : pd.Series
        Predicted probabilities returned from computeROC or computeCVROC"""

    colors = sns.color_palette('Set1', n_colors=2)

    tmp = pd.concat((outcome, prob), join='inner', axis=1)
    tmp = tmp.sort_values(by=[outcome.name, 'Prob'])
    tmp['x'] = np.arange(tmp.shape[0])
    
    plt.clf()
    for color, val in zip(colors, tmp[outcome.name].unique()):
        ind = tmp[outcome.name] == val
        lab = '%s = %1.0f (%d)' % (outcome.name, val, ind.sum())
        plt.scatter(tmp.x.loc[ind], tmp.Prob.loc[ind], label=lab, color=color, **kwargs)
    plt.plot([0, tmp.shape[0]], [0.5, 0.5], 'k--', lw=1)
    plt.legend(loc='upper left')
    plt.ylabel('Predicted Pr(%s)' % outcome.name)
    plt.ylim((-0.05, 1.05))
    plt.xlim(-1, tmp.shape[0])
    plt.show()

def plotLogisticL1Paths(lo):
    tmp = lo['paths'].mean(axis=0)
    
    if len(lo['Xvars']) == (tmp.shape[1] - 1):
        predVars = np.concatenate((np.array(lo['Xvars']), ['Intercept']))
    else:
        predVars = np.array(lo['Xvars'])
    
    plt.clf()
    plt.plot(np.log10(lo['Cs']), tmp, '-')
    yl = plt.ylim()
    xl = plt.xlim()
    plt.plot(np.log10([lo['optimalCs'].mean()]*2), yl, '--k')
    plt.ylabel('Coefficient')
    plt.xlabel('Regularization parameter ($log_{10} C$)\n(lower is more regularized)')
    
    topi = np.nonzero(lo['finalResult'].coef_.ravel() != 0)[0]
    plt.annotate(s='$N_{vars}=%d$' % len(topi),
         xy=(np.log10(lo['finalResult'].C), yl[1]),
         ha='left', va='top', size=10)

    for i in topi:
        a = predVars[i]
        cInd = np.where(tmp[:, i] != 0)[0][0]
        
        y = tmp[cInd+2, i]
        x = np.log10(lo['Cs'][cInd+2])
        plt.annotate(a, xy=(x, y), ha='left', va='center', size=7)

        y = tmp[-1, i]
        x = np.log10(lo['Cs'][-1])
        plt.annotate(a, xy=(x, y), ha='left', va='center', size=7)

    plt.show()

def plotLogisticL1NestedTuning(lo):
    plt.clf()
    colors = sns.color_palette('Set1', n_colors=10)
    for outi in range(lo['scores'].shape[0]):
        sc = lo['scores'][outi, :, :].mean(axis=0)
        plt.plot(np.log10(lo['Cs']), sc, '-', color=colors[outi])
        mnmx = sc.min(), sc.max()
        plt.plot(np.log10([lo['optimalCs'][outi]]*2), mnmx, '--', color=colors[outi])
    plt.xlim(np.log10(lo['Cs'][[0, -1]]))
    plt.ylabel('Score (log-likelihood)')
    plt.xlabel('Regularization parameter ($log_{10} C$)\n(lower is more regularized)')
    plt.title('Regularization tuning in nested CV')
    plt.show()

def plotLogisticL1Vars(lo):
    pctSelection = 100 * (lo['coefs'] != 0).mean(axis=0)
    finalInd = (lo['finalResult'].coef_ != 0).ravel()
    x = np.arange(len(pctSelection))
    plt.clf()
    plt.barh(width=pctSelection[finalInd], bottom=x[finalInd], align='center', color='red', label='Yes')
    plt.barh(width=pctSelection[~finalInd], bottom=x[~finalInd], align='center', color='blue', label='No')
    plt.yticks(range(len(pctSelection)), lo['Xvars'], size=8)
    plt.ylabel('Predictors')
    plt.xlabel('% times selected in 10-fold CV')
    plt.legend(loc=0, title='Final model?')

def logisticL1NestedCV(df, outcomeVar, predVars, nFolds=10, LPO=None, Cs=10, n_jobs=1):
    """Apply logistic regression with L1-regularization (LASSO) to df.
    Uses nested cross-validation framework with inner folds to optimize C
    and outer test folds to evaluate performance.
        
    Parameters
    ----------
    df : pd.DataFrame
        Must contain outcome and predictor variables.
    outcomeVar : str
    predVars : ndarray or list
        Predictor variables in the model.
    nFolds : int
        N-fold stratified cross-validation
    LPO : int or None
        Use Leave-P-Out cross-validation instead of StratifiedNFoldCV
    Cs : int or list
        Each of the values in Cs describes the inverse of regularization strength.
        If Cs is as an int, then a grid of Cs values are chosen in a logarithmic
        scale between 1e-4 and 1e4. Smaller values specify stronger regularization.

    Returns
    -------
    results : dict
        Contains results as keys below:
        fpr:            (100, ) average FPR for ROC
        tpr:            (100, ) average TPR for ROC
        AUC:            (outerFolds, ) AUC of ROC for each outer test fold
        meanAUC:        (1, ) AUC of the average ROC
        ACC:            (outerFolds, ) accuracy across outer test folds
        scores:         (outerFolds, innerFolds, Cs) log-likelihood for each C across inner and outer CV folds
        optimalCs:      (outerFolds, ) optimal C from each set of inner CV
        finalResult:    final fitted model with predict() exposed
        prob:           (N,) pd.Series of predicted probabilities avg over outer folds
        varList:        (Nvars, ) list of vars with non-zero coef in final model
        Cs:             (Cs, ) pre-specified grid of Cs
        coefs:          (outerFolds, predVars) refit with optimalC in each fold
        paths:          (outerFolds, Cs, predVars + intercept) avg across inner folds
        XVars:          list of all vars in X
        yVar:           name of outcome variable
        N:              total number of rows/instances in the model"""
    
    if not isinstance(predVars, list):
        predVars = list(predVars)
    
    tmp = df[[outcomeVar] + predVars].dropna()
    X,y = tmp[predVars].astype(float), tmp[outcomeVar].astype(float)

    if LPO is None:
        innerCV = StratifiedKFold(n_splits=nFolds, shuffle=True)
        outerCV = StratifiedKFold(n_splits=nFolds, shuffle=True)
    else:
        innerCV = LeavePOut(LPO)
        outerCV = StratifiedKFold(LPO)

    
    scorerFunc = sklearn.metrics.make_scorer(sklearn.metrics.log_loss,
                                             greater_is_better=False,
                                             needs_proba=True,
                                             needs_threshold=False)
    
    fpr = np.linspace(0, 1, 100)
    tpr = np.nan * np.zeros((fpr.shape[0], nFolds))
    acc = np.nan * np.zeros(nFolds)
    auc = np.nan * np.zeros(nFolds)
    paths = []
    coefs = []
    probs = []
    optimalCs = np.nan * np.zeros(nFolds)
    scores = []

    for outi, (trainInd, testInd) in enumerate(outerCV.split(X=X, y=y)):
        Xtrain, Xtest = X.iloc[trainInd], X.iloc[testInd]
        ytrain, ytest = y.iloc[trainInd], y.iloc[testInd]

        model = sklearn.linear_model.LogisticRegressionCV(Cs=Cs,
                                                          cv=innerCV,
                                                          penalty='l1',
                                                          solver='liblinear',
                                                          scoring=scorerFunc,
                                                          refit=True,
                                                          n_jobs=n_jobs)
        """With refit = True, the scores are averaged across all folds,
        and the coefs and the C that corresponds to the best score is taken,
        and a final refit is done using these parameters."""

        results = model.fit(X=Xtrain, y=ytrain)
        prob = results.predict_proba(Xtest)
        
        class1Ind = np.nonzero(results.classes_ == 1)[0][0]
        fprTest, tprTest, _ = sklearn.metrics.roc_curve(ytest, prob[:, class1Ind])


        tpr[:, outi] = np.interp(fpr, fprTest, tprTest)
        auc[outi] = sklearn.metrics.auc(fprTest, tprTest)
        acc[outi] = sklearn.metrics.accuracy_score(ytest, np.round(prob[:, class1Ind]), normalize=True)
        optimalCs[outi] = results.C_[0]
        scores.append(results.scores_[1])
        paths.append(results.coefs_paths_[1])
        coefs.append(results.coef_)
        probs.append(pd.Series(prob[:, class1Ind], index=Xtest.index))
    
    meanTPR = np.mean(tpr, axis=1)
    meanTPR[0], meanTPR[-1] = 0, 1
    meanACC = np.mean(acc)
    meanAUC = sklearn.metrics.auc(fpr, meanTPR)
    meanC = 10**np.mean(np.log10(optimalCs))
    paths = np.concatenate([p.mean(axis=0, keepdims=True) for p in paths], axis=0)
    scores = np.concatenate([s[None, :, :] for s in scores], axis=0)
    
    """Compute mean probability over test predictions in CV"""
    probS = pd.concat(probs).groupby(level=0).agg(np.mean)
    probS.name = 'Prob'

    """Refit all the data with the optimal C for variable selection and 
    classification of holdout data"""
    model = sklearn.linear_model.LogisticRegression(C=meanC,
                                                    penalty='l1',
                                                    solver='liblinear')
    result = model.fit(X=X, y=y)
    varList = np.array(predVars)[result.coef_.ravel() != 0].tolist()

    rocRes = rocStats(y, np.round(probS))
    
    outD = {'fpr':fpr,                      # (100, ) average FPR for ROC
            'tpr':meanTPR,                  # (100, ) average TPR for ROC
            'AUC':auc,                      # (outerFolds, ) AUC of ROC for each outer test fold
            'mAUC': meanAUC,                # (1, ) AUC of the average ROC
            'ACC':acc,                      # (outerFolds, ) accuracy across outer test folds
            'mACC':np.mean(acc),
            'scores': scores,               # (outerFolds, innerFolds, Cs) score for each C across inner and outer CV folds
            'optimalCs':optimalCs,          # (outerFolds, ) optimal C from each set of inner CV
            'C':meanC,
            'finalResult': result,          # final fitted model with predict() exposed
            'prob':probS,                   # (N,) pd.Series of predicted probabilities avg over outer folds
            'varList':varList,              # list of vars with non-zero coef in final model
            'Cs':Cs,                        # pre-specified grid of Cs
            'coefs':np.concatenate(coefs),  # (outerFolds, predVars) refit with optimalC in each fold
            'paths':paths,                  # (outerFolds, Cs, predVars + intercept) avg across inner folds 
            'Xvars':predVars,
            'Yvar':outcomeVar,
            'N':tmp.shape[0]}                  
    outD.update(rocRes[['Sensitivity', 'Specificity']].to_dict())
    return outD

def nestedCVClassifier(df, outcomeVar, predVars, model, params={}, nFolds=10, LPO=None, scorer='log_loss', n_jobs=1):
    """Apply model to df in nested cross-validation framework
    with inner folds to optimize hyperparameters.
    and outer test folds to evaluate performance.
        
    Parameters
    ----------
    df : pd.DataFrame
        Must contain outcome and predictor variables.
    outcomeVar : str
    predVars : ndarray or list
        Predictor variables in the model.
    model : sklearn model
    nFolds : int
        N-fold stratified cross-validation
    LPO : int or None
        Use Leave-P-Out cross-validation instead of StratifiedNFoldCV
    params : dict
        Keys of model hyperparameters withe values to try in
        a grid search.

    Returns
    -------
    results : dict
        Contains results as keys below:
        fpr:            (100, ) average FPR for ROC
        tpr:            (100, ) average TPR for ROC
        AUC:            (outerFolds, ) AUC of ROC for each outer test fold
        meanAUC:        (1, ) AUC of the average ROC
        ACC:            (outerFolds, ) accuracy across outer test folds
        scores:         (outerFolds, innerFolds, Cs) log-likelihood for each C across inner and outer CV folds
        optimalCs:      (outerFolds, ) optimal C from each set of inner CV
        finalResult:    final fitted model with predict() exposed
        prob:           (N,) pd.Series of predicted probabilities avg over outer folds
        varList:        (Nvars, ) list of vars with non-zero coef in final model
        Cs:             (Cs, ) pre-specified grid of Cs
        coefs:          (outerFolds, predVars) refit with optimalC in each fold
        paths:          (outerFolds, Cs, predVars + intercept) avg across inner folds
        XVars:          list of all vars in X
        yVar:           name of outcome variable
        N:              total number of rows/instances in the model"""
    
    if not isinstance(predVars, list):
        predVars = list(predVars)
    
    tmp = df[[outcomeVar] + predVars].dropna()
    X,y = tmp[predVars].astype(float), tmp[outcomeVar].astype(float)

    if LPO is None:
        innerCV = StratifiedKFold(n_splits=nFolds, shuffle=True)
        outerCV = StratifiedKFold(n_splits=nFolds, shuffle=True)
    else:
        innerCV = LeavePOut(LPO)
        outerCV = StratifiedKFold(LPO)
    
    if scorer == 'log_loss':
        scorerFunc = sklearn.metrics.make_scorer(sklearn.metrics.log_loss,
                                                 greater_is_better=False,
                                                 needs_proba=True,
                                                 needs_threshold=False)
    elif scorer == 'accuracy':
        scorerFunc = sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score,
                                                 greater_is_better=True,
                                                 needs_proba=False,
                                                 needs_threshold=False)
    
    fpr = np.linspace(0, 1, 100)
    tpr = np.nan * np.zeros((fpr.shape[0], nFolds))
    acc = np.nan * np.zeros(nFolds)
    auc = np.nan * np.zeros(nFolds)
    probs = []
    optimalParams = []
    optimalScores = []
    cvResults = []

    for outi, (trainInd, testInd) in enumerate(outerCV.split(X=X, y=y)):
        Xtrain, Xtest = X.iloc[trainInd], X.iloc[testInd]
        ytrain, ytest = y.iloc[trainInd], y.iloc[testInd]

        clf = GridSearchCV(estimator=model, param_grid=params, cv=innerCV, refit=True, scoring=scorerFunc, n_jobs=n_jobs)
        clf.fit(Xtrain, ytrain)
        cvResults.append(clf.cv_results_)
        optimalParams.append(clf.best_params_)
        optimalScores.append(clf.best_score_)

        prob = clf.predict_proba(Xtest)
        fprTest, tprTest, _ = sklearn.metrics.roc_curve(ytest, prob[:, 1])
        tpr[:, outi] = np.interp(fpr, fprTest, tprTest)
        auc[outi] = sklearn.metrics.auc(fprTest, tprTest)
        acc[outi] = sklearn.metrics.accuracy_score(ytest, np.round(prob[:, 1]), normalize=True)
        
        probs.append(pd.Series(prob[:, 1], index=Xtest.index))
    
    meanTPR = np.mean(tpr, axis=1)
    meanTPR[0], meanTPR[-1] = 0, 1
    meanACC = np.mean(acc)
    meanAUC = sklearn.metrics.auc(fpr, meanTPR)
    
    """Compute mean probability over test predictions in CV"""
    probS = pd.concat(probs).groupby(level=0).agg(np.mean)
    probS.name = 'Prob'

    """Select "outer" optimal param for final model"""
    avgFunc = lambda v: 10**np.mean(np.log10(v))
    # avgFunc = lambda v: np.mean(v)
    optP = {k:avgFunc([o[k] for o in optimalParams]) for k in optimalParams[0].keys()}
    
    for k,v in optP.items():
        setattr(model, k, v)
    result = model.fit(X=X, y=y)
    
    rocRes = rocStats(y, np.round(probS))
    
    outD = {'fpr':fpr,                     
            'tpr':meanTPR,               
            'AUC':auc,                   
            'mAUC': meanAUC,          
            'mACC':np.mean(acc),
            'ACC':acc,
            'CVres':cvResults,          
            'optimalScores': np.array(optimalScores),
            'optimalParams': optimalParams,
            'finalParams':optP,
            'finalResult': result,          # final fitted model with predict() exposed
            'prob':probS,                   # (N,) pd.Series of predicted probabilities avg over outer folds
            'Xvars':predVars,
            'Yvar':outcomeVar,
            'N':tmp.shape[0],
            'params':params}                  
    outD.update(rocRes[['Sensitivity', 'Specificity']].to_dict())
    return outD

def plotNestedCVScores(lo):
    scores = _reshape(lo, 'mean_test_score').mean(axis=0)

    paramKeys = sorted(lo['params'].keys())
    plt.clf()
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    """plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))"""
    plt.pcolormesh(scores)
    plt.xlabel('$log_{10} %s$' % paramKeys[1])
    plt.ylabel('$log_{10} %s$' % paramKeys[0])
    plt.colorbar()
    plt.yticks(np.arange(len(lo['params'][paramKeys[0]]))[::2] + 0.5,
               np.round(np.log10(lo['params'][paramKeys[0]])[::2], 2))
    plt.xticks(np.arange(len(lo['params'][paramKeys[1]]))[::2] + 0.5,
               np.round(np.log10(lo['params'][paramKeys[1]])[::2], 2))
    plt.title('Mean score over outer CV')
    plt.show()

def _reshape(lo, key):
    paramKeys = sorted(lo['params'].keys())
    paramL = [len(lo['params'][k]) for k in paramKeys]
    tmp = [lo['CVres'][i][key][None, :] for i in range(len(lo['CVres']))]
    folds = len(tmp)
    tmp = [np.array(t, dtype=float) for t in tmp]
    tmp = np.concatenate(tmp, axis=0)
    rs = (folds, paramL[0], paramL[1])
    return tmp.reshape(rs)

def plotNestedCVParams(lo):
    """Shows variability in the outer folds"""
    scores = _reshape(lo, 'mean_test_score')

    paramKeys = sorted(lo['params'].keys())
    nFolds = scores.shape[0]
    colors = sns.color_palette('Set1', n_colors=nFolds)
    
    plt.clf()
    ax1 = plt.subplot(1,2,1)
    for foldi in range(nFolds):
        y = scores.mean(axis=2)[foldi,:]
        plt.plot(np.log10(lo['params'][paramKeys[0]]), y, color=colors[foldi])
        plt.plot(np.log10([lo['optimalParams'][foldi][paramKeys[0]]]*2), [np.min(y), np.max(y)], '--', color=colors[foldi])
    x = np.log10([lo['finalParams'][paramKeys[0]]]*2)
    yl = plt.ylim()
    plt.plot(x, yl, '--k')
    plt.xlabel('$log_{10} %s$' % paramKeys[0])
    plt.ylabel('Score')
    ax2 = plt.subplot(1,2,2)
    for foldi in range(nFolds):
        y = scores.mean(axis=1)[foldi,:]
        plt.plot(np.log10(lo['params'][paramKeys[1]]), y, color=colors[foldi])
        plt.plot(np.log10([lo['optimalParams'][foldi][paramKeys[1]]]*2), [np.min(y), np.max(y)], '--', color=colors[foldi])
    x = np.log10([lo['finalParams'][paramKeys[1]]]*2)
    yl = plt.ylim()
    plt.plot(x, yl, '--k')
    plt.xlabel('$log_{10} %s$' % paramKeys[1])

    ylim1 = ax1.get_ylim()
    ylim2 = ax2.get_ylim()
    yl = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
    ax1.set_ylim(yl)
    ax2.set_ylim(yl)
    plt.show()

def computeROC(df, model, outcomeVar, predVars):
    """Apply model to df and return performance metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain outcome and predictor variables.
    model : sklearn or other model
        Model must have fit and predict methods.
    outcomeVar : str
    predVars : ndarray or list
        Predictor variables in the model.

    Returns
    -------
    fpr : np.ndarray
        False-positive rate
    tpr : np.ndarray
        True-positive rate
    auc : float
        Area under the ROC curve
    acc : float
        Accuracy score
    results : returned by model.fit()
        Model results object for test prediction in CV
    prob : pd.Series
        Predicted probabilities with index from df"""

    if not isinstance(predVars, list):
        predVars = list(predVars)
    tmp = df[[outcomeVar] + predVars].dropna()

    try:
        results = model.fit(X=tmp[predVars], y=tmp[outcomeVar])
        if hasattr(results, 'predict_proba'):
            prob = results.predict_proba(tmp[predVars])[:, 1]
        else:
            prob = results.predict(tmp[predVars])
            results.predict_proba = results.predict
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(tmp[outcomeVar].values, prob)

        acc = sklearn.metrics.accuracy_score(tmp[outcomeVar].values, np.round(prob), normalize=True)
        auc = sklearn.metrics.auc(fpr, tpr)
        tpr[0], tpr[-1] = 0, 1
    except:
        print('PerfectSeparationError: %s (N = %d; %d predictors)' % (outcomeVar, tmp.shape[0], len(predVars)))
        acc = 1.
        fpr = np.zeros(5)
        tpr = np.ones(5)
        tpr[0], tpr[-1] = 0, 1
        prob = df[outcomeVar].values.astype(float)
        auc = 1.
        results = None
    assert acc <= 1
    outD = {'fpr':fpr,
            'tpr':tpr,
            'AUC':auc,
            'ACC':acc,
            'result':results,
            'probs':pd.Series(prob, index=tmp.index, name='Prob')}
    return outD

def computeCVROC(df, model, outcomeVar, predVars, nFolds=10, LOO=False):
    """Apply model to df and return performance metrics in a cross-validation framework.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain outcome and predictor variables.
    model : sklearn or other model
        Model must have fit and predict methods.
    outcomeVar : str
    predVars : ndarray or list
        Predictor variables in the model.
    nFolds : int
        N-fold cross-validation (not required for LOO)

    Returns
    -------
    fpr : np.ndarray
        Pre-specified vector of FPR thresholds for interpolation
        fpr = np.linspace(0, 1, 100)
    meanTPR : np.ndarray
        Mean true-positive rate in test fraction.
    auc : float
        Area under the mean ROC curve.
    acc : float
        Mean accuracy score in test fraction.
    results : returned by model.fit()
        Training model results object for each fold
    prob : pd.Series
        Mean predicted probabilities on test data with index from df
    success : bool
        An indicator of whether the cross-validation was completed."""

    if not isinstance(predVars, list):
        predVars = list(predVars)
    
    tmp = df[[outcomeVar] + predVars].dropna()
    X,y = tmp[predVars].astype(float), tmp[outcomeVar].astype(float)

    if LOO:
        cv = LeaveOneOut()
        nFolds = cv.get_n_splits(y)
        cv_iter = cv.split(y=y)
    else:
        cv = StratifiedKFold(n_splits=nFolds, shuffle=True)
        cv_iter = cv.split(X=X, y=y)
    
    fpr = np.linspace(0, 1, 100)
    tpr = np.nan * np.zeros((fpr.shape[0], nFolds))
    acc = np.nan * np.zeros(nFolds)
    auc = np.nan * np.zeros(nFolds)
    coefs = []
    probs = []

    for outi, (trainInd, testInd) in enumerate(cv_iter):
        Xtrain, Xtest = X.iloc[trainInd], X.iloc[testInd]
        ytrain, ytest = y.iloc[trainInd], y.iloc[testInd]

        results = model.fit(X=Xtrain, y=ytrain)
        prob = results.predict_proba(Xtest)
        
        class1Ind = np.nonzero(results.classes_ == 1)[0][0]
        fprTest, tprTest, _ = sklearn.metrics.roc_curve(ytest, prob[:, class1Ind])

        tpr[:, outi] = np.interp(fpr, fprTest, tprTest)
        auc[outi] = sklearn.metrics.auc(fprTest, tprTest)
        acc[outi] = sklearn.metrics.accuracy_score(ytest, np.round(prob[:, class1Ind]), normalize=True)
        coefs.append(results.coef_[None,:])
        probs.append(pd.Series(prob[:, class1Ind], index=Xtest.index))
    
    meanTPR = np.mean(tpr, axis=1)
    meanTPR[0], meanTPR[-1] = 0, 1
    meanACC = np.mean(acc)
    meanAUC = sklearn.metrics.auc(fpr, meanTPR)
    
    """Compute mean probability over test predictions in CV"""
    probS = pd.concat(probs).groupby(level=0).agg(np.mean)
    probS.name = 'Prob'

    """Refit all the data for final model"""
    result = model.fit(X=X, y=y)

    rocRes = rocStats(y, np.round(probS))
    
    outD = {'fpr':fpr,                      # (100, ) average FPR for ROC
            'tpr':meanTPR,                  # (100, ) average TPR for ROC
            'AUC':auc,                      # (CVfolds, ) AUC of ROC for each outer test fold
            'mAUC': meanAUC,                # (1, ) AUC of the average ROC
            'mACC': np.mean(acc),
            'ACC':acc,                      # (CVfolds, ) accuracy across outer test folds
            'finalResult': result,          # final fitted model with predict() exposed
            'prob':probS,                   # (N,) pd.Series of predicted probabilities avg over outer folds
            'coefs':np.concatenate(coefs),  # (CVfolds, predVars)
            'Xvars':predVars,
            'Yvar':outcomeVar,
            'nFolds':nFolds,
            'LOO':'Yes' if LOO else 'No',
            'N':tmp.shape[0]}                  
    outD.update(rocRes[['Sensitivity', 'Specificity']].to_dict())
    return outD

class smLogisticRegression(object):
    """A wrapper of statsmodels.GLM to use with sklearn interface"""
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.classes_ = np.array([0., 1.])

    def fit(self, X, y):
        if self.fit_intercept:
            exog = sm.add_constant(X, has_constant='add')
        else:
            exog = X
        self.res = sm.GLM(endog=y, exog=exog, family=sm.families.Binomial()).fit()
        self.coef_ = self.res.params[X.columns].values.ravel()
        return self

    def predict_proba(self, X):
        prob = np.zeros((X.shape[0], 2))
        prob[:, 0] = 1 - self.predict(X)
        prob[:, 1] = self.predict(X)
        return prob

    def predict(self, X):
        if self.fit_intercept:
            exog = sm.add_constant(X, has_constant='add')
        else:
            exog = X
        pred = self.res.predict(exog)
        return pred

def rocStats(obs, pred, returnSeries=True):
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

    n = obs.shape[0]
    a = (obs.astype(bool) & pred.astype(bool)).sum() # TP
    b = (obs.astype(bool) & (~pred.astype(bool))).sum() # FN
    c = ((~obs.astype(bool)) & pred.astype(bool)).sum() # FP
    d = ((~obs.astype(bool)) & (~pred.astype(bool))).sum() # TN 

    sens = a / (a+b)
    spec = d / (c+d)
    ppv = a / (a+c)
    npv = d / (b+d)
    nnt = 1 / (a/(a+c) - b/(b+d))
    acc = (a + d)/n
    rr = (a / (a+c)) / (b / (b+d))
    OR = (a/b) / (c/d)

    if returnSeries:
        vec = [sens, spec, ppv, npv, nnt, acc, rr, OR]
        out = pd.Series(vec, name='ROC', index=['Sensitivity', 'Specificity', 'PPV', 'NPV', 'NNT', 'ACC', 'RR', 'OR'])
    else:
        out = (sens, spec, ppv, npv, nnt, acc, rr, OR)
    return out

def rocStats2x2(a, b, c, d):
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
    a = a / n
    b = b / n
    c = c / n
    d = d / n

    sens = a / (a+c)
    spec = d / (b+d)
    ppv = a / (a+b)
    npv = d / (c+d)
    nnt = 1 / (a/(a+b) - c/(c+d))
    acc = (a + d)/n
    rr = (a / (a+b)) / (c / (c+d))
    OR = (a/c) / (b/d)
    prevOut = a + c
    prevPred = a + b

    vec = [sens, spec, ppv, npv, nnt, acc, rr, OR, prevOut, prevPred, a, b, c, d]
    labels = ['Sensitivity', 'Specificity',
                'PPV', 'NPV', 'NNT',
                'ACC', 'RR', 'OR',
                'prevOut', 'prevPred',
                'A', 'B', 'C', 'D']
    if np.isscalar(a):
        out = pd.Series(vec, name='ROC', index=labels)
    else:
        out = pd.DataFrame({k:v for k,v in zip(labels, vec)})
    return out

def compute2x2FromSensSpecPrev(sens, spec, prev, returnSeries=True):
    """Compute the 2x2 probabilities a, b, c, d from sensitivity,
    specificity and marginal outcome prevalence. Can be used to translate
    sensitivity and specificity in one cohort for simulation in another cohort
    with known, but different outcome prevalence.

    Parameters
    ----------
    sensitivity : float
        Rate of detecting positives among the true positives.
        1 - false-negative rate
        a / (a + c)
    specificity : float
        Rate of rejecting negatives among the true negatives.
        1 - false-positive rate
        d / (d + b)
    prev : float
        Marginal prevalence of the outcome.
        a + c or 1 - (c + d)


    Returns
    -------
    a, b, c, d : float or pd.Series
        Probabilities for each bin in the 2x2 table."""

    a = prev * sens  # Pr(OUTCOME+, PRED+)
    d = (1-prev) * spec # Pr(-, -)
    b = (d/spec) - d # Pr(-, +)
    c = (a/sens) - a # Pr(+, -)

    assert a + b + c + d == 1
    assert sens == a / (a + c)
    assert spec == d / (b + d)
    assert prev == a + c

    return pd.Series([a, b, c, d,], index=['A', 'B', 'C', 'D'])


def possibleProbFromMarginals(outPrev, predPrev):
    """Fix the marginals of the outcome and biomarker prevalence
    and compute possible 2x2 joint probabilities for which all 
    probabilities are on the interval [0, 1]"""
    def fixB(n):
        b = np.linspace(0, 1, n)
        a = predPrev - b
        c = outPrev - a
        d = 1 - a - b - c
        return a, b, c, d
    def fixA(n):
        a = np.linspace(0, 1, n)
        b = predPrev - a
        c = outPrev - a
        d = 1 - a - b - c
        return a, b, c, d
    def fixC(n):
        c = np.linspace(0, 1, n)
        a = outPrev - c
        b = predPrev - a
        d = 1 - a - b - c
        return a, b, c, d
    def fixD(n):
        d = np.linspace(0, 1, n)
        c = (1 - predPrev) - d
        a = outPrev - c
        b = predPrev - a
        return a, b, c, d

    out = []
    for f in [fixA, fixB, fixC, fixD]:
        fLab = f.__qualname__.split('.')[-1]
        a, b, c, d = f(10000)
        abcd = np.concatenate((a[:, None], b[:, None], c[:, None], d[:, None]), axis=1)
        anyNan = np.any((abcd < 0) | (abcd > 1), axis=1)
        abcd = abcd[~anyNan, :]
        resDf = rocStats2x2(abcd[:, 0], abcd[:, 1], abcd[:, 2], abcd[:, 3])
        resDf.loc[:, 'A'] = abcd[:, 0]
        resDf.loc[:, 'B'] = abcd[:, 1]
        resDf.loc[:, 'C'] = abcd[:, 2]
        resDf.loc[:, 'D'] = abcd[:, 3]
        if resDf.shape[0] > 0:
            resDf.loc[:, 'Meth'] = fLab
            out.append(resDf)

    return pd.concat(out).drop_duplicates()



"""Code below here is old but I may still update at some point"""

def plotCVROC(df, model, outcomeVar, predictorsList, predictorLabels=None, rocFunc=computeCVROC, **rocKwargs):
    """Plot of multiple ROC curves using same model and same outcomeVar with
    different sets of predictors.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain outcome and predictor variables.
    model : sklearn or other model
        Model must have fit and predict methods.
    outcomeVar : str
    predictorsList : list
        List of lists of predictor variables for each model.
    predictorLabels : list
        List of labels for the models (optional)
    rocFunc : computeCVROC or computeROC
        Function for computing the ROC
    rocKwargs : kwargs
        Additional arguments for rocFunc"""
    
    if predictorLabels is None:
        predictorLabels = [' + '.join(predVars) for predVars in predictorsList]
    
    colors = sns.color_palette('Set1', n_colors=8)

    fprList, tprList, labelList = [], [], []

    for predVarsi, predVars in enumerate(predictorsList):
        fpr, tpr, auc, acc, res, probS, success = rocFunc(df,
                                                         model,
                                                         outcomeVar,
                                                         predVars,
                                                         **rocKwargs)

        if success:
            label = '%s (AUC = %0.2f; ACC = %0.2f)' % (predictorLabels[predVarsi], auc, acc)
        else:
            label = '%s (AUC* = %0.2f; ACC* = %0.2f)' % (predictorLabels[predVarsi], auc, acc)
        labelList.append(label)
        fprList.append(fpr)
        tprList.append(tpr)
    plotROC(fprList, tprList, labelL=labelList)


def plot2Prob(df, outcomeVar, prob, **kwargs):
    """Scatter plot of probabilities for two outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain two outcome variables.
    model : sklearn or other model
        Model must have fit and predict methods.
    outcomeVar : list
        Contains two outcomeVar for comparison
    prob : list
        Contains two pd.Series with predicted probabilities
        from computeROC or computeCVROC"""
    labels = {(0, 0):'Neither',
              (1, 1):'Both',
              (0, 1):'%s only' % outcomeVar[1],
              (1, 0):'%s only' % outcomeVar[0]}
    markers = ['o', 's', '^', 'x']
    colors = sns.color_palette('Set1', n_colors=4)
    tmp = df[outcomeVar].join(prob[0], how='inner').join(prob[1], how='inner', rsuffix='_Y')

    plt.clf()
    plt.gca().set_aspect('equal')
    prodIter = itertools.product(tmp[outcomeVar[0]].unique(), tmp[outcomeVar[1]].unique())
    for color, m, val in zip(colors, markers, prodIter):
        valx, valy = val
        ind = (tmp[outcomeVar[0]] == valx) & (tmp[outcomeVar[1]] == valy)
        lab = labels[val] + ' (%d)' % ind.sum()
        plt.scatter(tmp.Prob.loc[ind], tmp.Prob_Y.loc[ind], label=lab, color=color, marker=m, **kwargs)
    plt.plot([0.5, 0.5], [0, 1], 'k--', lw=1)
    plt.plot([0, 1], [0.5, 0.5], 'k--', lw=1)
    plt.ylim((-0.05, 1.05))
    plt.xlim((-0.05, 1.05))
    plt.legend(loc='upper left')
    plt.ylabel('Predicted Pr(%s)' % outcomeVar[1])
    plt.xlabel('Predicted Pr(%s)' % outcomeVar[0])
    plt.show()