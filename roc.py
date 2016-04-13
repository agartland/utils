import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import statsmodels.api as sm
import sklearn
import sklearn.ensemble
import palettable

sns.set(style='darkgrid', palette='muted', font_scale=1.5)

__all__ = ['computeROC',
           'computeCVROC',
           'plotROC',
           'plotProb',
           'plot2Prob',
           'lassoVarSelect',
           'smLogisticRegression']

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

    if not type(predVars) is list:
        predVars = list(predVars)
    tmp = df[[outcomeVar] + predVars].dropna()

    try:
        results = model.fit(X=tmp[predVars], y=tmp[outcomeVar])
        if hasattr(results, 'predict_proba'):
            prob = results.predict_proba(tmp[predVars])[:,1]
        else:
            prob = results.predict(tmp[predVars])
            results.predict_proba = results.predict
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(tmp[outcomeVar].values, prob)

        acc = sklearn.metrics.accuracy_score(tmp[outcomeVar].values, np.round(prob))
        auc = sklearn.metrics.auc(fpr, tpr)
        tpr[0], tpr[-1] = 0,1
    except sm.tools.sm_exceptions.PerfectSeparationError:
        print 'PerfectSeparationError: %s (N = %d; %d predictors)' % (outcomeVar, tmp.shape[0], len(predVars))
        acc = 1.
        fpr = np.zeros(5)
        tpr = np.ones(5)
        tpr[0], tpr[-1] = 0,1
        prob = tmp[outcomeVar].values.astype(float)
        auc = 1.
        results = None
    return fpr, tpr, auc, acc, results, pd.Series(prob, index=tmp.index, name='Prob')

def computeCVROC(df, model, outcomeVar, predVars, LOO=False, nFolds=10):
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
    LOO : bool
        Leave-one-out cross-validation?
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
        Mean predicted probabilities on test data with index from df"""

    if not type(predVars) is list:
        predVars = list(predVars)
    tmp = df[[outcomeVar] + predVars].dropna()
    if LOO:
        cv = sklearn.cross_validation.LeaveOneOut(n=tmp.shape[0])
        nFolds = tmp.shape[0]
    else:
        cv = sklearn.cross_validation.KFold(n=tmp.shape[0],
                                            n_folds=nFolds,
                                            shuffle=True,
                                            random_state=110820)
    fpr = np.linspace(0, 1, 100)
    tpr = np.nan * np.zeros((fpr.shape[0], nFolds))
    acc = 0
    counter = 0
    results = []
    prob = []
    for i, (trainInd, testInd) in enumerate(cv):
        trainDf = tmp.iloc[trainInd]
        testDf = tmp.iloc[testInd]
        trainFPR, trainTPR, trainAUC, trainACC, res, trainProb = computeROC(trainDf,
                                                                            model,
                                                                            outcomeVar,
                                                                            predVars)
        if not res is None:
            counter += 1
            testProb = res.predict_proba(testDf[predVars])[:,1]
            testFPR, testTPR, _ = sklearn.metrics.roc_curve(testDf[outcomeVar].values, testProb)
            tpr[:,i] = np.interp(fpr, testFPR, testTPR)
            acc += sklearn.metrics.accuracy_score(testDf[outcomeVar].values, np.round(testProb))
            results.append(res)
            prob.append(pd.Series(testProb, index=testDf.index))
    if counter != nFolds:
        print 'ROC: did not finish all folds (%d of %d)' % (counter, nFolds)
    if counter >=1:
        meanTPR = np.nanmean(tpr, axis=1)
        meanTPR[0], meanTPR[-1] = 0,1
        meanACC = acc / counter
        meanAUC = sklearn.metrics.auc(fpr, meanTPR)
        """Compute mean probability over test predictions in CV"""
        probS = pd.concat(prob).groupby(level=0).agg(np.mean)
        probS.name = 'Prob'
    else:
        meanTPR = np.nan * fpr
        meanTPR[0], meanTPR[-1] = 0,1
        meanACC = np.nan
        meanAUC = np.nan
        """Compute mean probability over test predictions in CV"""
        probS = np.nan

    return fpr, meanTPR, meanAUC, meanACC, results, probS

def plotROC(df, model, outcomeVar, predictorsList, predictorLabels=None, rocFunc=computeCVROC, **rocKwargs):
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
    LOO : bool
        Leave-one-out cross-vlaidation?
    rocFunc : computeCVROC or computeROC
        Function for computing the ROC
    rocKwargs : kwargs
        Additional arguments for rocFunc"""
    
    if predictorLabels is None:
        predictorLabels = [' + '.join(predVars) for predVars in predictorsList]
    
    colors = palettable.colorbrewer.qualitative.Set1_8.mpl_colors

    plt.clf()
    plt.gca().set_aspect('equal')
    for predVarsi,predVars in enumerate(predictorsList):
        fpr, tpr, auc, acc, res, probS = rocFunc(df,
                                                 model,
                                                 outcomeVar,
                                                 predVars,
                                                 **rocKwargs)

        label = '%s (AUC = %0.2f; ACC = %0.2f)' % (predictorLabels[predVarsi], auc, acc)
        plt.plot(fpr, tpr, color=colors[predVarsi], lw=2, label=label)
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for %s' % outcomeVar)
    plt.legend(loc="lower right")
    plt.show()

def plotProb(df, outcomeVar, prob, **kwargs):
    """Scatter plot of probabilities for one ourcome.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain outcome and predictor variables.
    outcomeVar : str
    prob : pd.Series
        Predicted probabilities returned from computeROC or computeCVROC"""

    colors = palettable.colorbrewer.qualitative.Set1_3.mpl_colors

    tmp = df.join(prob, how='inner')
    tmp = tmp.sort_values(by='Prob')
    tmp['x'] = np.arange(tmp.shape[0])
    
    plt.clf()
    for color,val in zip(colors, tmp[outcomeVar].unique()):
        ind = tmp[outcomeVar] == val
        lab = '%s = %1.0f (%d)' % (outcomeVar, val, ind.sum())
        plt.scatter(tmp.x.loc[ind], tmp.Prob.loc[ind], label=lab, color=color, **kwargs)
    plt.plot([0,tmp.shape[0]],[0.5, 0.5], 'k--', lw=1)
    plt.legend(loc='upper left')
    plt.ylabel('Predicted Pr(%s)' % outcomeVar)
    plt.ylim((-0.05, 1.05))
    plt.xlim(-1, tmp.shape[0])
    plt.show()

def plot2Prob(df, outcomeVar, prob, **kwargs):
    """Scatter plot of probabilities for two outcomes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain outcome and predictor variables.
    model : sklearn or other model
        Model must have fit and predict methods.
    outcomeVar : list
        Contains two outcomeVar for comparison
    prob : list
        Contains two pd.Series with predicted probabilities
        from computeROC or computeCVROC"""
    labels = {(0,0):'Neither',
              (1,1):'Both',
              (0,1):'%s only' % outcomeVar[1],
              (1,0):'%s only' % outcomeVar[0]}
    colors = palettable.colorbrewer.qualitative.Set1_5.mpl_colors
    tmp = df.join(prob[0], how='inner').join(prob[1], how='inner', rsuffix='_Y')

    plt.clf()
    plt.gca().set_aspect('equal')
    prodIter = itertools.product(tmp[outcomeVar[0]].unique(), tmp[outcomeVar[1]].unique())
    for color,val in zip(colors, prodIter):
        valx, valy = val
        ind = (tmp[outcomeVar[0]] == valx) & (tmp[outcomeVar[1]] == valy)
        lab = labels[val] + ' (%d)' % ind.sum()
        plt.scatter(tmp.Prob.loc[ind], tmp.Prob_Y.loc[ind], label=lab, color=color, **kwargs)
    plt.plot([0.5,0.5], [0,1], 'k--', lw=1)
    plt.plot([0,1], [0.5,0.5], 'k--', lw=1)
    plt.ylim((-0.05, 1.05))
    plt.xlim((-0.05, 1.05))
    plt.legend(loc=0)
    plt.ylabel('Predicted Pr(%s)' % outcomeVar[1])
    plt.xlabel('Predicted Pr(%s)' % outcomeVar[0])
    plt.show()

def lassoVarSelect(df, outcomeVar, predVars, nFolds=10, alpha=None):
    """Apply LASSO to df and return performance metrics,
    optionally in a cross-validation framework to select alpha.

    ROC metrics computed on all data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain outcome and predictor variables.
    outcomeVar : str
    predVars : ndarray or list
        Predictor variables in the model.
    nFolds : int
        N-fold cross-validation (not required for LOO)
    alpha : float
        Constant that multiplies the L1 term (aka lambda)
        Defaults to 1.0
        alpha = 0 is equivalent to OLS
        Use None to set to maximum value given by:
            abs(X.T.dot(Y)).max() / X.shape[0]

    Returns
    -------
    fpr : np.ndarray
        False-positive rate.
    meanTPR : np.ndarray
        True-positive rate.
    auc : float
        Area under the ROC curve.
    acc : float
        Sccuracy score
    results : returned by Lasso.fit()
        Model results object
    prob : pd.Series
        Predicted probabilities with index from df
    varList : list
        Variables with non-zero coefficients
    alpha : float
        Optimal alpha value using coordinate descent path"""
    if not type(predVars) is list:
        predVars = list(predVars)
    tmp = df[[outcomeVar] + predVars].dropna()
    if nFolds == 1 or not alpha is None:
        """Pre-specify alpha, no CV needed"""
        if alpha is None:
            """Use the theoretical max alpha (not sure this is right though)"""
            alpha = np.abs(tmp[predVars].T.dot(tmp[outcomeVar])).max() / tmp.shape[0]
        model = sklearn.linear_model.Lasso(alpha=alpha)
    else:
        model = sklearn.linear_model.LassoCV(cv=nFolds)# , alphas=np.linspace(0.001,0.1,50))
    results = model.fit(y=tmp[outcomeVar], X=tmp[predVars])

    if hasattr(model,'alpha_'):
        optimalAlpha = model.alpha_
    else:
        optimalAlpha = model.alpha
    
    prob = results.predict(tmp[predVars])
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(tmp[outcomeVar].values, prob)
    acc = sklearn.metrics.accuracy_score(tmp[outcomeVar].values, np.round(prob))
    auc = sklearn.metrics.auc(fpr, tpr)
    varList = np.array(predVars)[results.coef_ != 0].tolist()
    probS = pd.Series(prob, index=tmp.index, name='Prob')
    return fpr, tpr, auc, acc, results, probS, varList, optimalAlpha

class smLogisticRegression(object):
    """A wrapper of statsmodels.GLM to use with sklearn interface"""
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            exog = sm.add_constant(X)
        else:
            exog = X
        self.res = sm.GLM(endog=y, exog=exog, family=sm.families.Binomial()).fit()
        return self

    def predict_proba(self, X):
        prob = np.zeros((X.shape[0],2))
        prob[:,0] = 1 - self.predict(X)
        prob[:,1] = self.predict(X)
        return prob

    def predict(self, X):
        if self.fit_intercept:
            exog = sm.add_constant(X)
        else:
            exog = X
        pred = self.res.predict(exog)
        return pred
