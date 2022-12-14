import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

def roc_auc_np(y_true, y_prob):
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    nfalse = np.cumsum(1 - y_true)
    auc = np.cumsum(y_true * nfalse)
    auc = auc[-1] / (nfalse[-1] * (n - nfalse[-1]))
    return auc

def cv_regularized_logistic_regression(X, y, alphas, n_folds=10, **fit_kwargs):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True)

    cv_iter = cv.split(X=X, y=y)
    
    n_alphas = len(alphas)
    auc = np.nan * np.zeros((n_alphas, n_folds))
    coefs = np.zeros((n_alphas, X.shape[1]))
    for outi, (train_ind, test_ind) in enumerate(cv_iter):
        for i, alpha in enumerate(alphas):
            Xtrain, Xtest = X.iloc[train_ind], X.iloc[test_ind]
            ytrain, ytest = y.iloc[train_ind], y.iloc[test_ind]

            results = sm.GLM(endog=ytrain, exog=Xtrain, family=sm.families.Binomial()).fit_regularized(alpha=alpha, **fit_kwargs)
            prob = results.predict(Xtest)
            auc[i, outi] = roc_auc_np(ytest, prob)
            coefs[i, :] = results.params.values
    mini = np.argmax(np.mean(auc, axis=1))
    alpha = alphas[mini]

    results = sm.GLM(endog=y, exog=X, family=sm.families.Binomial()).fit_regularized(alpha=alpha, **fit_kwargs)
    prob = results.predict(X)
    refit_auc = roc_auc_np(y, prob)

    out = dict(auc_folds=pd.Series(auc[mini, :], index=range(n_folds)),
               aucs=pd.DataFrame(aucs, index=alphas, columns=range(n_folds)),
               cv_auc=np.mean(auc[mini, :]),
               refit_auc=refit_auc,
               alpha=alpha,
               refit=results,
               coefs=pd.DataFrame(coefs, index=alphas, columns=X.columns))
    return out