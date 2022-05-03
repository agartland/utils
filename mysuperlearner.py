import numpy as np
import pandas as pd
import itertools

from scipy import optimize

import sklearn
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold


__all__ = ['Superlearner',
           'SuperLearnerCV',
           'binary_classification_score_wrapper']

"""TODO:
 - add NNLS LinearRegression(positive=True) with log link for LogisticRegression"""

""""SuperLearner with sklearn"""

def binary_classification_score_wrapper(metric, **kwargs):
    def wrapped(y_true, y_pred, **kwargs):
        return metric(y_true, np.round(y_pred), **kwargs)
    return wrapped

class SuperLearnerCV:
    def __init__(self, learners, meta_learner=None, inner_cv=None, outer_cv=None, scorers=[]):
        self.learners = learners
        if meta_learner is None:
            self.meta_learner = AUCMinimizer()
        else:
            self.meta_learner = meta_learner

        if inner_cv is None:
            self.inner_cv = StratifiedKFold(n_splits=5)
        elif np.isscalar(inner_cv):
            self.inner_cv = StratifiedKFold(n_splits=inner_cv)
        else:
            self.inner_cv = inner_cv

        if outer_cv is None:
            self.outer_cv = StratifiedKFold(n_splits=1)
        elif np.isscalar(outer_cv):
            self.outer_cv = StratifiedKFold(n_splits=outer_cv)
        else:
            self.outer_cv = outer_cv

        self.scorers = scorers

        self.sl_mod = SuperLearner(learners=self.learners,
                                   meta_learner=self.meta_learner,
                                   cv=self.inner_cv,
                                   scorers=self.scorers)

    def fit_cv(self, X, y, subsets=[]):
        if subsets is None:
            self.subsets = [('all', X.columns)]
        else:
            self.subsets = subsets

        n_splits = self.outer_cv.n_splits
        scores = np.zeros((n_splits, len(self.scorers)))
        for i, (train_idxs, test_idxs) in enumerate(self.outer_cv.split(X, y)):
            X_train, X_test = X.iloc[train_idxs], X.iloc[test_idxs]
            y_train, y_test = y.iloc[train_idxs], y.iloc[test_idxs]
            self.sl_mod.fit(X_train, y_train, subsets)
            yhat_test = self.sl_mod.predict(X_test)
            for score_i, (scorer_name, scorer) in enumerate(self.scorers):
                    scores[i, score_i] = scorer(y_test, yhat_test)

        self.scores = pd.DataFrame(scores, index=range(n_splits), columns=[s[0] for s in self.scorers])
        return self.sl_mod.fit(X_train, y_train)

    def fit(self, X, y, subsets=[]):
        if subsets is None:
            self.subsets = [('all', X.columns)]
        else:
            self.subsets = subsets
        return self.sl_mod.fit(X, y, subsets)
    def predict(self, X):
        yhat = self.sl_mod.predict(X)
        scores = np.zeros(len(self.scorers))
        for score_i, (scorer_name, scorer) in enumerate(self.scorers):
            scores[score_i] = scorer(y, yhat)
        self.scores = pd.Series(scores, index=[s[0] for s in self.scorers])
        return yhat

class SuperLearner:
    def __init__(self, learners, meta_learner=None, cv=None, scorers=[]):
        self.learners = learners
        if meta_learner is None:
            """QUESTION: Can I use linear regression for the SL? If so, could use positive=True for NNLS)
            TODO: Add Nelder-Mean for binary outcome and AUC-ROC loss"""
            self.meta_learner = linear_model.LogisticRegression()
        else:
            self.meta_learner = meta_learner

        if cv is None:
            self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=110820)
        elif np.isscalar(cv):
            self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=110820)
        else:
            self.cv = cv

        """Scorers expected to be list of tuples [('auc', sklearn.metrics.roc_auc), ...]"""
        self.scorers = scorers
        self.frozen_learners = {}
        self.scores = None

    def get_learner_labels(self, as_str=True):
        labels = []
        for j, ((name, mod), (ss_name, ss)) in enumerate(itertools.product(self.learners, self.subsets)):
            if as_str:
                labels.append(f'{name} [{ss_name}]')
            else:
                labels.append((name, ss_name))
        return labels

    def iter_learners(self):
        for j, ((name, mod), (ss_name, ss)) in enumerate(itertools.product(self.learners, self.subsets)):
            yield j, name, mod, ss_name, ss
    
    def fit(self, X, y, subsets=None):
        """Will fit the meta_learner on test data from a CV loop
        and store the frozen learners in self.frozen_learners"""
        if subsets is None:
            self.subsets = [('all', X.columns)]
        else:
            self.subsets = subsets

        n_splits = self.cv.n_splits
        n_learners = len(self.learners) * len(self.subsets)

        """For each of the learners in cross-validation, generating training data for the meta-learner"""
        X_meta = np.zeros((X.shape[0], n_learners))
        y_meta = np.zeros(X.shape[0])
        scores = np.zeros((n_splits, n_learners, len(self.scorers)))

        data_i = 0
        """================CROSS-VALIDATION LOOP==============="""
        for i, (train_idxs, test_idxs) in enumerate(self.cv.split(X, y)):
            X_train, X_test = X.iloc[train_idxs], X.iloc[test_idxs]
            y_train, y_test = y.iloc[train_idxs], y.iloc[test_idxs]

            """y_test becomes y_meta and yhat_test becomes X_meta"""
            y_meta[data_i : data_i + len(y_test)] = y_test
            for learner_i, learner_name, mod, ss_name, ss_cols in self.iter_learners():
                mod.fit(X_train[ss_cols], y_train)
                """Use column 1 which contains probability of the indicator = 1 value"""
                yhat_test = mod.predict_proba(X_test[ss_cols])[:, 1]
                """Using the predicted probabilities from each learner"""
                X_meta[data_i : data_i + len(y_test), learner_i] = yhat_test
                for score_i, (scorer_name, scorer) in enumerate(self.scorers):
                    scores[i, learner_i, score_i] = scorer(y_test, yhat_test)
            data_i += len(y_test)

        """Fit the meta/super-learner to the inner CV predictions"""

        """X_meta are predicted probabilities via the learners using test data
            y_meta are observed binary class labels fom the data"""
        self.X_meta = X_meta
        self.y_meta = y_meta
        self.meta_learner.fit(X_meta, y_meta)

        """Fit each of the learners to all the data and store"""
        for learner_i, learner_name, mod, ss_name, ss_cols in self.iter_learners():
            self.frozen_learners[(learner_name, ss_name)] = deepcopy(mod.fit(X[ss_cols], y))
        
        index = pd.MultiIndex.from_tuples(self.get_learner_labels(as_str=False),
                                          names=['Learner', 'Subset'])
        tmpl = []
        for score_i, (scorer_name, scorer) in enumerate(self.scorers):
            tmp = pd.DataFrame(scores[:, :, score_i].T, index=index)
            tmp = tmp.assign(scorer=scorer_name).set_index('scorer', append=True)
            tmpl.append(tmp)
        self.scores = pd.concat(tmpl, axis=0)
        return self
    
    def predict(self, X):
        """Use the frozen fitted base-learners to predict on the full dataset"""
        n_learners = len(self.learners) * len(self.subsets)
        X_meta_full = np.zeros((X.shape[0], n_learners))
        for learner_i, learner_name, mod, ss_name, ss_cols in self.iter_learners():
            yhat = self.frozen_learners[(learner_name, ss_name)].predict_proba(X[ss_cols])[:, 1]
            X_meta_full[:, learner_i] = yhat

        """Finally, use the fitted meta-learner to predict the labels"""
        yhat_full = self.meta_learner.predict_proba(X_meta_full)[:, 1]
        return yhat_full

    def evaluator(self, X, y):
        n_learners = len(self.learners) * len(self.subsets)
        scores = np.zeros((n_learners, len(self.scorers)))
        """Use the frozen fitted base-learners to predict and produce scores for each learner (not CV)"""
        for learner_i, learner_name, mod, ss_name, ss_cols in self.iter_learners():
            yhat = self.frozen_learners[(learner_name, ss_name)].predict_proba(X[ss_cols])[:, 1]
            #print(learner_name)
            #print(yhat)
            #print(yhat.round())
            for score_i, (scorer_name, scorer) in enumerate(self.scorers):
                scores[learner_i, score_i] = scorer(y, yhat)

        index = pd.MultiIndex.from_tuples(self.get_learner_labels(as_str=False),
                                          names=['Learner', 'Subset'])
        return pd.DataFrame(scores, index=index, columns=[s[0] for s in self.scorers])

    def evaluator_cv(self, X, y):
        n_splits = self.cv.n_splits
        n_learners = len(self.learners) * len(self.subsets)

        scores = np.zeros((n_splits, n_learners, len(self.scorers)))

        data_i = 0
        """================CROSS-VALIDATION LOOP==============="""
        for i, (train_idxs, test_idxs) in enumerate(self.cv.split(X, y)):
            X_train, X_test = X.iloc[train_idxs], X.iloc[test_idxs]
            y_train, y_test = y.iloc[train_idxs], y.iloc[test_idxs]
            #print(np.sum(y_train), np.sum(y_test))
            #print(X.mean())

            for learner_i, learner_name, mod, ss_name, ss_cols in self.iter_learners():
                mod.fit(X_train[ss_cols], y_train)
                """Use column 1 which contains probability of the indicator = 1 value"""
                yhat_test = mod.predict_proba(X_test[ss_cols])[:, 1]
                for score_i, (scorer_name, scorer) in enumerate(self.scorers):
                    scores[i, learner_i, score_i] = scorer(y_test, yhat_test)

        index = pd.MultiIndex.from_tuples(self.get_learner_labels(as_str=False),
                                  names=['Learner', 'Subset'])
        tmpl = []
        for score_i, (scorer_name, scorer) in enumerate(self.scorers):
            tmp = pd.DataFrame(scores[:, :, score_i].T, index=index)
            tmp = tmp.assign(scorer=scorer_name).set_index('scorer', append=True)
            tmpl.append(tmp)
        return pd.concat(tmpl, axis=0)


class AUCMinimizer():
    """Use Nelder-Mead optimization and AUC loss.
    Translated directly from R SuperLearner package.
    Uses Nelder-Mead on one coef per learner, constrained to be positive.
    https://github.com/ecpolley/SuperLearner/blob/ac1aa02fc8b92d4044949102df8eeea4952da753/R/method.R#L359"""
    def __init__(self, maxiter=1000, disp=False):
        self.coef = None
        self.auc_i = None
        self.auc = None
        self.optim = None
        self.disp = disp
        self.maxiter = maxiter

    @staticmethod
    def _auc_diagnostic(X_data, y_data):
        auc = np.zeros(X_data.shape[1])
        for i in range(X_data.shape[1]):
            auc[i] = roc_auc_np(y_data, X_data[:, i])
        return auc

    @staticmethod
    def _auc_loss(x, X_data, y_data):
        auc = roc_auc_np(y_data, np.dot(X_data, x))
        return 1 - auc

    def fit(self, X, y):
        n_learners = X.shape[1]
        bounds = [(0, 1)] * n_learners
        options = dict(maxiter=self.maxiter, disp=self.disp, return_all=True, xatol=0.0001, fatol=0.0001)
        res = optimize.minimize(fun=self._auc_loss,
                                x0=np.ones(n_learners)/n_learners,
                                args=(X_meta, y_meta),
                                method='Nelder-Mead',
                                bounds=bounds,
                                callback=None,
                                options=options)

        self.coef = res.x / np.sum(res.x)
        self.optim = res
        self.auc = 1 - self._auc_loss(res.x, X, y)

        self.auc_i = self._auc_diagnostic(X, y)
        return self

    def predict_proba(self, X):
        tmp = np.dot(X, self.coef)
        return np.concatenate((1 - tmp[:, None], tmp[:, None]), axis=1)

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




class nnls_logistic_regression(linear_model.LinearRegression):
    def __init__():
        super().__init__(fit_intercept=False, positive=True)
    def fit(self, X, y):
        """THIS WON"T WORK AS A SIMPLE WRAPPER: MAYBE CHECKOUT FIT CODE
        TO APPLY LOGIT TO RHS"""
    def predict_proba(self, X):
        yhat = super().predict(X)
        return yhat

    @staticmethod
    def logit(p):
        return np.log(p / (1 - p))
    @staticmethod
    def inv_logit(y):
        return 1 / (1 + np.exp(y))




