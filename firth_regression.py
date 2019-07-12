import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

__all__ = ['fit_firth']
  
"""Python implementation of Firth regression by John Lees
See https://www.ncbi.nlm.nih.gov/pubmed/12758140
Source: https://gist.github.com/johnlees/3e06380965f367e4894ea20fbae2b90d

    # How to call and calculate p-values
    (intercept, beta, bse, fitll) = fit_firth(y, X)

    # Wald test
    waldp = 2 * (1 - stats.norm.cdf(abs(beta[0]/bse[0]))

    # LRT
    null_X = np.delete(X, 1, axis=1)
    (null_intercept, null_beta, null_bse, null_fitll) = fit_firth(y, null_X)
    lrstat = -2*(null_fitll - fitll)
    lrt_pvalue = 1
    if lrstat > 0: # non-convergence
        lrt_pvalue = stats.chi2.sf(lrstat, 1)

"""

def firth_likelihood(beta, logit):
    return -(logit.loglike(beta) + 0.5*np.log(np.linalg.det(-logit.hessian(beta))))

# Do firth regression
# Note information = -hessian, for some reason available but not implemented in statsmodels
def fit_firth(y, X, start_vec=None, step_limit=1000, convergence_limit=0.0001):

    logit_model = smf.Logit(y, X)
    
    if start_vec is None:
        start_vec = np.zeros(X.shape[1])
    
    beta_iterations = []
    beta_iterations.append(start_vec)
    for i in range(0, step_limit):
        pi = logit_model.predict(beta_iterations[i])
        W = np.diagflat(np.multiply(pi, 1-pi))
        var_covar_mat = np.linalg.pinv(-logit_model.hessian(beta_iterations[i]))

        # build hat matrix
        rootW = np.sqrt(W)
        H = np.dot(np.transpose(X), np.transpose(rootW))
        H = np.matmul(var_covar_mat, H)
        H = np.matmul(np.dot(rootW, X), H)

        # penalised score
        U = np.matmul(np.transpose(X), y - pi + np.multiply(np.diagonal(H), 0.5 - pi))
        new_beta = beta_iterations[i] + np.matmul(var_covar_mat, U)

        # step halving
        j = 0
        while firth_likelihood(new_beta, logit_model) > firth_likelihood(beta_iterations[i], logit_model):
            new_beta = beta_iterations[i] + 0.5*(new_beta - beta_iterations[i])
            j = j + 1
            if (j > step_limit):
                raise Exception('Firth regression failed')
                return None

        beta_iterations.append(new_beta)
        if i > 0 and (np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) < convergence_limit):
            break

    return_fit = None
    if np.linalg.norm(beta_iterations[i] - beta_iterations[i-1]) >= convergence_limit:
        raise Exception('Firth regression failed')
    else:
        # Calculate stats
        fitll = -firth_likelihood(beta_iterations[-1], logit_model)
        beta = beta_iterations[-1]
        bse = np.sqrt(np.diagonal(-logit_model.hessian(beta_iterations[-1])))
        
        # Wald test
        pvalues = 2 * (1 - stats.norm.cdf(np.abs(beta/bse)))
        return_fit = beta, bse, fitll, pvalues, i

    return return_fit