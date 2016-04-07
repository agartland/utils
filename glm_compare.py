"""
Functions taken from statsmodels result methods
for comparing GLM results and model selection.
"""

#import statsmodels.api as sm
import scipy.stats as stats


def compare_lr_test(model_result, restricted_result):
    """
    Likelihood ratio test to test whether restricted model is correct

    Parameters
    ----------
    model_result : Result instance

    restricted : Result instance
        The restricted model is assumed to be nested in the current model.
        The result instance of the restricted model is required to have two
        attributes, log-likelihood function, `llf`, and residual degrees of
        freedom, `df_resid`.
    
    Returns
    -------
    lr_stat : float
        likelihood ratio, chisquare distributed with df_diff degrees of
        freedom
    p_value : float
        p-value of the test statistic
    df_diff : int
        degrees of freedom of the restriction, i.e. difference in df between
        models
    """

    llf_full = model_result.llf
    llf_restr = restricted_result.llf
    df_full = model_result.df_resid
    df_restr = restricted_result.df_resid

    lrdf = (df_restr - df_full)
    lrstat = -2*(llf_restr - llf_full)
    lr_pvalue = stats.chi2.sf(lrstat, lrdf)

    return lrstat, lr_pvalue, lrdf