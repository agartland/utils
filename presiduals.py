import numpy as np
import pandas as pd
from os.path import join as opj
import sys

from fg_shared import *

sys.path.append(opj(_git, 'utils'))
from quickr import *


__all__ = ['partial_rank_correlation']


def partial_rank_correlation(data, formulas):
    """partial_Spearman computes the partial Spearman’s rank correlation between variable X and variable Y
    adjusting for other variables, Z. The basic approach involves fitting a specified model of X on Z, a
    specified model of Y on Z, obtaining the probability-scale residuals from both models, and then
    calculating their Pearson’s correlation. X and Y can be any orderable variables, including continuous
    or discrete variables. By default, partial_Spearman uses cumulative probability models (also referred
    as cumulative link models in literature) for both X on Z and Y on Z to preserve the rank-based nature
    of Spearman’s correlation, since the model fit of cumulative probability models only depends on the
    order information of variables. However, for some specific types of variables, options of fitting
    parametric models are also available. See details in fit.x and fit.y

    R Usage
    -----
    partial_Spearman(formula, data, fit.x = "orm", fit.y = "orm",
                    link.x = c("logit", "probit", "cloglog", "loglog", "cauchit",
                    "logistic"), link.y = c("logit", "probit", "cloglog", "loglog",
                    "cauchit", "logistic"), subset, na.action = getOption("na.action"),
                    fisher = TRUE, conf.int = 0.95)

    Parameters
    ----------
    formulas : list of str formulas
    data : pd.DataFrame
    
    NOT IMPLEMENTED
    fit_x/fit_y : str
    link_x/link_y : str

    Returns
    -------
    res : pd.DataFrame, columns: est, lcl, ucl, index: len(formulas)
        Estimates and CIs for each partial rank correlation specified in the list of formulas"""

    rcmd = """
formulas = c({formulas})
results = lapply(formulas, function(frm) PResiduals::partial_Spearman(as.formula(frm), data=INPUTDF))
purrr::map_df(results, ~.x$TS$TB) %>% readr::write_csv(OUTPUTFN)
"""

    pattern = r'[.|/$ \\@%\-+*]'
    clean_formulas = [re.sub(pattern, '_', f) for f in formulas]
    old_columns = data.columns
    new_columns = [re.sub(pattern, '_', c) for c in old_columns]
    data.columns = new_columns

    try:
        stdout, res = runRscript(rcmd.format(formulas=str(list(clean_formulas))[1:-1]),
                                 inDf=data, outputFiles=1, removeTempFiles=None, Rpath=None)
        print(stdout)
    except:
        data.columns = old_columns
        raise
    res = res.assign(formula=formulas)
    return res

