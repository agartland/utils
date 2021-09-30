import pandas as pd
import numpy as np
from os.path import join as opj
import sys

from fg_shared import *

sys.path.append(opj(_git, 'utils'))
import quickr

corncob = """
require(dplyr)
require(tidyr)
require(purr)
require(corncob)

###########################################################################################
# APPLIED TO MANY FEATURES: 
###########################################################################################
# Function to Fit A Beta-Binomial Model to A Single Feature
# Note: YOU HAVE A CHOICE OF W,WR,or W0 which repressent different counts
  #  W are counts of Meta-Clonotype (RADIUS ONLY)
  #  WR are counts of Meta-Clonotype (RADIUS + REGEX ONLY) 
  #  W0 are counts of Clonotype (TCRDIST0 basically EXACT CLONOTYPE) 
  #  M total counts
  #  AGE age in years
  #  SEX "Male" or "Female"
  #  DAYS 1 if > 2 days post diagnosis, 0 otherwise
  #  HLA "MATCH" or "NON-MATCH" (in this case A*01)
###########################################################################################
​
#' do_corncob 
#' 
#' Define the beta-binomial we are attempting to fit
#' 
#' @param mydata data.frame
do_corncob <- function(mydata, frm = as.formula('cbind(W, M - W) ~ AGE+SEX+DAYS+HLA')){
  cb1 = bbdml(formula = frm,
              phi.formula = ~ 1,
              data = mydata)
  return(cb1)
}
​
# This wrapper is useful for avoiding crashes do to errors:
possibly_do_corncob = purrr::possibly(do_corncob, otherwise = NA)
​
###########################################################################################
# Split Data by Feature
###########################################################################################
list_of_df_by_feature = example_df %>% split(f = example_df$feature)
###########################################################################################
# Fit Models
###########################################################################################
list_of_fit_models    = purrr::map(list_of_df_by_feature, ~possibly_do_corncob(mydata = .x, frm = as.formula('cbind(W, M - W) ~ AGE+SEX+DAYS+HLA')))
​
list_of_fit_models    = list_of_fit_models[!is.na(list_of_fit_models)]
​
###########################################################################################
# Parse Models
###########################################################################################
#' get bbdml coefficients into a table
#' 
#' 
#' @param cb is object result of corncob::bbdml
#' @param i is a label for the feature name 
#' 
#' @example 
#' purrr::map2(list_of_fit_models, names(list_of_fit_models), ~parse_corncob(cb = .x, i = .y))
parse_corncob <- function(cb,i =1){
  y = summary(cb)$coefficients
  rdf = as.data.frame(y) 
  rdf$param = rownames(rdf)
  rdf = rdf %>% mutate(estimate = Estimate,  se = `Std. Error`, tvalue =  `t value`, pvalue = `Pr(>|t|)`, param) %>% 
    mutate(type = ifelse(grepl(param, pattern = "phi"), "phi", "mu")) %>% 
    mutate(type2 = ifelse(grepl(param, pattern = "Intercept"), "intercept", "covariate")) 
  rdf$feature = i
  return(rdf)
}
​
tabular_results = purrr::map2(list_of_fit_models, names(list_of_fit_models), ~parse_corncob(cb = .x, i = .y))
tabular_results = do.call(rbind, tabular_results) %>% tibble::remove_rownames()
clean_tabular_results = tabular_results %>% select(feature, Estimate, pvalue, param, type, type2) %>% 
  arrange(type2, type, pvalue)
"""