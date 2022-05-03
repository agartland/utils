import numpy as np
import pandas as pd
from scipy import stats, special
import scipy
import sys

# from fg_shared import *
from os.path import join as opj
import os

sys.path.append(opj(_git, 'utils'))
from quickr import R_PATH, R_ENV, runRscript


def _prep_cmd(formula, nsamps=1000):
    cmd = f"""
require("corncob")
require("magrittr")

# data is a matrix of counts [OTUs x samples]
# data = as.data.frame(dfsr[["TRGV8*01"]]['W'])
rownames(INPUTDF0) <- INPUTDF0$OTU
data = subset(INPUTDF0, select = -c(OTU, TAXID, X) )

#tax_table is a matrix with [OTUs x something]
#here attempting to create it from the count matrix
tax = subset(INPUTDF0, select = c(TAXID) )

#samples is a dataframe [samples x covariates (including M)]
# sample_data = as.data.frame(dfsr[["TRGV8*01"]][,c('ptid', 'Timepoint', 'M')])
rownames(INPUTDF1) <- colnames(data)

# samples = INPUTDF1[, rownames(data)]

# rownames(data) <- seq(1, dim(data)[1])
# rownames(sample_data) <- seq(1, dim(data)[1])

pseq = phyloseq(otu_table(as.matrix(data), taxa_are_rows = TRUE),
                tax_table(as.matrix(tax)),
                sample_data(INPUTDF1))
sd = sample_data(INPUTDF1)

cb = differentialTest(formula = as.formula({formula}),
                        phi.formula = ~ 1,
                        formula_null = ~ 1,
                        phi.formula_null = ~ 1,
                        test = "Wald", boot = TRUE,
                        B = {nsamps},
                        data = pseq,
                        sample_data = sd,
                        fdr_cutoff = 0.05)
summary(cb)$coefficients
    """
    return cmd

"""cb = differentialTest(formula = as.formula(~covar),
                        phi.formula = ~ 1,
                        formula_null = ~ 1,
                        phi.formula_null = ~ 1,
                        test = "Wald", boot = TRUE,
                        B = 1000,
                        data = pseq,
                        sample_data = sd,
                        fdr_cutoff = 0.05)
"""

def _test_data():
    np.random.seed(110820)
    N = 100
    M = 500 * np.ones(N, dtype=np.int64)

    mus =  [0.20, 0.10, 0.10, 0.05, 0.05, 0.01, 0.01, 0.001]
    phis = [0.05, 0.10, 0.05, 0.10, 0.05, 0.10, 0.05, 0.10]
    cts = np.zeros((len(mus), N))
    for i, (mu, phi) in enumerate(zip(mus, phis)):
        a1, a2 = beta_binom_full.params_to_a1a2(mu=mu, phi=phi)
        W = stats.betabinom.rvs(M, a1, a2, size=N)
        cts[i, :] = W

    """cts is [OTUs x samples]"""
    sampleid = [f'S{i+1:03d}' for i in range(N)]
    taxid = [f'TX{i+1:02d}' for i in range(len(mus) + 1)]
    ctsdf = pd.DataFrame(cts, columns=sampleid)
    ctsdf.loc[ctsdf.shape[0], :] = M - ctsdf.sum(axis=0).values 
    # ctsdf = pd.concat((ctsdf, M - ctsdf.sum(axis=0).to_frame()), axis=0)
    ctsdf.index = range(ctsdf.shape[0])
    ctsdf = ctsdf.assign(OTU=taxid, TAXID=taxid)

    """samples is [samples x covars]"""
    covar = np.random.randint(2, size=N)
    samples = pd.DataFrame({'covar':covar, 'M':M})
    return ctsdf, samples

def _test_quickr(ctsdf, samples):
    cmd = _prep_cmd(formula='~covar')
    stdout, coeff = runRscript(cmd, inDf=[ctsdf, samples], outputFiles=1)# Rpath=R_PATH, env=R_ENV)
    return coeff