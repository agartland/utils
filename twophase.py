"""Utility to help with conducting analyses with two-phase sampled data.
Uses calls to the osDesign and survival packages

Example
-------

from quickr import runRscript

matchCols = ['infected', 'sex', 'white', 'black', 'hispanic', 'asian', 'otherrace']

tmpPhaseII = df.loc[(adf['visit_type'] == 'trough') & (df['treatment'] == 1)].drop_duplicates('ptid')
tmpPhaseI = vxDf.loc[(vxDf['valid'] == 1) & (vxDf['treatment'] == 1)]

wcorDf, controlN, caseN = joinIPWeights(tmpPhaseII, tmpPhaseI, matchCols=matchCols)
plotDf = wcorDf.loc[(wcorDf['male'] == 1) & (wcorDf['visit_days'] <= wcorDf['daysrnaneg'])]

controlN = controlN.set_index('stratumID')
caseN = caseN.set_index('stratumID')

Rcmd = cchCmd.format(formula='Surv(daysinfectiondiagnosis, infected) ~ reading_cal')
res, resDf = runRscript(Rcmd, inDf=[plotDf, controlN], outputFile=True)
print(res)
"""

import pandas as pd
import numpy as np

__all__ = ['joinIPWeights',
            'cchCmd',
            'tpsCmd',
            'logisticCmd',
            'printCCH',
            'printTPS',
            'printLogistic']

def joinIPWeights(twoDf, oneDf, matchCols, caseCol='infected'):
    """Compute inverse probability weights for each sample in twoDf,
    limited to the specified visit type (e.g. peak, trough, addtional_peak).
    Strata are defined by matchCols, which default to the sex and race variables
    used in the sampling design.

    Weights are computed by counting the individuals in each strata in the target
    population (i.e. oneDf controls, Phase I sample) and comparing to the Phase II
    sampled population (i.e. twoDf controls). Cases are given a weight of 1 since all
    cases were sampled.

    Parameters
    ----------
    twoDf : pd.DataFrame
        Phase II data from cases and controls.
        Data will be subsetted on visit_type
    oneDf : pd.DataFrame
        Phase I sample including all participants. Constitutes the target population.
    caseCol : str
        Column that defines cases which are fully sampled and given weight one
    matchCols : list
        List of columns that define the Phase II sampling strata

    Returns
    -------
    twoDf : pd.DataFrame
        A copy of twoDf, after subsetting on visit_type and
        joining columns IPWeights (inverse probability weights)
        stratumID and insubcohort (all controls eligible for sampling)
    controlN : pd.DataFrame
        Number of controls in each strata with a column for stratumID
        Index contains variables that define the strata.
    caseN : pd.DataFrame
        Number of cases in each strata with a column for stratumID
        Index contains variables that define the strata."""

    """Subset on visit_type"""
    targetPop = oneDf.groupby(matchCols)['itt'].agg(np.sum)
    sampledPop = twoDf.groupby(matchCols)['itt'].agg(np.sum)

    """Inverse probability weighting"""
    ipwS = targetPop/sampledPop
    ipwS.name = 'IPWeights'
    
    """Strata IDs: label the strata in controls, will be same for cases"""
    stratumID = targetPop.copy()
    stratumID.name = 'stratumID'
    stratumID[:] = np.arange(targetPop.shape[0])
    stratumID = stratumID.xs(0, level=caseCol)
        
    """N of each strata, excluding cases"""
    controlN = targetPop.copy()
    controlN.name = 'controlN'
    controlN = controlN.xs(0, level=caseCol)
    controlN = pd.concat((controlN, stratumID), axis=1)
    
    """N of each strata, excluding controls"""
    caseN = targetPop.copy()
    caseN.name = 'caseN'
    caseN = caseN.xs(1, level=caseCol)
    caseN = pd.concat((caseN, stratumID), axis=1)

    """Set case weights to 1: cases should not be reweighted."""
    ipwS.loc[[1]] = 1.
    twoDf = twoDf.copy().set_index(matchCols).join(ipwS)
    """Assign cases the same strata IDs as controls"""
    notInfCols = [c for c in matchCols if not c == caseCol]
    twoDf = twoDf.reset_index().set_index(notInfCols).join(stratumID)
    twoDf = twoDf.reset_index()

    """Only controls are in the subcohort eligible for sampling"""
    twoDf.loc[:, 'insubcohort'] = (twoDf[caseCol] == 0).astype(int)
    
    """Optional check since the reweighted controls should equal  the target pop"""
    reweightedPop = twoDf.groupby(matchCols)['IPWeights'].agg(np.sum).xs(0, level=caseCol)
    success = np.all(np.isclose(targetPop.xs(0, level=caseCol).values, reweightedPop.values))
    if not success:
        print('Warning: reweighted controls do not match the target population!')
        print(reweightedPop)
        print(targetPop.xs(0, level=caseCol))

    return twoDf, controlN, caseN

cchCmd = """library(survival)
INPUTDF0$stratumID <- factor(INPUTDF0$stratumID)
cohortSize = table(INPUTDF1$stratumID) * INPUTDF1$controlN

fit <- cch({formula},
             data=INPUTDF0,
             subcoh=INPUTDF0$insubcohort,
             stratum=INPUTDF0$stratumID,
             id=INPUTDF0$ptid,
             cohort.size=cohortSize,
             method="II.Borgan")
summary(fit)
write.csv(data.frame(summary(fit)$coeff), OUTPUTFN)"""


tpsCmd = """library(osDesign)
INPUTDF0$stratumID <- factor(INPUTDF0$stratumID)
nn0 = table(INPUTDF1$stratumID) * INPUTDF1$controlN
nn1 = table(INPUTDF2$stratumID) * INPUTDF2$caseN
print(nn0)
print(nn1)
fit <- tps({formula},
             data=INPUTDF0,
             nn0=nn0,
             nn1=nn1,
             group=INPUTDF0$stratumID)
summary(fit)
output <- as.matrix(cbind(exp(fit$coef[2]),exp(fit$coef[2] - sqrt(fit$cove[2,2])*1.96),
exp(fit$coef[2] + sqrt(fit$cove[2,2])*1.96),min(2*(1-pnorm(abs(fit$coef[2]/sqrt(fit$cove[2,2])))),1.0)))
print(output)
write.csv(data.frame(summary(fit)$coeff), OUTPUTFN)"""

logisticCmd = """fit <- glm({formula},
                         data=INPUTDF0,
                         family=binomial(link='logit'))
summary(fit)
write.csv(data.frame(summary(fit)$coeff), OUTPUTFN)"""

def printCCH(resDf):
    for i in resDf.index:
        var = resDf.loc[i, 'Unnamed: 0'] 
        v = resDf.loc[i, 'Value']
        se = resDf.loc[i, 'SE']
        pvalue = resDf.loc[i, 'p']
        print('%s HR = %1.2f [%1.2f, %1.2f]; p = %1.3f' % (var, np.exp(v), np.exp(v-se*1.96), np.exp(v+se*1.96), pvalue))

def printTPS(resDf):
    for i in resDf.index:
        var = resDf.loc[i, 'Unnamed: 0'] 
        v = resDf.loc[i, 'Value']
        se = resDf.loc[i, 'Mod.SE']
        pvalue = resDf.loc[i, 'Mod.p']
        print('%s coef = %1.2f [%1.2f, %1.2f]; p = %1.3f' % (var, v, v-se*1.96, v+se*1.96, pvalue))

def printLogistic(resDf):
    for i in resDf.index:
        var = resDf.loc[i, 'Unnamed: 0'] 
        v = resDf.loc[i, 'Estimate']
        se = resDf.loc[i, 'Std..Error']
        pvalue = resDf.loc[i, 'Pr...z..']
        print('%s coef = %1.2f [%1.2f, %1.2f]; p = %1.3f' % (var, v, v-se*1.96, v+se*1.96, pvalue))

