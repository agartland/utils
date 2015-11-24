from __future__ import division
import pandas as pd
import numpy as np
from lifelines.estimation import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from myfisher import *

__all__ = ['estTECI',
            'estTEPH']

def estTECI(df, followup, treatment_col='treated', duration_col='dx', event_col='disease', alpha=0.05):
    """Estimates treatment efficacy using cumulative incidence (CI) Nelson-Aalen (NA) estimators.

    TE = 1 - RR
    
    RR = CI_NA1 / CI_NA2

    Confidence 
    
    Parameters
    ----------
    df : pandas.DataFrame
        Each row is a participant.
    followup : float
        Follow-up time tau, at which CI will be estimated.
    treatment_col : string
        Column in df indicating treatment (values 1 or 0).
    dx_col : string
        Column in df indicating time of the event or censoring.
    event_col : string
        Column in df indicating events (values 1 or 0 with censored data as 0).
    
    Returns
    -------
    est : float
        Estimate of treatment efficacy
    ci : vector, length 2
        (1-alpha)%  confidence interval, [LL, UL]
    pvalue : float
        P-value for H0: TE=0 from Wald statistic"""

    def _gsurv(x, y):
        """Contrast function g"""
        return np.log((1-y) / (1-x))
    def _gxsurv(x, y):
        """Derivative of gsurv in x"""
        return 1 / (1-x)
    def _gysurv(x, y):
        """Derivative of gsurv in y"""
        return -1 / (1-y)

    def _vsurv(nsamp1, nsamp2, sa1, sa2, varsa1, varsa2):
        return ((nsamp1*varsa1/((1-sa1)**2)) + (nsamp2*varsa2/((1-sa2)**2)))**(-0.5)

    def _vsurv2(nsamp1, nsamp2, sa1, sa2, varsa1, varsa2):
        """By the delta method"""
        return ((varsa1/((1-sa1)**2)) + (varsa2/((1-sa2)**2)))**(-0.5)

    def _additive_var(self, population, deaths):
        """Variance of KM estimator from Greenwood's formula"""
        return (1. * deaths / (population * (population - deaths))).replace([np.inf], 0)
    def Usurv(N,t,inds,nsamp,time,sa,t1,t2):
        """Sub-functions:
        Parzen, Wei and Ying: Simultaneous Confidence Bands for the Difference of Two Survival Functions (SJS, 1997)"""

        s = np.zeros(N)

        for j in inds:

            x = len(time[time >= time[j]])

            atrisk = ifelse(x > 0, 1/x, 0)

            s += (ifelse(time[j] >= t1 & time[j] <= t2,1,0)*atrisk * ifelse(time[j] <= t,1,0) * rnorm(N))

        return -(nsamp**(0.5)) * sa * s 

    def _Vtildesurv(N,i,t,inds1,inds2,nsamp,nsamp1,nsamp2,time1,time2,sa1,sa2,varsa1,varsa2,t1,t2,g1surv,g2surv,vargsurv):
        tmpU2 = Usurv(N,t,inds2,nsamp,time2,sa2[i],t1,t2)
        tmpU1 = Usurv(N,t,inds1,nsamp,time1,sa1[i],t1,t2)
        return vargsurv[i]*(g2surv(sa1[i],sa2[i]) * tmpU2 + g1surv(sa1[i],sa2[i]) * tmpU1)

    def Gtildesurv(N,inds1,inds2,nsamp,nsamp1,nsamp2,timesunique,time1,time2,delta1,delta2,sa1,sa2,varsa1,varsa2,lenunique,t1,t2,g1surv,g2surv,vargsurv):
        mx = np.zeros(N)

        for i in range(len(timesunique)):

            tt = timesunique[i]

            x = np.abs(Vtildesurv(N,i,tt,inds1,inds2,nsamp,nsamp1,nsamp2,time1,time2,sa1,sa2,varsa1,varsa2,t1,t2,g1surv,g2surv,vargsurv))

            mx = ifelse(x > mx, x, mx)

        return mx


    def _critvaluesurv(alpha, N, inds1, inds2, nsamp, nsamp1, nsamp2, timesunique, time1, time2, delta1,delta2,sa1,sa2,varsa1,varsa2,lenunique,t1,t2,g1surv,g2surv,vargsurv):
        Gtildevect = Gtildesurv(N, inds1, inds2, nsamp, nsamp1, nsamp2, timesunique,time1,time2,delta1,delta2,sa1,sa2,varsa1,varsa2,lenunique,t1,t2,g1surv,g2surv,vargsurv)
        return sort(Gtildevect)[np.floor((1-alpha)*N)]

    
    criticalz = -stats.norm.ppf(alpha/2)

    ind1 = df[treatment_col] == 0
    naf1 = NelsonAalenFitter()
    naf1.fit(durations=df.loc[ind1, duration_col], event_observed=df.loc[ind1, event_col])

    
    deaths = naf1.event_table['observed']
    population = naf1.event_table['entrance'].cumsum() - naf1.event_table['removed'].cumsum().shift(1).fillna(0)  # slowest line here.
    varsa1 = np.cumsum(_additive_var(population, deaths))
    varsa1 = varsa1.reindex(timeline, method='pad')
    varsa1.index.name = 'timeline'

    
    kmf1 = KaplanMeierFitter()
    kmf1.fit(durations=df.loc[ind1, duration_col], event_observed=df.loc[ind1, event_col])
    nsamp1 = ind1.sum()
    sa1 = np.exp(-naf1.cumulative_hazard_)

    ind2 = df[treatment_col] == 1
    naf2 = NelsonAalenFitter()
    naf2.fit(durations=df.loc[ind2, duration_col], event_observed=df.loc[ind2, event_col])
    nsamp2 = ind2.sum()
    oldsa1 = np.exp(-naf1.cumulative_hazard_)

    nsamp = nsamp1 + nsamp2

    """Compute the reciprocal of the standard error of gsurv(x,y) [the vtilde function in Parzen et al.]
    analytic variance calculation (the default, performed in any case)"""
    vargsurv = _vsurv(nsamp1, nsamp2, sa1, sa2, varsa1, varsa2)
    vargsurv2 = _vsurv2(nsamp1, nsamp2, sa1, sa2, varsa1, varsa2)

    pointests = gsurv(sa1,sa2)
    vval  =  vsurv(nsamp1, nsamp2, sa1, sa2, varsa1, varsa2) 
    vval2 = vsurv2(nsamp1, nsamp2, sa1, sa2, varsa1, varsa2)

    lowint = pointests - criticalz/vval2
    upint = pointests + criticalz/vval2

    """'critvalband' is a vector with 3 components, each being a critical value for construction of
    simultaneous CI at (1-alpha)*100% confidence level"""
    critvalband = Critvaluesurv(alpha,N,jumpinds1,jumpinds2,nsamp,nsamp1,nsamp2,timesunique,time1,time2,delta1,delta2,sa1,sa2,varsa1,varsa2,lenunique,t1,t2,g1surv,g2surv,vargsurv)
    critvalband95 = critvalband[0]
    critvalband90 = critvalband[1]
    critvalband80 = critvalband[2]

    lowband95 = pointests - ((nsamp**(-1/2))*critvalband95)/vval
    upband95 =  pointests + ((nsamp**(-1/2))*critvalband95)/vval
  
    pointests = 1 - np.exp(pointests)
    lowint = 1 - np.exp(upint)
    upint = 1 - np.exp(lowint)    

    idx1 = np.where(naf1.timeline <= followup)[0][-1]
    idx2 = np.where(naf2.timeline <= followup)[0][-1]
    te = 1 - naf2.cumulative_hazard_.iloc[idx2,0] / naf1.cumulative_hazard_.iloc[idx1,0]
    
    ci = np.array([np.nan, np.nan])
    pvalue = np.nan
    
    return te, ci, pvalue

def estTEPH(df, treatment_col='treated', duration_col='dx2', event_col='disease2',covars=[]):
    """Estimates treatment efficacy using proportional hazards (Cox model).
    
    Parameters
    ----------
    df : pandas.DataFrame
    
    treatment_col : string
        Column in df indicating treatment.
    duration_col : string
        Column in df indicating survival times.
    event_col : string
        Column in df indicating events (censored data are 0)
    covars : list
        List of other columns to include in Cox model as covariates.
    
    Returns
    -------
    est : float
        Estimate of vaccine efficacy
    ci : vector, length 2
        95% confidence interval, [LL, UL]
    pvalue : float
        P-value for H0: VE=0"""
    
    coxphf = CoxPHFitter()
    
    coxphf.fit(df[[treatment_col, duration_col, event_col]+covars],duration_col = duration_col,event_col = event_col)
    
    te = 1-exp(coxphf.hazards_.loc['coef',treatment_col])
    ci = 1-exp(coxphf.confidence_intervals_[treatment_col].loc[['upper-bound','lower-bound']])
    pvalue = coxphf._compute_p_values()[0]
    return te,ci,pvalue

def scoreci(x, n, conf_level=0.95):
    """Wilson's confidence interval for a single proportion.
    Score CI based on inverting the asymptotic normal test
    using the null standard error

    Wilson, E.B. (1927) Probable inference, the law of succession, and statistical inference
    J. Amer. Stat. Assoc 22, 209-212

    Parameters
    ----------
    x : int
        Number of events
    n : int
        Number of trials/subjects
    conf_level : float
        Specifies coverage of the confidence interval (1 - alpha)

    Returns
    -------
    ci : array
        Confidence interval array [LL, UL]"""

    zalpha = abs(stats.norm.ppf((1-conf_level)/2))
    phat = x/n
    bound = (zalpha*((phat*(1-phat)+(zalpha**2)/(4*n))/n)**(1/2))/(1+(zalpha**2)/n)
    midpnt = (phat+(zalpha**2)/(2*n))/(1+(zalpha**2)/n)

    uplim = midpnt + bound
    lowlim = midpnt - bound

    return array([lowlim, uplim])

def unconditionalVE(nv,Nv, np, Np, alpha=0.025):
    """VE point-estimate, CI and p-value, without conditioning on the total number of events"""
    rr = (nv/(Nv)) / (np/(Np))

    ve = 1 - rr
    se = sqrt((Nv-nv)/(nv*Nv) + (Np-np)/(np*Np))

    z = stats.norm.ppf(1 - alpha)

    ci = 1 - exp(array([log(rr) + se*z, log(rr) - se*z]))
    
    """Wald CI"""
    pvalue = stats.norm.cdf(log(rr)/se)

    return  pd.Series([ve, ci[0], ci[1], pvalue], index=['VE','LL', 'UL','p'])

def AgrestiScoreVE(nv,Nv, np, Np, alpha=0.025):
    """Conditional test based on a fixed number of events,
    n = nv + np
    phat = nv/n"""
    def scoreCIbinProp(pHat, n, alpha):
        """Score confidence interval for binomial proportion following Agresti and Coull (Am Statistician, 1998)"""
        z = stats.norm.ppf(1-alpha)
        return ((pHat + z**2/(2*n) + z*sqrt((pHat*(1-pHat)+z**2/(4*n))/n))/(1+z**2/n),
                (pHat + z**2/(2*n) - z*sqrt((pHat*(1-pHat)+z**2/(4*n))/n))/(1+z**2/n))

    veFunc = lambda pvhat, Nv, Np: 1 - (Np/Nv)*(pvhat/(1-pvhat))
    rr = (nv/Nv)/(np/Np)
    ve = 1 - rr
    n = nv + np
    pvhat = nv/n

    ll,ul = scoreCIbinProp(pvhat, n, alpha=alpha)
    
    
    ve = veFunc(pvhat, Nv, Np)
    ci = veFunc(ll, Nv, Np), veFunc(ul, Nv, Np)
    p = stats.binom.cdf(nv,n,Nv/(Nv+Np))
    
    return  pd.Series([ve, ci[0], ci[1], p], index=['VE','LL', 'UL','p'])