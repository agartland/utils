import pandas as pd
import numpy as np
from lifelines.estimation import KaplanMeierFitter, NelsonAalenFitter
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter
from myfisher import *
from scipy import stats

__all__ = ['estCumTE',
            'estCoxPHTE',
            'scoreci',
            'AgrestiScoreVE',
            'unconditionalVE']

def estCumTE(df, treatment_col='treated', duration_col='dx', event_col='disease', followupT=None, alpha=0.05, H1=0, bootstraps=None):
    """Estimates treatment efficacy using cumulative incidence (CI) Nelson-Aalen (NA) estimators.

    TODO:
        (1) Base the p-value and confidence intervals on the NA variance estimator (instead of KM)
        (2) Test different H1/alternative hypotheses
    
    TE = 1 - RR
    
    RR = CI_NA1 / CI_NA2

    P-value tests the hypothesis:
        H0: TE = H1
    
    Status
    ------
    Point estimates and pointe-wise confidence intervals match those from R
    Bootstrap and analytic point-wise confidence intervals are not correct.
    Need to add a pvalue to final timepoint.
    Simultaneous confidence bands are close, but commented out for now.

    Parameters
    ----------
    df : pandas.DataFrame
        Each row is a participant.
    treatment_col : string
        Column in df indicating treatment (values 1 or 0).
    dx_col : string
        Column in df indicating time of the event or censoring.
    event_col : string
        Column in df indicating events (values 1 or 0 with censored data as 0).
    followupT : float (optional)
        Follow-up time inlcuded in the anlaysis
        (also therefore the time at which a p-value is computed)
    H1 : float
        Alternative hypothesis for p-value on the fractional TE scale.
    bootstraps : int or None (optional)
        If not None, then confidence interval and p-value are estimated using
            a bootstrap approach with nstraps.

    Returns
    -------
    resDf : pd.DataFrame
        Estimate of treatment efficacy with (1-alpha)% confidence intervals.
        A p-value is included for the last timepoint only.
        columns: TE, UB, LB, pvalue"""

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

    def _additive_var(population, deaths):
        """Variance of KM estimator from Greenwood's formula"""
        return (1. * deaths / (population * (population - deaths))).replace([np.inf], 0)
    def _estimateSurv(df, ind):
        naf = NelsonAalenFitter()
        naf.fit(durations=df.loc[ind, duration_col], event_observed=df.loc[ind, event_col])
        
        """Borrowed from lifelines"""
        timeline = sorted(naf.timeline)
        deaths = naf.event_table['observed']
        """Slowest line here."""
        population = naf.event_table['entrance'].cumsum() - naf.event_table['removed'].cumsum().shift(1).fillna(0)
        varsa = np.cumsum(_additive_var(population, deaths))
        varsa = varsa.reindex(timeline, method='pad')
        varsa.index.name = 'timeline'
        varsa.name = 'surv_var'
        
        sa = np.exp(-naf.cumulative_hazard_.iloc[:, 0])
        sa.name = 'surv'
        return naf, sa, varsa
    def _alignTimepoints(x1, x2):
        new_index = np.concatenate((x1.index, x2.index))
        new_index = np.unique(new_index)
        return x1.reindex(new_index, method='ffill'), x2.reindex(new_index, method='ffill')
    def _vval2ByBootstrap(timeline, nstraps=1000):
        sa1_b, sa2_b = np.zeros((timeline.shape[0], nstraps)), np.zeros((timeline.shape[0], nstraps))
        for sampi in range(nstraps):
            tmp = df.sample(frac=1, replace=True, axis=0)

            ind1 = tmp[treatment_col] == 0
            naf1 = NelsonAalenFitter()
            naf1.fit(durations=tmp.loc[ind1, duration_col], event_observed=tmp.loc[ind1, event_col])
            sa1 = np.exp(-naf1.cumulative_hazard_.iloc[:, 0])
            sa1 = sa1.reindex(timeline, method='ffill')
            sa1_b[:, sampi] = sa1.values
            
            ind2 = df[treatment_col] == 1
            naf2 = NelsonAalenFitter()
            naf2.fit(durations=tmp.loc[ind2, duration_col], event_observed=tmp.loc[ind2, event_col])
            sa2 = np.exp(-naf2.cumulative_hazard_.iloc[:, 0])
            sa2 = sa2.reindex(timeline, method='ffill')
            sa2_b[:, sampi] = sa2.values
        vval2 = 1/np.sqrt(np.nanvar(np.log(sa1_b), axis=1) + np.nanvar(np.log(sa2_b), axis=1))
        return vval2

        
    '''
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

    def _Gtildesurv(N,inds1,inds2,nsamp,nsamp1,nsamp2,timesunique,time1,time2,delta1,delta2,sa1,sa2,varsa1,varsa2,lenunique,t1,t2,g1surv,g2surv,vargsurv):
        mx = np.zeros(N)
        for i in range(len(timesunique)):
            tt = timesunique[i]
            x = np.abs(Vtildesurv(N,i,tt,inds1,inds2,nsamp,nsamp1,nsamp2,time1,time2,sa1,sa2,varsa1,varsa2,t1,t2,g1surv,g2surv,vargsurv))
            mx = ifelse(x > mx, x, mx)
        return mx
    def _critvaluesurv(alpha, N, inds1, inds2, nsamp, nsamp1, nsamp2, timesunique, time1, time2, delta1,delta2,sa1,sa2,varsa1,varsa2,lenunique,t1,t2,g1surv,g2surv,vargsurv):
        Gtildevect = Gtildesurv(N, inds1, inds2, nsamp, nsamp1, nsamp2, timesunique,time1,time2,delta1,delta2,sa1,sa2,varsa1,varsa2,lenunique,t1,t2,g1surv,g2surv,vargsurv)
        return sort(Gtildevect)[np.floor((1-alpha)*N)]
    '''
    
    criticalz = -stats.norm.ppf(alpha/2)

    ind1 = df[treatment_col] == 0
    nsamp1 = ind1.sum()
    naf1, sa1, varsa1 = _estimateSurv(df, ind1)

    ind2 = df[treatment_col] == 1
    nsamp2 = ind2.sum()
    naf2, sa2, varsa2 = _estimateSurv(df, ind2)

    #acumh1, acumh2 = _alignTimepoints(naf1.cumulative_hazard_, naf2.cumulative_hazard_)
    asa1, asa2 = _alignTimepoints(sa1, sa2)
    avarsa1, avarsa2 = _alignTimepoints(varsa1, varsa2)

    if not followupT is None:
        keepInd = asa1.index <= followupT
        asa1 = asa1.loc[keepInd]
        asa2 = asa2.loc[keepInd]
        avarsa1 = avarsa1.loc[keepInd]
        avarsa2 = avarsa2.loc[keepInd]

    if bootstraps is None:
        """Compute the reciprocal of the standard error of gsurv(x,y) [the vtilde function in Parzen et al.]
        analytic variance calculation (the default, performed in any case)"""
        #vval = _vsurv(nsamp1, nsamp2, asa1, asa2, avarsa1, avarsa2)
        vval2 = _vsurv2(nsamp1, nsamp2, asa1, asa2, avarsa1, avarsa2)
    else:
        vval2 = _vval2ByBootstrap(asa1.index, nstraps=bootstraps)

    pointests = _gsurv(asa1, asa2)
    lowint = pointests - criticalz/vval2
    upint = pointests + criticalz/vval2

    '''
    """'critvalband' is a vector with 3 components, each being a critical value for construction of
    simultaneous CI at (1-alpha)*100% confidence level"""
    critvalband = Critvaluesurv(alpha,N,jumpinds1,jumpinds2,nsamp,nsamp1,nsamp2,timesunique,time1,time2,delta1,delta2,sa1,sa2,varsa1,varsa2,lenunique,t1,t2,g1surv,g2surv,vargsurv)
    critvalband95 = critvalband[0]
    critvalband90 = critvalband[1]
    critvalband80 = critvalband[2]

    lowband95 = pointests - ((nsamp**(-1/2))*critvalband95)/vval
    upband95 =  pointests + ((nsamp**(-1/2))*critvalband95)/vval
    '''
    pointests = 1 - np.exp(pointests)
    lowint = 1 - np.exp(lowint)
    upint = 1 - np.exp(upint)

    resDf = pd.concat((pointests, lowint, upint), axis=1, ignore_index=True)
    resDf.columns = ['TE', 'UB', 'LB']
    
    pvalues = np.nan * np.zeros(resDf.shape[0])

    # avarsa1, avarsa2
    wald_stat = (asa1 - asa2) / np.sqrt(avarsa1 + avarsa2)
    wald_pvalue = 2 * stats.norm.cdf(-np.abs(wald_stat))
    resDf['pvalue'] = wald_pvalue
    return resDf

def estCoxPHTE(df, treatment_col='treated', duration_col='dx', event_col='disease', covars=[]):
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
    
    coxphf.fit(df[[treatment_col, duration_col, event_col]+covars], duration_col=duration_col, event_col=event_col)
    
    te = 1 - np.exp(coxphf.hazards_.loc['coef', treatment_col])
    ci = 1 - np.exp(coxphf.confidence_intervals_[treatment_col].loc[['upper-bound', 'lower-bound']])
    pvalue = coxphf._compute_p_values()[0]

    ind1 = df[treatment_col] == 0
    ind2 = df[treatment_col] == 1
    results = logrank_test(df[duration_col].loc[ind1], df[duration_col].loc[ind2], event_observed_A=df[event_col].loc[ind1], event_observed_B=df[event_col].loc[ind2])
    index = ['TE', 'UB', 'LB', 'pvalue', 'logrank_pvalue', 'model']
    return pd.Series([te, ci['upper-bound'], ci['lower-bound'], pvalue, results.p_value, coxphf], index=index)

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

    return  pd.Series([ve, ci[0], ci[1], pvalue], index=['VE', 'LL', 'UL', 'p'])

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

    ll, ul = scoreCIbinProp(pvhat, n, alpha=alpha)
    
    
    ve = veFunc(pvhat, Nv, Np)
    ci = veFunc(ll, Nv, Np), veFunc(ul, Nv, Np)
    p = stats.binom.cdf(nv, n, Nv/(Nv+Np))
    
    return  pd.Series([ve, ci[0], ci[1], p], index=['VE', 'LL', 'UL', 'p'])