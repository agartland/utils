import pandas as pd
import numpy as np
from scipy import stats

import numba

try:
    from lifelines import KaplanMeierFitter, NelsonAalenFitter
    from lifelines.statistics import logrank_test
    from lifelines import CoxPHFitter
except ModuleNotFoundError:
    print('Module "lifelines" could not be imported.')

__all__ = [ 'na_est',
            'estimate_cumulative_incidence',
            'estimate_cumulative_incidence_ratio',
            'estimate_cumulative_incidence_difference',
            'cumulative_contrast',
            'estCoxPHTE',
            'scoreci',
            'AgrestiScoreVE',
            'unconditionalVE',
            'diffscoreci',
            'riskscoreci',
            'binpropci_katz',
            'binprop_pvalue']

"""Cumulative incidence estimates/CIs and CIR estimates/CIs match
those obtained using R code from Michal Juraska, Erika Rudnicki and Doug Grove"""

@numba.jit(nopython=True, parallel=True, error_model='numpy')
def na_est(T, event, times):
    """Results match R survit with
    "fleming-harrington" estimator and "tsiatis" variance"""
    # sorti = np.argsort(T)
    # T = T[sorti]
    # event = event[sorti]

    N = T.shape[0]
    # uT = np.unique(T)
    uT = np.sort(np.unique(np.concatenate((T, times))))

    T_count = np.zeros(len(uT))
    event_count = np.zeros(len(uT))
    for i in range(len(uT)):
        ind = T == uT[i]
        T_count[i] = np.sum(ind)
        event_count[i] = np.sum(event[ind])
    at_risk = N - np.cumsum(T_count) + T_count

    cumhaz = np.cumsum(event_count / at_risk)
    """Variance estimator recommended in Ornulf Borgan paper on NA"""
    #cumhaz_var = np.cumsum(((at_risk - event_count) * event_count) / ((at_risk - 1) * at_risk**2))
    
    """Used by SAS LIFETEST and matches R survfit error='tsiatis' """
    cumhaz_var = np.cumsum(event_count / at_risk**2)

    """Only return at requested times"""
    cumhaz_out = np.zeros(len(times))
    var_out = np.zeros(len(times))
    atrisk_out = np.zeros(len(times))
    events_out = np.zeros(len(times))
    for i in range(len(times)):
        ix = np.where(uT == times[i])[0][0]
        cumhaz_out[i] = cumhaz[ix]
        var_out[i] = cumhaz_var[ix]
        atrisk_out[i] = at_risk[ix]
        events_out[i] = event_count[ix]
    return times, cumhaz_out, var_out, atrisk_out, events_out

@numba.jit(nopython=True, parallel=True, error_model='numpy')
def _CIR_est(treatment, T, event, add_times):
    """Replaced by cumulative contrast"""
    tvec = np.unique(T)
    if len(add_times) > 0:
        tvec = np.unique(np.concatenate((tvec, add_times)))

    ind = treatment == 1
    t_cmp, cumhaz_cmp, cumhaz_var_cmp, atrisk_cmp, events_cmp = na_est(T[ind], event[ind], tvec)
    t_ref, cumhaz_ref, cumhaz_var_ref, atrisk_ref, events_ref = na_est(T[~ind], event[~ind], tvec)

    cuminc_cmp = 1 - np.exp(-cumhaz_cmp)
    cuminc_ref = 1 - np.exp(-cumhaz_ref)

    se_cuminc_cmp = np.sqrt(cumhaz_var_cmp) * np.exp(-cumhaz_cmp)
    se_cuminc_ref = np.sqrt(cumhaz_var_ref) * np.exp(-cumhaz_ref)

    logCIR = np.log( cuminc_cmp ) - np.log( cuminc_ref )
    """Matches Juraska code"""
    se_logCIR = np.sqrt( (se_cuminc_cmp / cuminc_cmp)**2 + (se_cuminc_ref / cuminc_ref)**2 )

    """Replace NaN caused by zeros in above step"""
    cuminc0 = (cuminc_cmp == 0) | (cuminc_ref == 0)  
    logCIR[cuminc0] = np.nan
    se_logCIR[cuminc0] = np.nan

    return tvec, logCIR, se_logCIR, cumhaz_ref, cumhaz_var_ref, cumhaz_cmp, cumhaz_var_cmp

@numba.jit(nopython=True, parallel=True, error_model='numpy')
def _CID_est(treatment, T, event, add_times):
    """Replaced by cumulative contrast"""
    """Cumulative incidence difference estimation"""
    tvec = np.unique(T)
    if len(add_times) > 0:
        tvec = np.unique(np.concatenate((tvec, add_times)))

    ind = treatment == 1
    t_cmp, cumhaz_cmp, cumhaz_var_cmp, atrisk_cmp, events_cmp = na_est(T[ind], event[ind], tvec)
    t_ref, cumhaz_ref, cumhaz_var_ref, atrisk_ref, events_ref = na_est(T[~ind], event[~ind], tvec)

    cuminc_cmp = 1 - np.exp(-cumhaz_cmp)
    cuminc_ref = 1 - np.exp(-cumhaz_ref)

    se_cuminc_cmp = np.sqrt(cumhaz_var_cmp) * np.exp(-cumhaz_cmp)
    se_cuminc_ref = np.sqrt(cumhaz_var_ref) * np.exp(-cumhaz_ref)

    cid = cuminc_cmp - cuminc_ref
    se_cid = np.sqrt( se_cuminc_cmp**2 + se_cuminc_ref**2 )

    chd = cumhaz_cmp - cumhaz_ref
    se_chd = np.sqrt( cumhaz_var_cmp + cumhaz_var_ref )

    return tvec, cid, se_cid, chd, se_chd, cumhaz_ref, cumhaz_var_ref, cumhaz_cmp, cumhaz_var_cmp

@numba.jit(nopython=True, parallel=True, error_model='numpy')
def cumulative_contrast(treatment, T, event, add_times, weights, log_ratio=True, cuminc=True):
    """Compute contrast between two groups using either cumulative incidence (cuminc=1, default)
    or cumulative hazard (cuminc=0).

    Contrast can be either a ratio returned on the log-scale (log_ratio=1):
        log[Pr(T<t | treatment=1) / Pr(T<t | treatment=0)]
    or a difference on the additive scale:
        Pr(T<t | treatment=1) - Pr(T<t | treatment=0)

    Parameters
    ----------
    treatment : np.ndarray shape(n)
        Vector of binary treatment assignments
    T : np.ndarray shape(n)
        Vector of event times/durations
    event : np.ndarray shape(n)
        Vector of binary event indicators
    add_times : np.ndarray
        Vector of additional times at which the contrast should be computed
    weights : np.ndarray shape(n)
        Vector of weights for each subject. Sets of unique weights indicate
        strata for a stratafied analysis. Unique weights must sum to 1.
        Pass vector of ones to perform an nonstratified analysis.
    log_ratio : int (1 or 0)
        Indicator of whether the contrast should be a ratio returned on a log-scale or
        a difference on the additive scale.
    cuminc : int (1 or 0)
        Indicator of whether the contrast should be compmuted on cumulative incidences

    Returns
    -------
    tvec : np.ndarray shape(n)
        Vector of times at which the contrast was estimated.
        May be longer than input data if add_times are specified.
    contrast : np.ndarray shape(n)
        Vector of estimated contrasts at each time
    se_contrast : np.ndarray shape(n)
        Vector of estimated contrast SE at each time
    cumx : np.ndarray shape(n, 2)
        Matrix of cumulative incidences/hazards for the reference [:, 0] and
        the comparator [:, 1] groups.
    se_cumx : np.ndarray shape(n, 2)
        Matrix of cumulative incidence/hazard SEs for the reference [:, 0] and
        the comparator [:, 1] groups.

    """
    tvec = np.unique(T)
    if len(add_times) > 0:
        tvec = np.unique(np.concatenate((tvec, add_times)))

    uw = np.unique(weights)
    if not np.abs(np.sum(uw) - 1) < 0.00001:
        """Check that the unique weights sum to 1"""
        # raise ValueError('Weights do not sum to 1')
        new_uw = uw / np.sum(uw)
        for i in range(len(uw)):
            weights[weights == uw[i]] = new_uw[i]
        uw = new_uw

    shp = (len(tvec), 2, len(uw))
    t = np.zeros(shp)
    s_cumhaz = np.zeros(shp)
    s_cumhaz_var = np.zeros(shp)
    s_atrisk = np.zeros(shp)
    s_events = np.zeros(shp)
    for i, w in enumerate(uw):
        """Split the NA estimates by strata and group indicators.
        Strata levels are indicated by a unique weight."""
        ind_cmp = (treatment == 1) & (weights == w)
        ind_ref = (treatment == 0) & (weights == w)
        t[:, 0, i], s_cumhaz[:, 0, i], s_cumhaz_var[:, 0, i], s_atrisk[:, 0, i], s_events[:, 0, i] = na_est(T[ind_ref], event[ind_ref], tvec)
        t[:, 1, i], s_cumhaz[:, 1, i], s_cumhaz_var[:, 1, i], s_atrisk[:, 1, i], s_events[:, 1, i] = na_est(T[ind_cmp], event[ind_cmp], tvec)

    """Take strata-weighted sums of the cumulative hazard and the SE"""
    cumhaz = np.sum(s_cumhaz * np.reshape(uw, (1, 1, len(uw))), axis=2)
    cumhaz_var = np.sum((np.sqrt(s_cumhaz_var) * np.reshape(uw, (1, 1, len(uw))))**2, axis=2)

    if cuminc:
        """Translate form cumhaz to cuminc scale"""
        cumx = 1 - np.exp(-cumhaz)
        se_cumx = np.sqrt(cumhaz_var) * np.exp(-cumhaz)
    else:
        cumx = cumhaz
        se_cumx = np.sqrt(cumhaz_var)
        
    if log_ratio:
        """Compute ratio, returning contrast and SE on the log-scale"""
        contrast = np.log( cumx[:, 1] ) - np.log( cumx[:, 0] )
        se_contrast = np.sqrt( (se_cumx[:, 1] / cumx[:, 1])**2 + (se_cumx[:, 0] / cumx[:, 0])**2 )

        """Replace NaN caused by zeros in above step"""
        cumx0 = (cumx[:, 1] == 0) & (cumx[:, 0] == 0)  
        contrast[cumx0] = np.nan
        se_contrast[cumx0] = np.nan
    else:
        """Compute difference, returning on the original scale"""
        contrast = cumx[:, 1] - cumx[:, 0]
        se_contrast = np.sqrt(np.sum(se_cumx**2, axis=1))

    return tvec, contrast, se_contrast, cumx, se_cumx

def estimate_cumulative_incidence(durations, events, times=None, weights=None, alpha=0.05):
    if times is None:
        times = np.unique(durations)
    if not weights is None:
        uw = np.unique(weights)
        if not np.abs(np.sum(uw) - 1) < 0.00001:
            """Check that the unique weights sum to 1"""
            # raise ValueError('Weights do not sum to 1')
            new_uw = uw / np.sum(uw)
            for i in range(len(uw)):
                weights[weights == uw[i]] = new_uw[i]
            uw = new_uw
        
        shp = (len(times), len(uw))
        t = np.zeros(shp)
        s_cumhaz = np.zeros(shp)
        s_cumhaz_var = np.zeros(shp)
        s_atrisk = np.zeros(shp)
        s_events = np.zeros(shp)
        for i,w in enumerate(uw):
            """Split the NA estimates by strata 
            Strata levels are indicated by a unique weight"""
            ind = weights == w
            t[:, i], s_cumhaz[:, i], s_cumhaz_var[:, i], s_atrisk[:, i], s_events[:, i] = na_est(np.asarray(durations)[ind],
                                                                                                 np.asarray(events)[ind],
                                                                                                 np.asarray(times))
        """Take strata-weighted sums of the cumulative hazard and the SE"""
        ch = np.sum(s_cumhaz * np.reshape(uw, (1, len(uw))), axis=1)
        cumhaz_var = np.sum((np.sqrt(s_cumhaz_var) * np.reshape(uw, (1, len(uw))))**2, axis=1)
        """Get the atrisk and events_vec from unweighted na_est"""
        tvec, _, _, atrisk, events_vec = na_est(np.asarray(durations), np.asarray(events), np.asarray(times))
    else:
        tvec, ch, cumhaz_var, atrisk, events_vec = na_est(np.asarray(durations), np.asarray(events), np.asarray(times))
    
    criticalz = -stats.norm.ppf(alpha / 2)
    """Matches R survfit"""
    lcl = ch - criticalz * np.sqrt(cumhaz_var)
    ucl = ch + criticalz * np.sqrt(cumhaz_var)

    cuminc_lcl = 1 - np.exp(-lcl)

    """Enforce positive LCL"""
    lcl[lcl < 0] = 0
    cuminc_lcl[cuminc_lcl < 0] = 0

    out = pd.DataFrame(dict(atrisk=atrisk,
                            events=events_vec,
                            cumhaz = ch,
                            se_cumhz = np.sqrt(cumhaz_var),
                            cumhaz_lcl = lcl,
                            cumhaz_ucl = ucl,
                            cuminc = 1 - np.exp(-ch),
                            se_cuminc = np.sqrt(cumhaz_var) * np.exp(-ch),
                            cuminc_lcl = cuminc_lcl,
                            cuminc_ucl = 1 - np.exp(-ucl)), index=tvec)
    return out

def estimate_cumulative_incidence_ratio(treatment, durations, events, weights=None, alpha=0.05, cir0=1, add_times=[]):
    criticalz = -stats.norm.ppf(alpha / 2)

    if weights is None:
        weights = np.ones(len(events))

    tvec, logCIR, se_logCIR, cuminc, se_cuminc = cumulative_contrast(np.asarray(treatment), 
                                                                      np.asarray(durations),
                                                                      np.asarray(events),
                                                                      np.asarray(add_times),
                                                                      np.asarray(weights), log_ratio=True, cuminc=True)

    logCIR_lcl = logCIR - criticalz * se_logCIR
    logCIR_ucl = logCIR + criticalz * se_logCIR

    """Compute Wald statistic without log transformation:
    this is actually testing the difference of cumulative hazard functions"""
    # wald_stat = (cumhaz_cmp - cumhaz_ref) / np.sqrt(cumhaz_var_ref + cumhaz_var_cmp)
    # wald_pvalue = 2 * stats.norm.cdf(-np.abs(wald_stat))
    # print('Wald, no log: %1.3f, p = %1.3f' % (wald_stat[3], wald_pvalue[3]))

    """Compared to computing p-value for logCIR = 0"""
    wald_stat = (logCIR - np.log(cir0)) / se_logCIR
    wald_pvalue_cir = 2 * stats.norm.cdf(-np.abs(wald_stat))
    #print('Wald, log-CIR scale: %1.3g, p = %1.3g' % (wald_stat[3], wald_pvalue[3]))    

    """Compute Wald statistic on log-cumulative hazards"""
    """Variance of the log-CH function, by the delta method"""
    # log_cumhaz_var_ref = cumhaz_var_ref / cumhaz_ref**2
    # log_cumhaz_var_cmp = cumhaz_var_cmp / cumhaz_cmp**2
    # wald_stat = (np.log(cumhaz_cmp) - np.log(cumhaz_ref) - np.log(cir0)) / np.sqrt(log_cumhaz_var_ref + log_cumhaz_var_cmp)
    # wald_pvalue = 2 * stats.norm.cdf(-np.abs(wald_stat))
    # print('Wald, log-scale: %1.3g, p = %1.3g' % (wald_stat[3], wald_pvalue[3]))
    
    out = pd.DataFrame(dict(CIR = np.exp(logCIR),
                            se_logCIR = se_logCIR,
                            CIR_lcl = np.exp(logCIR_lcl),
                            CIR_ucl = np.exp(logCIR_ucl),
                            TE = 1 - np.exp(logCIR),
                            TE_lcl = 1 - np.exp(logCIR_ucl),
                            TE_ucl = 1 - np.exp(logCIR_lcl),
                            CIR_pvalue = wald_pvalue_cir), index=tvec)
    return out

def estimate_cumulative_incidence_difference(treatment, durations, events, weights=None, alpha=0.05, cid0=0, add_times=[]):
    criticalz = -stats.norm.ppf(alpha / 2)

    if weights is None:
        weights = np.ones(len(events))

    tvec, cid, se_cid, cuminc, se_cuminc = cumulative_contrast(np.asarray(treatment), 
                                                               np.asarray(durations),
                                                               np.asarray(events),
                                                               np.asarray(add_times),
                                                               np.asarray(weights), log_ratio=False, cuminc=True)
    cid_lcl = cid - criticalz * se_cid
    cid_ucl = cid + criticalz * se_cid

    wald_stat = (cid - cid0) / se_cid
    wald_pvalue_cid = 2 * stats.norm.cdf(-np.abs(wald_stat))

    out = pd.DataFrame(dict(cid = cid,
                            se_cid = se_cid,
                            cid_lcl = cid_lcl,
                            cid_ucl = cid_ucl,
                            cid_pvalue = wald_pvalue_cid), index=tvec)
    return out

def _estCumTE(df, treatment_col='treated', duration_col='dx', event_col='disease', followupT=None, alpha=0.05, H1=0, bootstraps=None):
    """Estimates treatment efficacy using cumulative incidence (CI) Nelson-Aalen (NA) estimators.

    REPLACED BY THE FUNCTIONS ABOVE

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

def scoreci(x, n, alpha=0.05):
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

    zalpha = np.abs(stats.norm.ppf(alpha/2))
    phat = x/n
    bound = (zalpha*((phat*(1-phat)+(zalpha**2)/(4*n))/n)**(1/2))/(1+(zalpha**2)/n)
    midpnt = (phat+(zalpha**2)/(2*n))/(1+(zalpha**2)/n)

    uplim = midpnt + bound
    lowlim = midpnt - bound

    return np.array([lowlim, uplim])

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

def AgrestiScoreVE(nv,Nv, np, Np, alpha=0.05):
    """Conditional test based on a fixed number of events,
    n = nv + np
    phat = nv/n"""

    def scoreci_binprop(pHat, n, alpha):
        """Score confidence interval for binomial proportion following Agresti and Coull (Am Statistician, 1998)"""
        z = stats.norm.ppf(1-alpha/2)
        lcl = (pHat + z**2/(2*n) + z*np.sqrt((pHat*(1-pHat)+z**2/(4*n))/n))/(1+z**2/n)
        ucl = (pHat + z**2/(2*n) - z*np.sqrt((pHat*(1-pHat)+z**2/(4*n))/n))/(1+z**2/n)
        return lcl, ucl

    veFunc = lambda pvhat, Nv, Np: 1 - (Np/Nv)*(pvhat/(1-pvhat))
    rr = (nv/Nv)/(np/Np)
    ve = 1 - rr
    n = nv + np
    pvhat = nv/n

    ll, ul = scoreci_binprop(pvhat, n, alpha=alpha)
    
    ve = veFunc(pvhat, Nv, Np)
    ci = veFunc(ll, Nv, Np), veFunc(ul, Nv, Np)
    p = stats.binom.cdf(nv, n, Nv/(Nv+Np))
    
    return  pd.Series([ve, ci[0], ci[1], p], index=['VE', 'LL', 'UL', 'p'])

def binpropci_katz(x1, n1, x2, n2, alpha=0.05):
    """Compute CI for the ratio of two binomial rates.
    Implements Katz method on the log-RR scale.

    Parameters
    ----------
    xi : int
        Number of events in group i
    ni : int
        Number of trials/subjects in group i
    alpha : float
        Specifies coverage of the confidence interval

    Returns
    -------
    ci : array
        Confidence interval array [LL, UL]"""
    z =  np.abs(stats.norm.ppf(alpha/2))
    a = x1
    b = n1 - x1
    c = x2
    d = n2 - x2
    
    rr = (x1 / n1) / (x2 / n2)

    se_logrr = np.sqrt(1/a + 1/c - 1/(a+b) - 1/(c+d))

    lcl = np.exp(np.log(rr) - z*se_logrr)
    ucl = np.exp(np.log(rr) + z*se_logrr)
    return np.array([lcl, ucl])

def riskscoreci(x1, n1, x2, n2, alpha=0.05, correction=True):
    """Compute CI for the ratio of two binomial rates.
    Implements the non-iterative method of Nam (1995).
    It has better properties than Wald/Katz intervals,
    especially with small samples and rare events.
    
    Translated from R-package 'PropCIs':
    https://github.com/shearer/PropCIs

    Nam, J. M. (1995) Confidence limits for the ratio of two binomial proportions based on likelihood
    scores: Non-iterative method. Biom. J. 37 (3), 375-379.
    
    Koopman PAR. (1984) Confidence limits for the ratio of two binomial proportions. Biometrics 40,
    513-517.
    
    Miettinen OS, Nurminen M. (1985) Comparative analysis of two rates. Statistics in Medicine 4,
    213-226.
    
    Nurminen, M. (1986) Analysis of trends in proportions with an ordinally scaled determinant. Biometrical
    J 28, 965-974
    
    Agresti, A. (2002) Categorical Data Analysis. Wiley, 2nd Edition.

    Parameters
    ----------
    xi : int
        Number of events in group i
    ni : int
        Number of trials/subjects in group i
    alpha : float
        Specifies coverage of the confidence interval
    correction : bool
        A corrected estimate of RR can be returned by adding 0.5 to each cell
        of the contingency table.

    Returns
    -------
    ci : array
        Confidence interval array [LL, UL, RR_est]"""

    if correction:
        rr_est = ((x1+0.5) / (n1+1)) / ((x2+0.5) / (n2+1))
    else:
        rr_est = (x1 / n1) / (x2 / n2)

    z =  np.abs(stats.norm.ppf(alpha/2))
    if x2==0 and x1 == 0:
        ul = np.inf
        ll = 0
    else:
        a1 =  n2*(n2*(n2+n1)*x1+n1*(n2+x1)*(z**2))
        a2 = -n2*(n2*n1*(x2+x1)+2*(n2+n1)*x2*x1+n1*(n2+x2+2*x1)*(z**2))
        a3 = 2*n2*n1*x2*(x2+x1)+(n2+n1)*(x2**2)*x1+n2*n1*(x2+x1)*(z**2)
        a4 = -n1*(x2**2)*(x2+x1)
        b1 = a2/a1
        b2 = a3/a1
        b3 = a4/a1
        c1 = b2-(b1**2)/3
        c2 = b3-b1*b2/3+2*(b1**3)/27
        ceta = np.arccos(np.sqrt(27)*c2/(2*c1*np.sqrt(-c1)))
        t1 = -2*np.sqrt(-c1/3)*np.cos(np.pi/3-ceta/3)
        t2 = -2*np.sqrt(-c1/3)*np.cos(np.pi/3+ceta/3)
        t3 = 2*np.sqrt(-c1/3)*np.cos(ceta/3)
        p01 = t1-b1/3
        p02 = t2-b1/3
        p03 = t3-b1/3
        p0sum = p01+p02+p03
        p0up = np.min([p01,p02,p03])
        p0low = p0sum-p0up-np.max([p01,p02,p03])

        if x2 == 0 and x1 != 0:
            ll = (1-(n1-x1)*(1-p0low)/(x2+n1-(n2+n1)*p0low))/p0low
            ul = np.inf
        elif x2 != n2 and x1==0:
            ul = (1-(n1-x1)*(1-p0up)/(x2+n1-(n2+n1)*p0up))/p0up
            ll = 0
        elif x2 == n2 and x1 == n1:
            ul = (n2+z**2)/n2
            ll =  n1/(n1+z**2)
        elif x1 == n1 or x2 == n2:
            if x2 == n2 and x1 == 0:
                ll = 0
            if x2 == n2 and x1 != 0:
                phat1  = x2/n2
                phat2  =  x1/n1
                phihat = phat2/phat1
                phil = 0.95*phihat
                chi2 = 0
                while chi2 <= z:
                    a = (n2+n1)*phil
                    b = -((x2+n1)*phil+x1+n2)
                    c = x2+x1
                    p1hat = (-b-np.sqrt(b**2-4*a*c))/(2*a)
                    p2hat = p1hat*phil
                    q2hat = 1-p2hat
                    var = (n2*n1*p2hat)/(n1*(phil-p2hat)+n2*q2hat)
                    chi2 = ((x1-n1*p2hat)/q2hat)/np.sqrt(var)
                    ll = phil
                    phil = ll/1.0001
            i = x2
            j = x1
            ni = n2
            nj = n1
            if x1 == n1:
                i = x1
                j = x2
                ni = n1
                nj = n2
            
            phat1  = i/ni
            phat2  =  j/nj
            phihat = phat2/phat1
            phiu = 1.1*phihat
            if x2 == n2 and x1 == 0:
                if n2<100:
                    phiu = .01
                else:
                    phiu = 0.001
            chi1 = 0
            while chi1 >= -z:
                a = (ni+nj)*phiu
                b = -((i+nj)*phiu+j+ni)
                c = i+j
                p1hat = (-b-np.sqrt(b**2-4*a*c))/(2*a)
                p2hat = p1hat*phiu
                q2hat = 1-p2hat
                var = (ni*nj*p2hat)/(nj*(phiu-p2hat)+ni*q2hat)
                chi1  = ((j-nj*p2hat)/q2hat)/np.sqrt(var)
                phiu1 = phiu
                phiu = 1.0001*phiu1

            if x1 == n1:
                ul = (1-(n1-x1)*(1-p0up)/(x2+n1-(n2+n1)*p0up))/p0up
                ll = 1/phiu1
            else:
                ul = phiu1
        else:
            ul = (1-(n1-x1)*(1-p0up)/(x2+n1-(n2+n1)*p0up))/p0up
            ll = (1-(n1-x1)*(1-p0low)/(x2+n1-(n2+n1)*p0low))/p0low
    return np.array([ll, ul, rr_est])

def diffscoreci(x1, n1, x2, n2, alpha):
    """Score interval for difference in proportions
    
    Method of Mee 1984 with Miettinen and Nurminen modification nxy / (nxy - 1), see Newcombe 1998

    Agresti, A. (2002) Categorical Data Analysis. Wiley, 2nd Edition.
    
    Mee, RW. (1984) Confidence bounds for the difference between two probabilities. Biometrics 40,
    1175-1176.
    
    Miettinen OS, Nurminen M. (1985) Comparative analysis of two rates. Statistics in Medicine 4,
    213-226.
    
    Nurminen, M. (1986) Analysis of trends in proportions with an ordinally scaled determinant. Biometrical
    J. 28, 965-974

    Parameters
    ----------
    xi : int
        Number of events in group i
    ni : int
        Number of trials/subjects in group i
    alpha : float
        Specifies coverage of the confidence interval (1 - alpha)

    Returns
    -------
    ci : array
        Confidence interval array [LL, UL]"""
   
    px = x1/n1
    py = x2/n2
    z = stats.chi2.ppf(1 - alpha,1)
    proot = px - py
    dp = 1 - proot
    niter = 1
    while niter <= 50:
        dp = 0.5 * dp
        up2 = proot + dp
        score = _z2stat(px,n1,py,n2,up2)
        if score < z:
            proot = up2
        niter = niter + 1
        if dp<0.0000001 or np.abs(z-score) < 0.000001:
            niter = 51
            ul = up2
    proot = px - py
    dp = 1 + proot
    niter = 1
    while niter <= 50:
        dp = 0.5 * dp
        low2 = proot - dp
        score = _z2stat(px,n1,py,n2,low2)
        if score < z:
            proot = low2
        niter = niter+1
        if dp<0.0000001 or np.abs(z-score)<0.000001:
            ll = low2
            niter = 51
    return np.array([ll, ul])

def _z2stat(p1x,nx,p1y,ny,dif):
    """Private function used by diffscoreci"""
    difference = p1x-p1y-dif
    if np.abs(difference) == 0:
        fmdifference = 0
    else:
        t = ny/nx
        a = 1+t
        b = -(1+ t + p1x + t*p1y + dif*(t+2))
        c = dif*dif + dif*(2*p1x + t +1) + p1x + t*p1y
        d = -p1x*dif*(1+dif)
        v = (b/a/3)**3 - b*c/(6*a*a) + d/a/2
        s = np.sqrt( (b/a/3)**2 - c/a/3)
        if v>0:
            u = s
        else:
            u = -s
        w = (np.pi + np.arccos(v/u**3))/3
        p1d = 2*u*np.cos(w) - b/a/3
        p2d = p1d - dif
        nxy = nx + ny
        var = (p1d*(1-p1d)/nx + p2d*(1-p2d)/ny) * nxy / (nxy - 1) ## added: * nxy / (nxy - 1)
        fmdifference = difference**2/var
    return fmdifference

def binomci(x, N, alpha=0.05, method='score'):
    """Return confidence interval on observing number of events in x
    given N trials (Agresti and Coull  2 sided 95% CI)
    Returns lower and upper confidence limits (lcl,ucl)

    Code has been checked against R binom package. "Score" was derived
    from the Agresti paper and is equivalent to Wilson (copied from the R package).
    From the paper this seems to be the best in most situations.

    A. Agresti, B. A. Coull, T. A. Statistician, N. May,
    Approximate Is Better than "Exact" for Interval Estimation of Binomial Proportions,
    52, 119–126 (2007)."""

    x = np.asarray(x)
    if isinstance(N, list):
        N = np.asarray(N)
    p = x/N
    z = stats.norm.ppf(1.-alpha/2.)
    if method == 'score':
        lcl = (p + (z**2)/(2*N) - z*np.sqrt((p*(1-p)+z**2/(4*N))/N)) / (1 + (z**2)/N)
        ucl = (p + (z**2)/(2*N) + z*np.sqrt((p*(1-p)+z**2/(4*N))/N)) / (1 + (z**2)/N)
    elif method == 'wilson':
        """p1 <- p + 0.5 * z2/n
            p2 <- z * sqrt((p * (1 - p) + 0.25 * z2/n)/n)
            p3 <- 1 + z2/n
            lcl <- (p1 - p2)/p3
            ucl <- (p1 + p2)/p3"""
        p1 = p + 0.5 * (z**2 / N)
        p2 = z * np.sqrt((p * (1 - p) + 0.25 * z**2/N)/N)
        p3 = 1 + z**2 / N
        lcl = (p1 - p2)/p3
        ucl = (p1 + p2)/p3
    elif method == 'agresti-coull':
        """.x <- x + 0.5 * z2
        .n <- n + z2
        .p <- .x/.n
        lcl <- .p - z * sqrt(.p * (1 - .p)/.n)
        ucl <- .p + z * sqrt(.p * (1 - .p)/.n)"""
        xtmp = x + 0.5 * z**2
        ntmp = N + z**2
        ptmp = xtmp / ntmp
        se = np.sqrt(ptmp * (1 - ptmp)/ntmp)
        lcl = ptmp - z * se
        ucl = ptmp + z * se
    elif method == 'exact':
        """Clopper-Pearson (1934)"""
        """ x1 <- x == 0
            x2 <- x == n
            lb <- ub <- x
            lb[x1] <- 1
            ub[x2] <- n[x2] - 1
            lcl <- 1 - qbeta(1 - alpha2, n + 1 - x, lb)
            ucl <- 1 - qbeta(alpha2, n - ub, x + 1)
            if(any(x1)) lcl[x1] <- rep(0, sum(x1))
            if(any(x2)) ucl[x2] <- rep(1, sum(x2))"""
        lb = x.copy()
        ub = x.copy()
        lb[x == 0] = 1
        ub[x == N] = N - 1

        lcl = 1 - stats.beta.ppf(1 - alpha/2, N + 1 - x, lb)
        ucl = 1 - stats.beta.ppf(alpha/2, N - ub, x + 1)

        lcl[x == 0] = 0
        ucl[x == N] = 1
    elif method == 'wald':
        se = np.sqrt(p*(1-p)/N)
        ucl = p + z * se
        lcl = p - z * se
    return lcl, ucl

def binprop_pvalue(x1, n1, x2, n2, rr0=1):
    """Use chi2 test which is consistent with the riskscore
    derived confidence interval of Nam and Koopman.
    Produces a two-sided p-value in that it is equally and symetrically
    sensitive to deviations from rr0 in either direction.
    
    Parameters
    ----------
    xi : int
        Number of events in group i
    ni : int
        Number of trials/subjects in group i
    rr0 : float
        Null-hypothesis of RR = (x1 / n1) / (x2 / n2)

    Returns
    -------
    pvalue : float"""

    obs = np.array([x1, n1 - x1, x2, n2 - x2])

    """Under the null-hypothesis, n1 and n2 remain constant,
    as do the total number of events (x1 + x2) and overall p
    x1_prime and x2_prime are the distribution of events under the null,
    as a function of these constants n1, n2, and x1 + x2 = x1_prime + x2_prime"""
    x1_prime = (n1 * (x1 + x2)) / (n2 / rr0 + n1)
    x2_prime = x1 + x2 - x1_prime
    ex = np.array([x1_prime, n1-x1_prime, x2_prime, n2-x2_prime])

    """Expression for the case when RR0 = 1: it matches"""
    # ex = np.array([exp_p*n1, n1 - exp_p*n1, exp_p*n2, n2 - exp_p*n2])

    chi2 = np.sum((obs - ex)**2 / ex)

    num_obs = 4
    ddof = 2
    dof = num_obs - 1 - ddof
    pvalue = stats.chi2.sf(chi2, dof)
    
    """These are all consistent, except that only stats.chisquare allows for different RR0"""
    # chi2, pvalue = stats.chisquare(obs, f_exp=ex, ddof=ddof)
    # chi2, pvalue, dof, ex = stats.chi2_contingency([[x1, n1 - x1], [x2, n2 - x2]], correction=False)
    return chi2, pvalue, dof, ex