import numpy as np
from scipy import stats
import pandas as pd
from fg_shared import *
import sys
from os.path import join as opj
import time

sys.path.append(opj(_git, 'utils'))
from imm_cor import binpropci_katz, riskscoreci, binprop_pvalue

def test_small_sample_rr_coverage(nsamples=10000, alpha=0.05):
    n1, n2 = 100, 100
    probs = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
    rrs = [1, 1.1, 1.5, 2, 3, 5, 10]
    res = []
    for p in probs:
        for rr in rrs:
            if p*rr < 1:
                grp2 = stats.binom.rvs(n2, p, size=nsamples)
                grp1 = stats.binom.rvs(n1, p*rr, size=nsamples)

                for x1, x2 in zip(grp1, grp2):
                    lcl_k, ucl_k = binpropci_katz(x1, n1, x2, n2, alpha=alpha)
                    lcl, ucl = riskscoreci(x1, n1, x2, n2, alpha=alpha)
                    chi2, pvalue_rr1, dof, ex = binprop_pvalue(x1, n1, x2, n2, rr0=1)
                    chi2, pvalue_rr, dof, ex = binprop_pvalue(x1, n1, x2, n2, rr0=rr)
                    tmp = {'N1':n1,'N2':n2,
                            'Prob':p,
                            'RR':rr,
                            'RRest':x1/x2,
                            'X1':x1,
                            'X2':x2,
                            'LCL_Katz':lcl_k,
                            'UCL_Katz':ucl_k,
                            'LCL_Nam':lcl,
                            'UCL_Nam':ucl,
                            'pvalue_RR1':pvalue_rr1,
                            'pvalue_RR':pvalue_rr}
                    res.append(tmp)
    res = pd.DataFrame(res)

    res = res.assign(lcl_Katz_cov=res['LCL_Katz']<=res['RR'],
                     ucl_Katz_cov=res['UCL_Katz']>=res['RR'],
                     Katz_cov= (res['LCL_Katz']<=res['RR']) & (res['UCL_Katz']>=res['RR']),
                     lcl_Nam_cov=res['LCL_Nam']<=res['RR'],
                     ucl_Nam_cov=res['UCL_Nam']>=res['RR'],
                     Nam_cov= (res['LCL_Nam']<=res['RR']) & (res['UCL_Nam']>=res['RR']),
                     true_discovery=res['pvalue_RR1'] < alpha,
                     false_discovery=res['pvalue_RR'] < alpha)
    return res

startT = time.time()
res = test_small_sample_rr_coverage(int(sys.argv[1]))
res.to_csv(opj(_git, 'CORTIS','FinalAnalysis', 'test', 'adata', 'binary_coverage_N%s_2019-10-14.csv' % sys.argv[1]), index=True)
print('Results saved to binary_coverage_N%s_2019-10-14.csv (%1.2f min)' % (sys.argv[1], (time.time() - startT)/60))

# res = pd.read_csv(opj(_git, 'CORTIS','FinalAnalysis', 'test', 'adata', 'binary_coverage_N10000_2019-10-14.csv'))