import pandas as pd
import numpy as np
import itertools
import warnings
import sys

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print('Importing hier_diff without matplotlib.')

import scipy.cluster.hierarchy as sch
from scipy.spatial import distance
from scipy import stats

try:
    from adjustwithin import adjustnonnan
except ImportError:
    print('Importing hier_diff without multiplicity adjustment package.')

__all__ = ['testHClusters',
           'getClusterMembers',
           'plotHClustProportions',
           'testCondition',
           'testSubset']

def testHClusters(cntsDf, members, cols=None, min_count=5):
    """Test each cluster for disproportionate representation of TCRs
    from a set of conditions (e.g. stimulations). Test is based on the Chi2 statistic,
    testing the observed proportions vs. expected proportions of TCRs
    that are in vs. not-in a cluster (i.e. 2 x N_cols table). 

    Parameters
    ----------
    cntsDf : pd.DataFrame [TCRs, conditions]
        Counts table of TCRs (rows) that have been observed in specific conditions (columns)
        Importantly the integer indices of the rows must match those used to define
        clusters in members.
    members : dict of lists
        Each element has a cluster ID (key) and a list of cluster members (indices into cntsDf)
        Can be generated from getClusterMembers with the result from calling sch.linkage (Z).
        Cluster need not be mutually exclusive, and are not when using hierarchical clustering.
    cols : list
        Columns in cntsDf to use as conditions (default: all columns of cntsDf)
    min_count : int
        Required minimum number of member TCRs in a cluster to run the test.

    Returns
    -------
    resDf : pd.DataFrame [clusters, result columns]
        Results from the tests with observed/expected counts and frequencies, Chi2 statistics,
        p-values, FWER and FDR adjusted p-values."""

    if cols is None:
        cols = cntsDf.columns

    tot = cntsDf.sum()
    Ncells = tot.sum()
    uCDR3 = list(cntsDf.index)

    results = []

    for cid, m in members.items():
        notM = [i for i in range(cntsDf.shape[0]) if not i in m]
        obs = np.concatenate((np.sum(cntsDf[cols].values[m, :], axis=0, keepdims=True),
                              np.sum(cntsDf[cols].values[notM, :], axis=0, keepdims=True)), axis=0)
        if np.sum(obs, axis=1)[0] > min_count:
            """Inner product of the marginal totals along both axes, divided by total cells"""
            expect = np.dot(np.sum(obs, keepdims=True, axis=1),
                            np.sum(obs, keepdims=True, axis=0)) / Ncells
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                chi2 = (obs - expect)**2 / expect

            sum_chi2 = np.sum(chi2)

            degf = len(cols) - 1
            pvalue = 1 - stats.chi2.cdf(sum_chi2, degf)
            results.append({'cid':cid,
                            'chi2':sum_chi2,
                            'pvalue':pvalue,
                            'observed':tuple(obs[0, :]),
                            'observed_prop':(obs / np.sum(obs, axis=0))[0, :],
                            'expected':tuple(expect[0, :]),
                            'expected_prop':(expect / np.sum(obs, axis=0))[0, :],
                            'members':tuple(m),
                            'labels':cols})
        else:
            results.append({'cid':cid,
                            'chi2':np.nan,
                            'pvalue':np.nan,
                            'observed':tuple(obs[0, :]),
                            'observed_prop': (obs / np.sum(obs, axis=0))[0, :],
                            'expected':(np.nan, )*len(cols),
                            'expected_prop': (np.nan, )*len(cols),
                            'members':tuple(m),
                            'labels':cols})
    resDf = pd.DataFrame(results)

    if 'adjustwithin' in sys.modules:
        resDf.loc[:, 'FWER-pvalue'] = adjustnonnan(resDf['pvalue'], method='holm')
        resDf.loc[:, 'FDR-qvalue'] = adjustnonnan(resDf['pvalue'], method='fdr_bh')
    return resDf.set_index('cid')

def getClusterMembers(Z):
    """Generate dict of lists where each key is a cluster ID from the results
    of linkage-based hierarchical clustering with scipy.cluster.hierarchy.linkage (Z)

    Parameters
    ----------
    Z : linkage matrix [clusters, 4]

    Returns
    -------
    members : dict of lists
        Each element has a cluster ID (key) and a list of
        cluster members (indices into the original data matrix)"""

    clusters = {}
    for i, merge in enumerate(Z):
        cid = 1 + i + Z.shape[0]
        clusters[cid] = [merge[0], merge[1]]

    def _getIndices(clusters, i):
        if i <= Z.shape[0]:
            return [int(i)]
        else:
            return _getIndices(clusters, clusters[i][0]) + _getIndices(clusters, clusters[i][1])

    members = {i:_getIndices(clusters, i) for i in range(Z.shape[0] + 1, max(clusters.keys()) + 1)}
    return members

def plotHClustProportions(figh, Z, resDf, alpha_col='pvalue', alpha=0.05, colors=None, ann='N', xLim=None, maxY=None, min_count=20):
    """Plot tree of linkage-based hierarchical clustering, with nodes colored with stacked bars
    representing proportion of cluster members associated with specific conditions. Nodes also optionally
    annotated with pvalue, number of members or cluster ID.

    Parameters
    ----------
    figh : mpl Figure() handle
    Z : linkage matrix
        Result of calling sch.linkage on a compressed pair-wise distance matrix 
    resDf : pd.DataFrame
        Result from calling testHClusters, with observed/frequencies and p-values for each node
    alpha_col : str
        Column in resDf to use for 'alpha' annotation
    alpha : float
        Threshold for plotting the stacked bars and annotation
    colors : tuple of valid colors
        Used for stacked bars of conditions at each node
    labels : list of condition labels
        Matched to tuples of colors and frequencies in resDf
    ann : str
        Indicates how nodes should be annotated: N, alpha, CID supported
    xLim : tuple
        Apply x-lims after plotting to focus on particular part of the tree"""

    nCategories = len(resDf['observed'].iloc[0])
    if colors is None:
        colors = sns.color_palette('Set1', n_colors=nCategories)
    labels = resDf['labels'].iloc[0]
    
    dend = sch.dendrogram(Z, no_plot=True,
                             color_threshold=None,
                             link_color_func=lambda lid: hex(lid),
                             above_threshold_color='FFFFF')
    figh.clf()
    axh = plt.axes((0.05, 0.07, 0.8, 0.8), facecolor='w')

    lowestY = None
    annotateCount = 0
    for xx, yy, hex_cid in zip(dend['icoord'], dend['dcoord'], dend['color_list']):
        cid = int(hex_cid, 16)
        xx = np.array(xx) / 10
        axh.plot(xx, yy, zorder=1, lw=0.5, color='k', alpha=1)

        N = np.sum(resDf.loc[cid, 'observed'])
        if alpha is None or resDf.loc[cid, alpha_col] <= alpha and N > min_count:
            obs = np.asarray(resDf.loc[cid, 'observed_prop'])
            obs = obs / np.sum(obs)
            L = (xx[2] - xx[1])
            xvec = L * np.concatenate(([0.], obs, [1.]))
            curX = xx[1]
            for i in range(len(obs)):
                c = colors[i]
                axh.plot([curX, curX + L*obs[i]],
                         yy[1:3],
                         color=c,
                         lw=10,
                         solid_capstyle='butt')
                curX += L*obs[i]
            if ann == 'N':
                s = '%1.0f' % N
            elif ann == 'CID':
                s = cid
            elif ann == 'alpha':
                if resDf.loc[cid, alpha_col] < 0.001:
                    s = '< 0.001'
                else:
                    s = '%1.3f' % resDf.loc[cid, alpha_col]
            if not ann == '':# and annotateCount < annC:
                xy = (xx[1] + L/2, yy[1])
                # print(s,np.round(xy[0]), np.round(xy[1]))
                annotateCount += 1
                axh.annotate(s,
                             xy=xy,
                             size='x-small',
                             horizontalalignment='center',
                             verticalalignment='center')
            if lowestY is None or yy[1] < lowestY:
                lowestY = yy[1]
    yl = axh.get_ylim()
    if not lowestY is None:
        yl0 = 0.9*lowestY
    else:
        yl0 = yl[0]
    if not maxY is None:
        yl1 = maxY
    else:
        yl1 = yl[1]
    axh.set_ylim((yl0, yl1))
    
    axh.set_yticks(())
    if not xLim is None:
        if xLim[1] is None:
            xl1 = axh.get_xlim()[1]
            xLim = (xLim[0], xl1)
        axh.set_xlim(xLim)
    else:
        xLim = axh.get_xlim()

    xt = [x for x in range(0, Z.shape[0]) if x <= xLim[1] and x>= xLim[0]]
    xt = xt[::len(xt) // 10]
    # xtl = [x//10 for x in xt]
    axh.set_xticks(xt)
    # axh.set_xticklabels(xtl)
    legh = axh.legend([plt.Rectangle((0,0), 1, 1, color=c) for c in colors],
            labels,
            loc='upper left', bbox_to_anchor=(1, 1))

def testCondition(df, indexCol, dmatDf, gbCol, gbValues=None, countCol='Cells', min_count=3):
    """Use hierarchical clustering to cluster data in df based on unique pair-wise distances
    in dmatDf. Then test clusters for disproportionate association of members with a condition
    indicated in gbCol.

    Parameters
    ----------
    df : pd.DataFrame [TCRs, metadata]
        Contains freqeuncy data for TCRs in longform.
        May be a subset of the larger dataset that was used for clustering.
    indexCol : str
        Column to use as the index for individual TCRs
    dmatDf : pd.DataFrame [unique indices, unique indices]
        Contains pairwise distances among all unique values in the indexCol of df
    gbCol : str
        Column of metadata in df containing conditions for testing
    gbValues : list
        List of values relevant for testing. Can be fewer than all values in gbCol to ignore
        irrelevant conditions.
    countCol : str
        Column containing the integer counts for testing
    min_count : int
        Required minimum number of member TCRs in a cluster to run the test."""

    if gbValues is None:
        gbValues = sorted(df[gbCol].unique())

    cnts = df.groupby([indexCol, gbCol])[countCol].agg(np.sum).unstack(gbCol, fill_value=0)[gbValues]
    uIndices = list(df[indexCol].dropna().unique())
    dmat = dmatDf.loc[:, uIndices].loc[uIndices, :]
    compressedDmat = distance.squareform(dmat.values)
    Z = sch.linkage(compressedDmat, method='complete')
    members = getClusterMembers(Z)
    resDf = testHClusters(cnts, members, gbValues, min_count=min_count)
    return Z, resDf, np.array(uIndices)

def testSubset(df, fullIndex, indexCol, members, gbCol='Stimulus', gbValues=None, countCol='Cells', min_count=7, nsamps=None, rseed=110820):
    """Test clusters for disproportionate association of members with a condition indicated in gbCol.
    Flexible for testing a subset of the data that was used for clustering
    (and which is represented in members). This is helpful when the clustering is more accurate with the
    larger dataset, but a questions is asked of only a subset of the data.

    Permutation-based testing has been indistinguisable from analytic Chi2-based testing in preliminary tests.

    Parameters
    ----------
    df : pd.DataFrame [TCRs, metadata]
        Contains freqeuncy data for TCRs in longform.
        May be a subset of the larger dataset that was used for clustering.
    fullIndex : list
        List of all unique values of the indexCol in the whole dataset.
        Order of values must match the integer indices in members.
    indexCol : str
        Column to use as the index for individual TCRs
    members : dict of lists
        Each element has a cluster ID (key) and a list of cluster members (indices into cntsDf)
        Can be generated from getClusterMembers with the result from calling sch.linkage (Z).
        Cluster need not be mutually exclusive, and are not when using hierarchical clustering.
    gbCol : str
        Column of metadata containing conditions for testing
    gbValues : list
        List of values relevant for testing. Can be fewer than all values in gbCol to ignore
        irrelevant conditions.
    countCol : str
        Column containing the integer counts for testing
    min_count : int
        Required minimum number of member TCRs in a cluster to run the test.
    nsamps : int
        Number of permutations for permutation-based testing
    rseed : int
        Random numer seed for permutation-based testing"""

    if gbValues is None:
        gbValues = sorted(df[gbCol].unique())
    cnts = df.groupby([indexCol, gbCol])[countCol].agg(np.sum).unstack(gbCol, fill_value=0)[gbValues]
    cnts = cnts.reindex(fullIndex, axis=0, fill_value=0)
    resDf = testHClusters(cnts, members, gbValues, min_count=min_count)
    
    if not nsamps is None:
        """Preliminarily, permutation-based p-values have correlated perfectly
        with the analytic p-values"""
        np.random.seed(rseed)
        rtmp = df.copy()
        rchi2 = np.zeros((resDf.shape[0], nsamps))
        rpvalue = np.zeros((resDf.shape[0], nsamps))
        for sampi in range(nsamps):
            rtmp.loc[:, gbCol] = rtmp[gbCol].values[np.random.permutation(rtmp.shape[0])]
            rcnts = rtmp.groupby([indexCol, gbCol])['Cells'].agg(np.sum).unstack(gbCol, fill_value=0)
            rcnts = rcnts.reindex(fullIndex, axis=0, fill_value=0)
            rres = testHClusters(rcnts, members, gbValues, min_count=min_count)
            rchi2[:, sampi] = rres['chi2']
            rpvalue[:, sampi] = rres['pvalue']
        ppvalue = ((rpvalue <= resDf['pvalue'].values[:, None]).sum(axis=1) + 1) / (nsamps + 1)
        pchi2 = ((rchi2 <= resDf['chi2'].values[:, None]).sum(axis=1) + 1) / (nsamps + 1)
        ppvalue[np.isnan(resDf['chi2'].values)] = np.nan
        pchi2[np.isnan(resDf['chi2'].values)] = np.nan
        resDf = resDf.assign(**{'Perm P-pvalue':ppvalue, 'Perm Chi2-pvalue':pchi2})

    return resDf