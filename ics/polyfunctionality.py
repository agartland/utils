import itertools
import numpy as np
import pandas as pd

try:
    from ortools.graph import pywrapgraph
except ModuleNotFoundError:
    print('Could not import ortools')

# import networkx as nx

from .loading import subset2vec, vec2subset, compressSubsets

__all__ = ['DenseICSDist',
           'pwICSDist',
           'decomposeDist']

"""Formulating polyfunctionality distance as a min cost flow problem"""

def pwICSDist(cdf, magCol='pctpos', cyCol='cytokine', indexCols=['ptid', 'visitday', 'tcellsub', 'antigen'], factor=100000, decompose=True, maxways=3):
    """Compute all pairwise ICS distances among samples indicated by index columns.
    
    Parameters
    ----------
    cdf : pd.DataFrame
        Contains one row per cell population, many rows per sample.
    magCol : str
        Column containing the magnitude which should add up to 1 for all rows in a sample
    cyCol : str
        Column containing the marker combination for the row. E.g. IFNg+IL2-TNFa+
    indexCols : list
        List of columns that make each sample uniquely identifiable
    factor : int
        Since cost-flow estimates are based on integers, its effectively the number of
        decimal places to be accurate to. Default 1e5 means magCol is multiplied by 1e5 before rounding to int.
    decompose : bool
        Flag to indicate that distance should be decomposed into marginal and higher-order interactions.
    maxways : int
        Specify the degree of higher-order interactions evaluated in the decomposition

    Returns
    -------
    dmatDf : pd.DataFrame
        Symetric pairwise distance matrix with hierarchical columns/index of indexCols
    decompDf : pd.DataFrame
        Optional accounting of costs-flows/distances decomposed into one-way, two-way and three-way interactions"""
    cdf = cdf.set_index(indexCols + [cyCol])[magCol].unstack(indexCols).fillna(0)
    metadata = cdf.columns.tolist()
    n = cdf.shape[1]
    dmat = np.zeros((n, n))
    tab = []
    for i in range(n):
        for j in range(n):
            if i <= j:
                d = DenseICSDist(cdf.iloc[:,i], cdf.iloc[:,j], factor=factor)
                dmat[i, j] = d
                dmat[j, i] = d
                dec = decomposeDist(cdf.iloc[:,i], cdf.iloc[:,j], DenseICSDist, maxways=maxways, factor=factor)
                dec.loc[:, 'samp_i'] = i
                dec.loc[:, 'samp_j'] = j
                tab.append(dec)
    dmatDf = pd.DataFrame(dmat, columns=cdf.columns, index=cdf.columns)
    decompDf = pd.concat(tab, axis=0)
    return dmatDf, decompDf

def DenseICSDist(freq1, freq2, factor=100000, verbose=False, tabulate=False):
    """Compute a positive, symetric distance between two frequency distributions,
    where each node of the distribution can be related to every other node based
    on marker combination (e.g. IFNg+IL2-TNFa-). Uses a cost-flow optimization
    approach to finding the minimum dist/cost to move probability density from
    one node (marker combination) to another, to have the effect of turning freq1
    into freq2.

    Parameters
    ----------
    freq1, freq2 : pd.Series
        Frequency distribution that should sum to one, with identical indices
        containing all marker combinations
    factor : int
        Since cost-flow estimates are based on integers, its effectively the number of
        decimal places to be accurate to. Default 1e5 means magCol is multiplied by 1e5 before rounding to int.
    verbose : bool
        Print all cost-flow arcs. Useful for debugging.
    tabulate : bool
        Optionally return a tabulation of all the cost-flows.

    Returns
    -------
    cost : float
        Total distance between distributions in probability units.
    costtab : np.ndarray [narcs x nmarkers + 1]
        Tabulation of the all the required flows to have freq1 == freq2
        Each row is an arc. First nmarker columns indicate the costs between
        the two nodes and last colum is the cost-flow/distance along that arc."""
    
    nodeLabels = freq1.index.tolist()
    nodeVecs = [subset2vec(m) for m in nodeLabels]
    markers = nodeLabels[0].replace('-', '+').split('+')[:-1]
    nmarkers = len(markers)
    # nodes = list(range(len(nodeLabels)))

    if nmarkers == 1:
        flow = freq1[markers[0] + '+'] - freq2[markers[0] + '+']
        if tabulate:
            return np.abs(flow), np.zeros((0,nmarkers+1))
        else:
            return np.abs(flow)

    def _cost(n1, n2):
        """Hamming distance between two node labels"""
        return int(np.sum(np.abs(np.array(nodeVecs[n1]) - np.array(nodeVecs[n2]))))

    diffv = freq1/freq1.sum() - freq2/freq2.sum()
    diffv = (diffv * factor).astype(int)
    extra = diffv.sum()
    
    if extra > 0:
        for i in range(extra):
            diffv[i] -= 1
    elif extra < 0:
        for i in range(-extra):
            diffv[i] += 1
    assert diffv.sum() == 0

    posNodes = np.nonzero(diffv > 0)[0]
    negNodes = np.nonzero(diffv < 0)[0]
    
    if len(posNodes) == 0:
        """Returns None when freq1 - freq2 is 0 for every subset/row"""
        if tabulate:
            return 0, np.zeros((0,nmarkers+1))
        else:
            return 0
    """Creates a dense network connecting all sources and sinks with cost/distance specified by how many functions differ
    TODO: Could this instead be a sparse network connecting function combinations that only differ by 1? Cells have to move
    multiple times along the network then. This may minimize to the same solution??"""
    tmp = np.array([o for o in itertools.product(posNodes, negNodes)])
    startNodes = tmp[:,0].tolist()
    endNodes = tmp[:,1].tolist()

    """Set capacity to max possible"""
    capacities = diffv[startNodes].tolist()
    costs = [_cost(n1,n2) for n1,n2 in zip(startNodes, endNodes)]
    supplies = diffv.tolist()
    
    """Instantiate a SimpleMinCostFlow solver."""
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    """Add each arc."""
    for i in range(len(startNodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(startNodes[i], endNodes[i],
                                                    capacities[i], costs[i])
    """Add node supplies."""
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])
    """Find the minimum cost flow"""
    res = min_cost_flow.SolveMaxFlowWithMinCost()
        
    if res != min_cost_flow.OPTIMAL:
        if verbose:
            print('No optimal solution found.')
        if tabulate:
            return np.nan, None
        else:
            return np.nan
    if verbose:
        print('Minimum cost:', min_cost_flow.OptimalCost())
        print('')
        print('  Arc    Flow / Capacity  Cost')
        for i in range(min_cost_flow.NumArcs()):
            cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
            print('%1s -> %1s   %3s  / %3s       %3s' % (
                  min_cost_flow.Tail(i),
                  min_cost_flow.Head(i),
                  min_cost_flow.Flow(i),
                  min_cost_flow.Capacity(i),
                  cost))
    cost = min_cost_flow.OptimalCost()/factor

    if tabulate:
        costtab = np.zeros((tmp.shape[0], nmarkers+1))
        for arci in range(min_cost_flow.NumArcs()):
            hVec = nodeVecs[min_cost_flow.Head(arci)]
            tVec = nodeVecs[min_cost_flow.Tail(arci)]
            costtab[arci, :nmarkers] = hVec - tVec
            costtab[arci, nmarkers] = min_cost_flow.Flow(arci) / factor
        return cost, costtab
    else:
        return cost

def decomposeDist(freq1, freq2, ICSDist=DenseICSDist, maxways=3, factor=100000):
    """Compute decomposed distances between freq1 and freq2. The
    decomposition includes distances based on marginal/one-way marker
    combinations, two-way combinations, etc. up to maxways-way interactions.

    Effectively this means compressing freq1/freq2 into lower-order representations
    and computing the distances. The lower-order approximations will have distances
    that are less than or equal to the total distance.

    Parameters
    ----------
    freq1, freq2 : pd.Series
        Frequency distribution that should sum to one, with identical indices
        containing all marker combinations
    ICSDist : function
        Function for computing the ICSDistance. Could conceivably
        work for different distance functions because it works by marginalizing
        the input distributions and does not rely on tabulation.
    maxways : int
        Indicates the maximum order of interactions (e.g. 3 means allowing
        for three-way marker combinations)
    factor : int
        Since cost-flow estimates are based on integers, its effectively the number of
        decimal places to be accurate to. Default 1e5 means magCol is multiplied by 1e5 before rounding to int.

    Returns
    -------
    ctDf : pd.DataFrame
        Decomposition of the distance with columns: markers, distance, nmarkers"""

    nodeLabels = freq1.index.tolist()
    nodeVecs = [subset2vec(m) for m in nodeLabels]
    markers = nodeLabels[0].replace('-', '+').split('+')[:-1]
    nmarkers = len(markers)

    def _prepFreq(freq):
        tmp = freq.reset_index()
        tmp.columns = ['cytokine', 'freq']
        tmp.loc[:, 'ptid'] = 0
        return tmp

    tmp1 = _prepFreq(freq1)
    tmp2 = _prepFreq(freq2)
    costs = []
    markerCombs = []
    for nwaysi in range(min(nmarkers, maxways)):
        icombs = [d for d in itertools.combinations(np.arange(nmarkers), nwaysi+1)]
        """Number of times each marker appears in all decompositions"""
        norm_factor = np.sum([0 in cyi for cyi in icombs])
        for cyi in icombs:
            cy = [markers[i] for i in cyi]
            cfreq1 = compressSubsets(tmp1, subset=cy, indexCols=['ptid'], magCols=['freq'], nsubCols=None)
            cfreq2 = compressSubsets(tmp2, subset=cy, indexCols=['ptid'], magCols=['freq'], nsubCols=None)
            cost = ICSDist(cfreq1.set_index('cytokine')['freq'],
                                    cfreq2.set_index('cytokine')['freq'], factor=factor)
            costs.append(cost / norm_factor)
            markerCombs.append(cy)
    ctDf = pd.DataFrame({'markers':['|'.join(mc) for mc in markerCombs],
                      'distance':costs,
                      'nmarkers':[len(mc) for mc in markerCombs]})
    return ctDf

_eg_3cytokine = ['IFNg-IL2-TNFa-',
                'IFNg+IL2-TNFa-',
                'IFNg-IL2+TNFa-',
                'IFNg-IL2-TNFa+',
                'IFNg+IL2+TNFa-',
                'IFNg+IL2-TNFa+',
                'IFNg-IL2+TNFa+',
                'IFNg+IL2+TNFa+']

_eg_2cytokine = ['IFNg-IL2-',
                 'IFNg+IL2-',
                 'IFNg-IL2+',
                 'IFNg+IL2+']

def _example_data():
    freq1 = pd.Series(np.zeros(len(cytokine)), index=_eg_3cytokine)
    freq2 = pd.Series(np.zeros(len(cytokine)), index=_eg_3cytokine)
    
    freq1['IFNg+IL2-TNFa+'] = 0.5
    freq1['IFNg+IL2+TNFa-'] = 0.5
    freq2['IFNg+IL2+TNFa+'] = 1
    return freq1, freq2
    
def _test_decompose_pair():
    freq1 = pd.Series(np.zeros(len(_eg_2cytokine)), index=_eg_2cytokine)
    freq1['IFNg-IL2-'] = 1
    freq2 = freq1.copy()
    freq2['IFNg+IL2+'] += 0.1
    freq2['IFNg-IL2-'] = 0.9
    cost, costtab = DenseICSDist(freq1, freq2, factor=100000)
    ctDf = decomposeDist(freq1, freq2, DenseICSDist)

def _test_decompose_pair_interaction():
    freq1 = pd.Series([0.1, 0.4, 0.4, 0.1], index=_eg_2cytokine)
    
    """All interaction"""
    freq2 = pd.Series([0.4, 0.1, 0.1, 0.4], index=_eg_2cytokine)
    """All marginal"""
    #freq2 = pd.Series([0.1, 0.1, 0.1, 0.7], index=_eg_2cytokine)
    
    cost, costtab = DenseICSDist(freq1, freq2, factor=100000)
    ctDf = decomposeDist(freq1, freq2, DenseICSDist)

def _test_decompose_all_marg():
    freq1 = pd.Series(np.zeros(len(_eg_3cytokine)), index=_eg_3cytokine)
    freq1['IFNg-IL2-TNFa-'] = 1
    #freq1['IFNg+IL2-TNFa-'] = 0.05
    #freq1['IFNg+IL2+TNFa-'] = 0.05
    
    freq2 = freq1.copy()
    freq2['IFNg-IL2-TNFa-'] = 0.9
    freq2['IFNg+IL2-TNFa-'] += 0.05
    #freq2['IFNg+IL2+TNFa-'] += 0.02
    freq2['IFNg-IL2-TNFa+'] += 0.05
    #freq2['IFNg+IL2+TNFa+'] += 0.02

    cost, costtab = DenseICSDist(freq1, freq2, factor=100000)
    ctDf = decomposeDist(freq1, freq2, DenseICSDist)

def _test_decompose_twoway():
    freq1 = pd.Series(np.ones(len(_eg_3cytokine)) / len(_eg_3cytokine), index=_eg_3cytokine)
    freq2 = freq1.copy()
    freq2['IFNg+IL2+TNFa-'] += 0.2
    freq2['IFNg+IL2-TNFa-'] += -0.1
    freq2['IFNg-IL2+TNFa-'] += 0.05
    freq2['IFNg-IL2-TNFa+'] += -0.1
    freq2['IFNg+IL2-TNFa+'] += 0.25
    freq2['IFNg+IL2+TNFa-'] += -0.25
    freq2['IFNg+IL2+TNFa+'] += 0.15
    freq2['IFNg-IL2-TNFa-'] = 1 - freq2.iloc[1:].sum()
    
    cost, costtab = DenseICSDist(freq1, freq2, factor=100000)
    ctDf = decomposeDist(freq1, freq2, DenseICSDist)

def _test_decompose_random():
    freq1 = pd.Series(np.random.rand(len(_eg_3cytokine)), index=_eg_3cytokine)
    freq1 = freq1 / freq1.sum()
    freq2 = pd.Series(np.random.rand(len(_eg_3cytokine)), index=_eg_3cytokine)
    freq2 = freq2 / freq2.sum()
    cost, costtab = DenseICSDist(freq1, freq2, factor=100000)
    ctDf = decomposeDist(freq1, freq2, DenseICSDist)
    return ctDf
def _test_decompose_pair_random():
    freq1 = pd.Series(np.random.rand(len(_eg_2cytokine)), index=_eg_2cytokine)
    freq1 = freq1 / freq1.sum()
    freq2 = pd.Series(np.random.rand(len(_eg_2cytokine)), index=_eg_2cytokine)
    freq2 = freq2 / freq2.sum()
    cost, costtab = DenseICSDist(freq1, freq2, factor=100000)
    ctDf = decomposeDist(freq1, freq2, DenseICSDist)
    return ctDf

def _OLD_prepICSData(freq1, freq2, factor):
    nodeLabels = freq1.index.tolist()
    nodeVecs = [subset2vec(m) for m in nodeLabels]
    # nodes = list(range(len(nodeLabels)))

    def _cost(n1, n2):
        """Hamming distance between two node labels"""
        return int(np.sum(np.abs(np.array(nodeVecs[n1]) - np.array(nodeVecs[n2]))))

    diffv = freq1/freq1.sum() - freq2/freq2.sum()
    diffv = (diffv * factor).astype(int)
    extra = diffv.sum()
    
    if extra > 0:
        for i in range(extra):
            diffv[i] -= 1
    elif extra < 0:
        for i in range(-extra):
            diffv[i] += 1
    assert diffv.sum() == 0

    posNodes = np.nonzero(diffv > 0)[0]
    negNodes = np.nonzero(diffv < 0)[0]

    if len(posNodes) == 0:
        return None
    
    """Creates a dense network connecting all sources and sinks with cost/distance specified by how many functions differ
    TODO: Could this instead be a sparse network connecting function combinations that only differ by 1? Cells have to move
    multiple times along the network then. This may minimize to the same solution??"""
    tmp = np.array([o for o in itertools.product(posNodes, negNodes)])
    startNodes = tmp[:,0].tolist()
    endNodes = tmp[:,1].tolist()

    """Set capacity to max possible"""
    capacities = diffv[startNodes].tolist()
    costs = [_cost(n1,n2) for n1,n2 in zip(startNodes, endNodes)]
    supplies = diffv.tolist()

    out = {'startNodes':startNodes,
           'endNodes':endNodes,
           'capacities':capacities,
           'costs':costs,
           'supplies':supplies}
    return out

def _OLD_googleMCF(startNodes, endNodes, capacities, costs, supplies, verbose=True, withConstraints=False):
    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    # Add each arc.
    for i in range(len(startNodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(startNodes[i], endNodes[i],
                                                    capacities[i], costs[i])

    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    if not withConstraints:
        res = min_cost_flow.SolveMaxFlowWithMinCost()
    else:
        res = min_cost_flow.Solve()

    # Find the minimum cost flow
    if res == min_cost_flow.OPTIMAL:
        if verbose:
            print('Minimum cost:', min_cost_flow.OptimalCost())
            print('')
            print('  Arc    Flow / Capacity  Cost')
            for i in range(min_cost_flow.NumArcs()):
                cost = min_cost_flow.Flow(i) * min_cost_flow.UnitCost(i)
                print('%1s -> %1s   %3s  / %3s       %3s' % (
                      min_cost_flow.Tail(i),
                      min_cost_flow.Head(i),
                      min_cost_flow.Flow(i),
                      min_cost_flow.Capacity(i),
                      cost))
        return min_cost_flow.OptimalCost(), min_cost_flow.MaximumFlow()
    else:
        if verbose:
            print('No optimal solution found.')
        return np.nan, np.nan

def _OLD_pwICSDist(cdf, magCol='mag_bg', indexCols=['ptid', 'visitday', 'tcellsub', 'antigen']):
    d = {ind:tmpdf.set_index('cytokine')[magCol] for ind, tmpdf in cdf.groupby(indexCols)}
    s = pd.Series(d)
    pwdist = computePWDist(s, s, ICSDist, symetric=True)
    return pwdist

def _distnorm(v):
    v[v<0] = 0
    v = v/v.sum()
    return v

def testMCFData(factor=100000):
    np.random.seed(110820)
    nodeVecs = [o for o in itertools.product((0,1), repeat=4)]

    def _cost(n1, n2):
        """Hamming distance between two node labels"""
        return int(np.sum(np.abs(np.array(nodeVecs[n1]) - np.array(nodeVecs[n2]))))
    
    popA = _distnorm(np.random.randn(len(nodeVecs)) + 0.5)
    popB = _distnorm(np.random.randn(len(nodeVecs)) + 0.5)
    diffv = popA - popB
    diffv = (diffv*factor).astype(int)
    diffv[0] -= diffv.sum()
    assert diffv.sum() == 0

    posNodes = np.nonzero(diffv > 0)[0]
    negNodes = np.nonzero(diffv < 0)[0]

    tmp = np.array([o for o in itertools.product(posNodes, negNodes)])
    startNodes = tmp[:,0].tolist()
    endNodes = tmp[:,1].tolist()

    """Set capacity to max possible"""
    capacities = diffv[startNodes].tolist()
    costs = [_cost(n1,n2) for n1,n2 in zip(startNodes, endNodes)]
    supplies = diffv.tolist()

    out = {'startNodes':startNodes,
           'endNodes':endNodes,
           'capacities':capacities,
           'costs':costs,
           'supplies':supplies}
    return out

def _OLD_nxMCF(startNodes, endNodes, capacities, costs, supplies):
    G = nx.DiGraph()
    for n, s in enumerate(supplies):
        G.add_node(n, demand=-s)

    for edgei in range(len(startNodes)):
        G.add_edge(startNodes[edgei],
                   endNodes[edgei],
                   weight=costs[edgei],
                   capacity=capacities[edgei])

    cost, flow = nx.network_simplex(G, demand='demand', capacity='capacity', weight='weight')
    totalFlow = 0
    for k1,v1 in flow.items():
        for k2,v2 in v1.items():
            totalFlow += v2
    return cost, totalFlow

def _OLD_decomposeDist(ct):
    """Development code
    -------------------
    nodeVecs = [nv for nv in itertools.product(*((0, 1),)*3)]
    costs = np.concatenate([(np.array(t) - np.array(h))[None, :] for h, t in itertools.product(nodeVecs, nodeVecs) if not np.all(h == t)], axis=0)
    flows = np.random.rand(costs.shape[0])[:, None]
    """
    nmarkers = ct.shape[1] - 1
    """This way of representing the costs of all the arcs may not be most efficient: there are duplicates"""
    costs = ct[:, :nmarkers]
    flows = ct[:, nmarkers][:, None]

    decompCosts = []
    decompCF = []
    markerCombs = []
    """Two-way cost flows"""
    for nways in range(nmarkers):
        #costflows = flows * costs / np.sum(np.abs(costs), axis=1)[:, None]
        #margcf = costflows.sum(axis=0)
        #icombs = [d for d in itertools.combinations(np.arange(nmarkers), nways+1)]
        icombs = [d for d in itertools.product(*((0, 1),)*(nways+1))]
        dec = np.zeros((costs.shape[0], len(icombs)))
        for j,cols in enumerate(icombs):
            dec[:, j] = np.mean(costs[:, cols], axis=1)
        # dcf = flows * dec / np.sum(np.abs(dec), axis=1)[:, None]
        dcf = flows * dec# * nways/len(icombs)
        margdcf = dcf.sum(axis=0)
        decompCosts.append(dcf)
        decompCF.append(margdcf)
        markerCombs.append(icombs)

    remCF = [decompCF[i].copy() for i in range(nmarkers)]
    for nways in range(1, nmarkers):
        """Subtract off the the (x-1)-way component from each of the x-way components"""
        for kways in range(nways):
            for j,cols in enumerate(markerCombs[nways]):
                inds = []
                for i in markerCombs[kways]:
                    if np.all([ii in cols for ii in i]):
                        inds.extend(i)
                remCF[nways][j] = remCF[nways][j] - np.sum(remCF[kways][inds])
    return markerCombs, decompCosts, decompCF, remCF