import itertools
from ortools.graph import pywrapgraph
import numpy as np
import pandas as pd
import networkx as nx

from kernel_regression import computePWDist

from .loading import subset2vec, vec2subset, compressSubsets


__all__ = ['ICSDist',
           'pwICSDist',
           'prepICSData',
           'googleMCF',
           'nxMCF',
           'computeMarginals']

"""Formulating polyfunctionality distance as a min cost flow problem"""

def computeMarginals(df, indexCols, magCol='mag'):
    """Compress df cytokine subsets to a single subset for each cytokine.

    Parameters
    ----------
    df : pd.DataFrame no index
        Raw or background subtracted ICS data
    indexCols : list
        Columns that make each sample unique
    magCol : str
        Typically "mag" or "bg"

    Returns
    -------
    df : pd.DataFrame
        Rows for each of the samples that were indexed,
        and for each cytokine"""

    cytokines = df.cytokine.iloc[0].replace('-', '+').split('+')[:-1]
    out = []
    for cy in cytokines:
        marg = compressSubsets(df, indexCols=indexCols, subset=[cy], magCol=magCol)
        marg = marg.loc[marg.cytokine == cy + '+']
        out.append(marg)
    out = pd.concat(out, axis=0)
    out.loc[:, magCol] = out
    return out

def ICSDist(freq1Df, freq2Df):
    mcfData = prepICSData(freq1Df, freq2Df, factor=1000)
    if mcfData is None:
        """Returns None when freq1Df - freq2Df is 0 for every subset"""
        return 0.
    else:
        cost, flow = googleMCF(**mcfData, verbose=False, withConstraints=False)
    return cost/1000

def pwICSDist(cdf, indexCols=['ptid', 'visitday', 'tcellsub', 'antigen']):
    d = {ptid:tmpdf.set_index('cytokine')['mag'] for ptid, tmpdf in cdf.groupby('ptid')}
    s = pd.Series(d)
    pwdist = computePWDist(s, s, ICSDist, symetric=True)
    return pwdist

def _distnorm(v):
    v[v<0] = 0
    v = v/v.sum()
    return v

def testMCFData(factor=1000):
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

def prepICSData(freq1, freq2, factor=1000):
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

def googleMCF(startNodes, endNodes, capacities, costs, supplies, verbose=True, withConstraints=False):
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

def nxMCF(startNodes, endNodes, capacities, costs, supplies):
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
