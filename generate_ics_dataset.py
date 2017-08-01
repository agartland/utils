"""Generate HVTN505 dataset for Michael on statsrv"""
import pandas as pd
import numpy as np
import re
import sns

"""Read in the raw ICS data"""
# fn = '/trials/vaccine/p505/analysis/lab/pdata/ics/e505ics_fh/csvfiles/e505ics_fh_p.csv'
fn = GIT_PATH + 'e505ics_fh_p.csv'
ctrlCols = ['ptid', 'visitday', 'tcellsub', 'cytokine']
indexCols = ctrlCols + ['antigen']
uAg = ['CMV',
       'Empty Ad5 (VRC)',
       'VRC ENV A',
       'VRC ENV B',
       'VRC ENV C',
       'VRC GAG B',
       'VRC NEF B',
       'VRC POL 1 B',
       'VRC POL 2 B']

rdf = pd.read_csv(fn, usecols=indexCols + ['nsub', 'cytnum', 'nrepl'],
                      dtype={'ptid':object,
                             'visitday':np.int,
                             'tcellsub':object,
                             'cytokine':object,
                             'antigen':object,
                             'nsub':np.int,
                             'cytnum':np.int,
                             'nrepl':np.int},
                      index_col=indexCols).sort_index()


"""Sum the negative control replicates"""
ndf = rdf.xs('negctrl', level='antigen').reset_index().groupby(ctrlCols)[['nsub', 'cytnum']].agg(np.sum)
ndf.loc[:, 'bg'] = ndf['cytnum'] / ndf['nsub']

"""Define the magnitude as the fraction of cytokine positive cells"""
pdf = rdf.loc[(slice(None), slice(None), slice(None), slice(None), uAg), :]
pdf.loc[:, 'mag'] = pdf['cytnum'] / pdf['nsub']

"""Subtract off the background/negative control"""
df = pdf['mag'].reset_index().join(ndf['bg'], on=ctrlCols)

"""Compress cytokine subsets using TBIMPAACT/polyfunctionality.py"""
subset = ['IFNg', 'IL2', 'TNFa', 'IL4']
cdf = compressSubsets(df,
                      indexCols=['ptid', 'visitday', 'tcellsub', 'antigen'],
                      subset=subset,
                      magCol='mag')

def subset2vec(cy, nsubsets=4):
    m = re.match(r'.*([\+-])'*nsubsets, cy)
    if not m is None:
        vec = np.zeros(len(m.groups()))
        for i,g in enumerate(m.groups()):
            vec[i] = 1 if g == '+' else 0
    return vec

def vec2subset(vec, cytokines=['IFNg', 'IL2', 'TNFa', 'IL4']):
    s = ''
    for i,cy in enumerate(cytokines):
        s += cy
        s += '+' if vec[i] == 1 else '-'
    return s

binSubsets = np.concatenate([m[None, :] for m in map(subset2vec, cdf.cytokine.unique())], axis=0)

nColors = (np.unique(binSubsets.sum(axis=1)) > 0).sum()
cmap = sns.light_palette('red', as_cmap=True, n_colors=nColors)

freqDf = cdf.groupby('cytokine')['mag'].agg(np.mean)
freqDf = freqDf.drop(vec2subset((0,0,0,0)), axis=0)

g = nx.Graph()
for ss,f in freqDf.iteritems():
    g.add_node(ss, freq=f, fscore=subset2vec(ss).sum())
for ss1, ss2 in itertools.product(freqDf.index, freqDf.index):
    if np.abs(subset2vec(ss1) - subset2vec(ss2)).sum() <= 1:
        g.add_edge(ss1, ss2)

nodesize = np.array([d['freq'] for n, d in g.nodes(data=True)])
nodecolor = np.array([d['fscore'] for n, d in g.nodes(data=True)])
nodecolor = (nodecolor - nodecolor.min() + 1) / (nodecolor.max() - nodecolor.min() + 1)

def szscale(vec, mx=np.inf, mn=1):
    """Normalize values of vec to [mn, mx] interval
    such that sz ratios remain representative."""
    factor = mn/np.nanmin(vec)
    vec = vec*factor
    vec[vec > mx] = mx
    vec[np.isnan(vec)] = mn
    return vec    

freq = {n:d['freq'] for n, d in g.nodes(data=True)}
pos = nx.nx_pydot.graphviz_layout(g, prog=layout, root=max(list(freq.keys()), key=freq.get))
#pos = spring_layout(g)
#pos = spectral_layout(g)
#layouts = ['twopi', 'fdp', 'circo', 'neato', 'dot', 'spring', 'spectral']
#pos = nx.graphviz_layout(g, prog=layout)


plt.figure(1)
plt.clf()
figh = plt.gcf()
axh = figh.add_axes([0.04, 0.04, 0.92, 0.92])
axh.axis('off')
figh.set_facecolor('white')

#nx.draw_networkx_edges(g,pos,alpha=0.5,width=sznorm(edgewidth,mn=0.5,mx=10), edge_color='k')
#nx.draw_networkx_nodes(g,pos,node_size=sznorm(nodesize,mn=500,mx=5000),node_color=nodecolors,alpha=1)

for e in g.edges_iter():
    x1, y1=pos[e[0]]
    x2, y2=pos[e[1]]
    props = dict(color='black', alpha=0.4, zorder=1)
    plt.plot([x1, x2], [y1, y2], '-', lw=2, **props)

plt.scatter(x=[pos[s][0] for s in g.nodes()],
            y=[pos[s][1] for s in g.nodes()],
            s=szscale(nodesize, mn=20, mx=200), #Units for scatter is (size in points)**2
            c=nodecolor,
            alpha=1, zorder=2, cmap=cmap)

for n, d in g.nodes(data=True):
    if d['freq'] >= 0:
        plt.annotate(n,
                    xy=pos[n],
                    fontname='Arial',
                    size=10,
                    weight='bold',
                    color='black',
                    va='center',
                    ha='center')

"""This visualization isn't promising, but its also the start to how
I'd think about defining a pairwise sample distance matrix. Instead
of considering each subset as independent they could be related by their
distance on this graph (just the sum of the binayr vector representation),
then the distance would be somekind of earth over's distance between the two graphs"""


"""Create a wide dataset with hierarchical columns"""
wideDf = df.set_index(indexCols).unstack(['visitday', 'tcellsub', 'antigen', 'cytokine'])

"""Flatten the columns as a string with "|" separator"""
def strcatFun(iter, sep='|'):
    s = ''
    for v in iter:
        s += str(v) + sep
    return s[:-len(sep)]
    
strCols = [strcatFun(c) for c in wideDf.columns.tolist()]
outDf = wideDf.copy()
outDf.columns = strCols
outDf.to_csv('hvtn505_ics_24Jul2017.csv')

"""Save column metadata"""
colMeta = wideDf.columns.to_frame().drop(0, axis=1)
colMeta.index = strCols
colMeta.index.name = 'column'
colMeta.to_csv('hvtn505_ics_colmeta_24Jul2017.csv')


"""Load PTID rx data"""
rxFn = '/trials/vaccine/p505/analysis/adata/rx_v2.csv'
trtCols = ['ptid', 'arm', 'grp', 'rx_code', 'rx', 'pub_id']
tmp = pd.read_csv(rxFn)
tmp = tmp.rename_axis({'Ptid': 'ptid'}, axis=1)
tmp.loc[:, 'ptid'] = tmp.ptid.str.replace('-', '')
trtDf = tmp[trtCols].set_index('ptid')
trtDf.to_csv('hvtn505_ics_rowmeta_24Jul2017.csv')


"""Formulating polyfunctionality distance as a min cost flow problem"""
import itertools
from ortools.graph import pywrapgraph
import networkx as nx
import numpy as np
import networkx as nx

def _distnorm(v):
    v[v<0] = 0
    v = v/v.sum()
    return v

def testMCFData(factor=1000):
    np.random.seed(110820)
    popA = _distnorm(np.random.randn(len(nodes)) + 0.5)
    popB = _distnorm(np.random.randn(len(nodes)) + 0.5)
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

def prepICSData(freq1, freq2, factor=100):
    nodeLabels = freq1.index.tolist()
    nodeVecs = [subset2vec(m) for m in nodeLabels]
    nodes = list(range(len(nodeLabels)))

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
    return cost, flow

samples = cdf[['ptid','tcellsub','antigen']].drop_duplicates()
freqs = []
for i, (ptid, tsub, ag) in samples.iterrows():
    tmp = cdf.loc[(cdf.ptid == ptid) & (cdf.tcellsub == tsub) & (cdf.antigen == ag)]
    freqs.append(tmp.set_index('cytokine')['mag'])
    if len(freqs) > 10:
        break

pwdist = np.zeros((len(freqs), len(freqs)))
for i,j in itertools.product(range(len(freqs)), repeat=2):
    f1,f2 = freqs[i], freqs[j]
    mcfData = prepICSData(f1, f2, factor=1000)
    if not mcfData is None:
        cost, flow = googleMCF(**mcfData, withConstraints=False, verbose=False)
    else:
        flow = 0
    pwdist[i,j] = flow

group1 = cdf.ptid.unique()[:20].tolist()
group2 = cdf.ptid.unique()[20:50].tolist()

freq1Df = cdf.loc[cdf.ptid.isin(group1)].groupby('cytokine')['mag'].agg(np.mean)
freq1Df = freq1Df.drop(vec2subset((0,0,0,0)), axis=0)

freq2Df = cdf.loc[cdf.ptid.isin(group2)].groupby('cytokine')['mag'].agg(np.mean)
freq2Df = freq2Df.drop(vec2subset((0,0,0,0)), axis=0)

mcfData = prepICSData(freq1Df, freq2Df, factor=1000)

mcfTest = testMCFData(factor=1000)

cost, flow = googleMCF(**mcfTest, withConstraints=False, verbose=False)
# cost,flow = nxMCF(**mcfData)
print(cost, flow)

cost, flow = googleMCF(**mcfTest, withConstraints=False)
cost,flow = nxMCF(**mcfTest)
print(cost, flow)