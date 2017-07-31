"""Generate HVTN505 dataset for Michael on statsrv"""
import pandas as pd
import numpy as np

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