import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import palettable

import itertools
from hclusterplot import plotHCluster
import re
from myboxplot import myboxplot

try:
    import networkx as nx
except ImportError:
    print('Could not import networkx.')
import seaborn as sns

sns.set(style='darkgrid', palette='muted', font_scale=1.5)

__all__ = ['icsTicks',
           'icsTickLabels',
           'swarmBox']

from .loading import *
from .analyzing import *

icsTicks = np.log10([0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1])
icsTickLabels = ['0.01', '0.025', '0.05', '0.1', '0.25', '0.5', '1']
# icsTicks = np.log10([0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1])
#icsTickLabels = ['0.01','0.025', '0.05', '0.1','0.2','0.4','0.6','0.8', '1']

def prepPlotDf(jDf, antigen, rxIDs, visitno, tcellsubset='CD4+', column='pvalue', cutoff='pvalue', pAdjust=True, allSubsets=False):
    cytokineSubsets = jDf.cytokine.unique()
    subset = cytokineSubsets[0].replace('-', '+').split('+')[:-1]
    cyCols = [c for c in cytokineSubsets if not c == '-'.join(subset)+'-'] 

    ind = (jDf.tcellsub == tcellsubset) & (jDf.visitno == visitno) & (jDf.TreatmentGroupID.isin(rxIDs))
    agInd = (jDf.antigen == antigen)  & ind

    pvalueDf = pivotPvalues(jDf.loc[agInd], adjust=pAdjust)
    """Use cutoff from HVTN ICS SAP, p < 0.00001"""
    responseAlpha = 1e-5
    callDf = (pvalueDf < responseAlpha).astype(float)

    magDf = jDf.loc[agInd].pivot(index='sample', columns='cytokine', values='mag')
    magAdjDf = jDf.loc[agInd].pivot(index='sample', columns='cytokine', values='mag_adj')
    bgDf = jDf.loc[agInd].pivot(index='sample', columns='cytokine', values='bg')
    
    
    """Positive subsets (to-be plotted) includes all columns unless a cutoff is specified"""
    if cutoff == 'mag':
        posSubsets = pvalueDf[cyCols].columns[(magDf[cyCols] > 0.00025).any(axis=0)]
    elif cutoff == 'mag_adj':
        posSubsets = pvalueDf[cyCols].columns[(magAdjDf[cyCols] > 0.00025).any(axis=0)]
    elif cutoff == 'bg':
        posSubsets = pvalueDf[cyCols].columns[(bgDf[cyCols] > 0).any(axis=0)]
    elif cutoff == 'pvalue':
        posSubsets = pvalueDf[cyCols].columns[(callDf[cyCols] > 0).any(axis=0)]
    else:
        posSubsets = pvalueDf[cyCols].columns
        
    if allSubsets:
        posSubsets = sorted(cytokineSubsets, key=lambda s: s.count('+'), reverse=True)        
    else:
        posSubsets = sorted(posSubsets, key=lambda s: s.count('+'), reverse=True)

    if column == 'pvalue':
        plotDf = callDf
    elif column == 'mag':
        plotDf = magDf.applymap(np.log)
    elif column == 'mag_adj':
        plotDf = magAdjDf.applymap(np.log)
    elif column == 'bg':
        plotDf = bgDf.applymap(np.log)

    """Give labels a more readable look"""
    plotDf = plotDf.rename_axis(cytokineSubsetLabel, axis=1)
    posSubsets = list(map(cytokineSubsetLabel, posSubsets))

    return plotDf[posSubsets]

def plotPolyBP(jDf,
               antigen,
               rxIDs,
               visitno,
               tcellsubset='CD4+',
               column='pvalue', cutoff='pvalue',
               pAdjust=True,
               allSubsets=False, plotSubsets=None, returnPlotSubsets=False):
    if plotSubsets is None:
        plotDf = prepPlotDf(jDf, antigen, rxIDs, visitno, tcellsubset=tcellsubset, column=column, cutoff=cutoff, pAdjust=pAdjust, allSubsets=allSubsets)
        posSubsets = plotDf.columns
    else:
        plotDf = prepPlotDf(jDf, antigen, rxIDs, visitno, tcellsubset=tcellsubset, column=column, cutoff=cutoff, pAdjust=pAdjust, allSubsets=True)
        posSubsets = plotSubsets

    cbt = np.log([0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01])
    cbtl = ['0.01', '0.025', '0.05', '0.1', '0.2', '0.4', '0.6', '0.8', '1']

    plt.clf()
    plotDf = pd.DataFrame(plotDf.stack().reset_index())
    plotDf = plotDf.set_index('sample')
    plotDf = plotDf.join(ptidDf[['TreatmentGroupID', 'TreatmentGroupName']], how='left').sort_values(by='TreatmentGroupID')

    if column == 'mag' or column == 'mag_adj':
        plotDf[0].loc[(plotDf[0] < np.log(0.00025)) | plotDf[0].isnull()] = np.log(0.00025)
        yl = np.log([0.0002, 0.01])
    elif column == 'bg':
        plotDf[0].loc[(plotDf[0] < np.log(0.00001)) | plotDf[0].isnull()] = np.log(0.00001)
        yl = np.log([0.00001, 0.01])
    else:
        print('Must specify mag, mag_adj or bg (not %s)' % column)
    
    axh = plt.subplot(111)
    sns.boxplot(x='cytokine', y=0, data=plotDf, hue='TreatmentGroupName', fliersize=0, ax=axh, order=posSubsets)
    sns.stripplot(x='cytokine', y=0, data=plotDf, hue='TreatmentGroupName', jitter=True, ax=axh, order=posSubsets)
    plt.yticks(cbt, cbtl)
    plt.ylim(yl)
    plt.xticks(list(range(len(posSubsets))), posSubsets, fontsize='large', fontname='Consolas')
    plt.ylabel('% cytokine expressing cells')

    handles, labels = axh.get_legend_handles_labels()
    l = plt.legend(handles[len(rxIDs):], labels[len(rxIDs):], loc='upper right')
    if returnPlotSubsets:
        return axh, posSubsets
    else:
        return axh

def plotPolyHeat(jDf, antigen, rxIDs, visitno, tcellsubset='CD4+', cluster=False, column='pvalue', cutoff='pvalue', pAdjust=True, allSubsets=False):
    plotDf = prepPlotDf(jDf, antigen, rxIDs, visitno, tcellsubset=tcellsubset, column=column, cutoff=cutoff, pAdjust=pAdjust, allSubsets=allSubsets)
    posSubsets = plotDf.columns
    plotDf = plotDf.join(ptidDf[['TreatmentGroupID', 'TreatmentGroupName']], how='left').sort_values(by='TreatmentGroupID')

    cbt = np.log([0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01])
    cbtl = ['0.01', '0.025', '0.05', '0.1', '0.2', '0.4', '0.6', '0.8', '1']

    if cluster:
        clusterBool = [True, True]
    else:
        clusterBool = [False, False]

    if column == 'pvalue':
        vRange = [0, 2]
    elif column == 'mag':
        vRange = np.log([0.0001, 0.01])
    elif column == 'mag_adj':
        vRange = np.log([0.0001, 0.01])
    elif column == 'bg':
        vRange = np.log([0.0001, 0.01])
        #valVec = tmp[posSubsets].values.flatten()
        #vRange = [log(valVec[valVec>0].min()),log(valVec.max())]
    
    ptidInd, cyColInd, handles = plotHCluster(plotDf[posSubsets],
                                              row_labels=plotDf.TreatmentGroupID,
                                              cmap=palettable.colorbrewer.sequential.YlOrRd_9.mpl_colormap,
                                              yTickSz=None,
                                              xTickSz='large',
                                              clusterBool=clusterBool,
                                              vRange=vRange)
    
    if column == 'pvalue':
        handles['cb'].remove()
    else:
        handles['cb'].set_ticks(cbt)
        handles['cb'].set_ticklabels(cbtl)
        handles['cb'].set_label('% cells')

    for xh in handles['xlabelsL']:
        xh.set_rotation(0)
    handles['heatmapAX'].grid(b=None)
    #handles['heatmapGS'].tight_layout(handles['fig'], h_pad=0.1, w_pad=0.5)
    return handles

def plotResponsePattern(jDf, antigen, rxIDs, visitno, tcellsubset='CD4+', column='pvalue', cluster=False, cutoff='pvalue', pAdjust=True, boxplot=False, allSubsets=False):
    if column == 'pvalue' and boxplot:
        boxplot = False
        print('Forced heatmap for p-value plotting.')

    if boxplot:
        axh = plotPolyBP(jDf, antigen, rxIDs, visitno, tcellsubset=tcellsubset, column=column, cutoff=cutoff, pAdjust=pAdjust, allSubsets=allSubsets)
    else:
        axh = plotPolyHeat(jDf, antigen, rxIDs, visitno, tcellsubset=tcellsubset, cluster=cluster, column=column, cutoff=cutoff, pAdjust=pAdjust, allSubsets=allSubsets)
    return axh

def _szscale(vec, mx=np.inf, mn=1):
        """Normalize values of vec to [mn, mx] interval
        such that sz ratios remain representative."""
        factor = mn/np.nanmin(vec)
        vec = vec*factor
        vec[vec > mx] = mx
        vec[np.isnan(vec)] = mn
        return vec    

def plotPolyFunNetwork(cdf):
    """This visualization isn't promising, but its also the start to how
    I'd think about defining a pairwise sample distance matrix. Instead
    of considering each subset as independent they could be related by their
    distance on this graph (just the sum of the binayr vector representation),
    then the distance would be somekind of earth over's distance between the two graphs"""
    binSubsets = np.concatenate([m[None, :] for m in map(_subset2vec, cdf.cytokine.unique())], axis=0)

    nColors = (np.unique(binSubsets.sum(axis=1)) > 0).sum()
    cmap = sns.light_palette('red', as_cmap=True, n_colors=nColors)

    freqDf = cdf.groupby('cytokine')['mag'].agg(np.mean)
    freqDf = freqDf.drop(vec2subset([0]*len(binSubsets)), axis=0)

    g = nx.Graph()
    for ss,f in freqDf.iteritems():
        g.add_node(ss, freq=f, fscore=subset2vec(ss).sum())
    for ss1, ss2 in itertools.product(freqDf.index, freqDf.index):
        if np.abs(subset2vec(ss1) - subset2vec(ss2)).sum() <= 1:
            g.add_edge(ss1, ss2)

    nodesize = np.array([d['freq'] for n, d in g.nodes(data=True)])
    nodecolor = np.array([d['fscore'] for n, d in g.nodes(data=True)])
    nodecolor = (nodecolor - nodecolor.min() + 1) / (nodecolor.max() - nodecolor.min() + 1)

    freq = {n:d['freq'] for n, d in g.nodes(data=True)}
    pos = nx.nx_pydot.graphviz_layout(g, prog=layout, root=max(list(freq.keys()), key=freq.get))
    #pos = spring_layout(g)
    #pos = spectral_layout(g)
    #layouts = ['twopi', 'fdp', 'circo', 'neato', 'dot', 'spring', 'spectral']
    #pos = nx.graphviz_layout(g, prog=layout)

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
                s=_szscale(nodesize, mn=20, mx=200), #Units for scatter is (size in points)**2
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

def swarmBox(data, x, y, hue, palette=None, order=None, hue_order=None, connect=False):
    """Depends on plot order of the swarm plot which does not seem dependable at the moment.
    Better idea would be to adopt code from the actual swarm function for this, adding boxplots separately"""
    if palette is None:
        palette = sns.color_palette('Set2',  n_colors=data[hue].unique().shape[0])
    if hue_order is None:
        hue_order = sorted(data[hue].unique())
    if order is None:
        order = sorted(data[x].unqiue())
        
    params = dict(data=data, x=x, y=y, hue=hue, palette=palette, order=order, hue_order=hue_order)
    sns.boxplot(**params, fliersize=0, linewidth=0.5)
    swarm = sns.swarmplot(**params, linewidth=0.5, edgecolor='black', dodge=True)
    if connect:
        zipper = [order] + [swarm.collections[i::len(hue_order)] for i in range(len(hue_order))]
        for z in zip(*zipper):
            curx = z[0]
            collections = z[1:]
            offsets = []
            for c,h in zip(collections, hue_order):
                ind = (data[x] == curx) & (data[hue] == h)
                sortii = np.argsort(np.argsort(data.loc[ind, y]))
                offsets.append(c.get_offsets()[sortii,:])

            for zoffsets in zip(*offsets):
                xvec = [o[0] for o in zoffsets]
                yvec = [o[1] for o in zoffsets]
                plt.plot(xvec, yvec, '-', color='gray', linewidth=0.5)
    plt.legend([plt.Circle(1, color=c) for c in palette], hue_order, title=hue)
