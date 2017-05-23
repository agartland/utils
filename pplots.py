import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import scipy.cluster.hierarchy as sch
from matplotlib.gridspec import GridSpec
import seaborn as sns
import palettable

def mapColors2Labels(labels, setStr='Set3', cmap=None):
    """Return pd.Series of colors based on labels"""
    if cmap is None:
        N = max(3, min(12, len(np.unique(labels))))
        # cmap = palettable.colorbrewer.get_map(setStr, 'Qualitative', N).mpl_colors
        """Use B+W colormap"""
    cmapLookup = {k:col for k,col in zip(sorted(np.unique(labels)), itertools.cycle(cmap))}
    return labels.map(cmapLookup.get)

def plotResultSummary(resDf, index, columns,
                      stat=('stat','Rho'),
                      qvalue='qvalue',
                      pvalue='pvalue',
                      include=('pvalue', 0.05),
                      censor=('pvalue', 0.05),
                      star=('qvalue', [0.2, 0.1, 0.01]),
                      vRange=None,
                      scalebar=True,
                      minColumns=None,
                      tickSz=12,
                      splitN = 100,
                      indexOrder=None):
    if type(stat) in [list, tuple]:
        statLabel = stat[1]
        stat = stat[0]
    else:
        statlabel = stat

    includeInd = resDf[include[0]] <= include[1]
    tmp = resDf.loc[includeInd, :]
    
    if not minColumns is None:
        """Exclude rows that don't have at least two columns not censored,
        and if they don't have at least one star."""
        gbN = tmp[[index, pvalue]].groupby(index).agg(len)
        gbMin = tmp[[index, star[0]]].groupby(index).agg(min)
        keepIndex = gbN.index[(gbN[pvalue] >= minColumns) | (gbMin[star[0]] <= star[1][0])]
        includeInd = tmp[index].isin(keepIndex)
        tmp = tmp.loc[includeInd, :]
    
    qH = tmp.pivot(index=index, columns=columns, values=qvalue)
    qH = qH.fillna(1)
    pH = tmp.pivot(index=index, columns=columns, values=pvalue)
    pH = pH.fillna(1)
    """Assumes that stat is normally distributed around 0"""
    statH = tmp.pivot(index=index, columns=columns, values=stat)
    statH = statH.fillna(0)
    pdata = {pvalue:pH, qvalue:qH, stat:statH}

    if statH.shape[0] == 0:
        return

    if indexOrder is None:
        Z = sch.linkage(statH.values, method='complete')
        dend = sch.dendrogram(Z, no_plot=True)
        indexOrder = statH.index[dend['leaves']]
    else:
        kept = pdata[pvalue].index.tolist()
        indexOrder = [ss for ss in indexOrder if ss in kept]

    for k in pdata.keys():
        pdata[k] = pdata[k].loc[indexOrder]
    
    censorInd = pdata[censor[0]] < censor[1]
    pdata[stat].values[censorInd] = 0
    for k in [pvalue, qvalue]:
        pdata[k].values[censorInd] = 1

    # cmap = palettable.colorbrewer.diverging.RdGy_9.mpl_colormap
    # cmap = palettable.colorbrewer.diverging.RdYlGn_9_r.mpl_colormap
    # cmap = palettable.colorbrewer.diverging.RdBu_9_r.mpl_colormap
    cmap = palettable.colorbrewer.diverging.PuOr_9_r.mpl_colormap
    
    if vRange is None:
        """Assumes that stat is normally distributed around 0"""
        mnmx = tmp[stat].abs().max()
        pcParams = dict(vmin=-mnmx, vmax=mnmx, cmap=cmap)
    else:
        pcParams = dict(vmin=vRange[0], vmax=vRange[1], cmap=cmap)
    
    nPanels = (pdata[qvalue].shape[0] // splitN) + 1
    figh = plt.gcf()
    plt.clf()

    if scalebar:
        left = 0.5
    else:
        left = 0.15
    gs = plt.GridSpec(1, nPanels, left=left, bottom=0.02, right=0.95, top=0.95, wspace=0.3)
    for i in range(nPanels):
        tmp = {k:v.iloc[int(splitN*i) : int((i+1) * splitN)] for k,v in pdata.items()}
        axh = figh.add_subplot(gs[0, i])
        axh.grid(None)
        pcolOut = plt.pcolormesh(tmp[stat].values, **pcParams)
        plt.yticks(np.arange(tmp[qvalue].shape[0]) + 0.5, tmp[qvalue].index, size=tickSz)
        plt.xticks(np.arange(tmp[qvalue].shape[1]) + 0.5, tmp[qvalue].columns, size=tickSz, rotation=90)
        axh.xaxis.set_ticks_position('top') 
        plt.xlim((0, tmp[qvalue].shape[1]))
        plt.ylim((0, tmp[qvalue].shape[0]))
        plt.xlabel(columns)
        if i == 0:
            plt.ylabel(index)
        
        axh.invert_yaxis()

    for rowi,row in enumerate(tmp[star[0]].index):
        for coli,col in enumerate(tmp[star[0]].columns):
            numStars = (tmp[star[0]].loc[row, col] < np.array(star[1])).sum()
            #ann = unichr(0x204e) * numStars
            ann = '+' * numStars
            if not ann == '':
                plt.annotate(ann,
                             xy=(coli+0.5, rowi+0.5),
                             weight='bold',
                             size=tickSz,
                             ha='center',
                             va='center')
    """Scale colorbar"""
    if scalebar:
        scaleLabel = statLabel
        scaleAxh = figh.add_subplot(plt.GridSpec(1, 1, left=0.1, bottom=0.80, right=0.2, top=0.98)[0,0])
        cb = figh.colorbar(pcolOut, cax=scaleAxh)
        cb.set_label(scaleLabel, size=tickSz)
        # cb.ax.set_yticklabels(ytl, fontsize=8)

def plotVolcano(df, pvalueCol, fcCol, pThresh=0.05, fcThresh=1.5, annotate=False):
    pFunc = lambda p: -10*np.log10(p)
    sigInd = (df[pvalueCol] <= pThresh) & (np.abs(df[fcCol]) >= fcThresh)
    mxFC = np.max(np.abs(df[fcCol]))
    plt.clf()
    plt.scatter(df.loc[sigInd, fcCol],
                df.loc[sigInd, pvalueCol].map(pFunc),
                color='r', alpha=0.5, s=5)
    plt.scatter(df.loc[~sigInd, fcCol],
                df.loc[~sigInd, pvalueCol].map(pFunc),
                color='black', alpha=0.3, s=5)
    plt.plot([-1.1*mxFC, mxFC*1.1], [pFunc(pThresh)]*2, '--k')

    if not annotate is None:
        tmp = df[fcCol].loc[sigInd].sort_values(ascending=False)
        bottomN = tmp.index[-annotate:].tolist()
        topN = tmp.index[:annotate].tolist()
        for i in topN + bottomN:
            xy = (df[[fcCol, pvalueCol]].loc[i]).values
            xy[1] = pFunc(xy[1])
            if xy[0] > 0:
                p = dict(ha='left',
                         va='bottom',
                         xytext=(5, 5))
            else:
                p = dict(ha='right',
                         va='bottom',
                         xytext=(-5, 5))
            plt.annotate(i,
                         xy=xy,
                         textcoords='offset points',
                         size=8,
                         **p)

    plt.xlim((-1.1*mxFC, mxFC*1.1))
    yt = plt.yticks()[0]
    plt.yticks(yt, ['%1.1g' % (10**(t/-10)) for t in yt])
    plt.ylim((1, yt[-1]))
    plt.ylabel(pvalueCol)
    plt.xlabel(fcCol)