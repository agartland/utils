import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec

import palettable
import pandas as pd
import scipy.spatial.distance as distance
import scipy.cluster.hierarchy as sch
from sklearn.cluster.bicluster import SpectralBiclustering, SpectralCoclustering
import numpy as np
import itertools

from corrplots import scatterfit

__all__ = ['plotHCluster',
            'plotHColCluster',
            'plotCorrHeatmap',
            'mapColors2Labels',
            'computeDMat',
            'computeHCluster',
            'plotBicluster',
            'labeledDendrogram',
            'clusterOrder']

def clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in list(ax.spines.values()):
        sp.set_visible(False)
    ax.grid(False)
    ax.set_facecolor('white')

def mapColors2Labels(labels, setStr='Set3', cmap=None, returnLookup=False):
    """Return pd.Series of colors based on labels"""
    if cmap is None:
        N = max(3, min(12, len(np.unique(labels))))
        cmap = palettable.colorbrewer.get_map(setStr, 'Qualitative', N).mpl_colors
    cmapLookup = {k:col for k, col in zip(sorted(np.unique(labels)), itertools.cycle(cmap))}
    if returnLookup:
        return labels.map(cmapLookup.get), cmapLookup
    else:
        return labels.map(cmapLookup.get)

def computeDMat(df, metric=None, minN=1, dfunc=None):
    if dfunc is None:
        if metric in ['spearman', 'pearson']:
            """Anti-correlations are also considered as high similarity and will cluster together"""
            """dmat = 1 - df.corr(method = metric, min_periods = minN).values
            dmat[np.isnan(dmat)] = 1
            """
            dmat = 1 - df.corr(method = metric, min_periods = minN).values**2
            dmat[np.isnan(dmat)] = 1
        elif metric in ['spearman-signed', 'pearson-signed']:
            """Anti-correlations are considered as dissimilar and will NOT cluster together"""
            dmat = (1 - df.corr(method = metric.replace('-signed', ''), min_periods = minN).values) / 2
            dmat[np.isnan(dmat)] = 1
        else:
            dmat = distance.squareform(distance.pdist(df.T, metric = metric))
    else:
        ncols = df.shape[1]
        dmat = np.zeros((ncols, ncols))
        for i in range(ncols):
            for j in range(ncols):
                """Assume its symetrical"""
                if i<=j:
                    tmpdf = df.iloc[:, [i, j]]
                    tmpdf = tmpdf.dropna()
                    if tmpdf.shape[0] >= minN:
                        d = dfunc(df.iloc[:, i], df.iloc[:, j])
                    else:
                        d = np.nan
                    dmat[i, j] = d
                    dmat[j, i] = d
    assert dmat.shape[0] == dmat.shape[1]
    assert dmat.shape[0] == df.shape[1]
    return dmat

def clusterOrder(df, axis=0, metric='correlation', method='complete'):
    if axis == 0:
        dvec = distance.pdist(df, metric=metric)
    else:
        dvec = distance.pdist(df.T, metric=metric)
    
    clusters = sch.linkage(dvec, method=method)
    den = sch.dendrogram(clusters, color_threshold=np.inf, no_plot=True)
    
    if axis == 0:
        order = df.index[den['leaves']].tolist()
    else:
        order = df.T.index[den['leaves']].tolist()
    return order

def computeHCluster(dmat, method='complete'):
    """Compute dmat, clusters and dendrogram of df using
    the linkage method and distance metric given"""
    if dmat.shape[0] == dmat.shape[1]:
        if type(dmat) is pd.DataFrame:
            #compressedDmat = dmat.values[np.triu_indices_from(dmat.values)].ravel()
            compressedDmat = distance.squareform(dmat.values)
        else:
            #compressedDmat = dmat[np.triu_indices_from(dmat)].ravel()
            compressedDmat = distance.squareform(dmat)
    else:
        raise
    clusters = sch.linkage(compressedDmat, method=method)
    den = sch.dendrogram(clusters, color_threshold=np.inf, no_plot=True)
    return clusters, den

def testData(rows=50,columns=20):
    data = np.random.multivariate_normal(rand(columns), rand(columns, columns), rows)
    df = pd.DataFrame(data, columns=[''.join([lett]*9) for lett in 'ABCDEFGHIJKLMNOPQRST'])
    rowLabels = pd.Series(rand(rows).round(), index=df.index)
    columnLabels = pd.Series(rand(columns).round(), index=df.columns)
    return {'df':df,'row_labels':rowLabels,'col_labels':columnLabels}

def addColorbar(fig,cb_ax,data_ax,label='Correlation'):
    """Colorbar"""
    cb = fig.colorbar(data_ax, cb_ax) # note that we could pass the norm explicitly with norm=my_norm
    cb.set_label(label)
    """Make colorbar labels smaller"""
    for t in cb.ax.yaxis.get_ticklabels():
        t.set_fontsize('small')

def plotCorrHeatmap(df=None, metric='pearson', rowInd=None, colInd=None, col_labels=None, titleStr=None, vRange=None, tickSz='large', cmap=None, dmat=None, cbLabel='Correlation', minN=1):
    """Plot a heatmap of a column-wise distance matrix defined by metric (can be 'spearman' as well)
    Can provide dmat as a pd.DataFrame instead of df.
    Optionally supply a column index colInd to reorder the columns to match a previous clustering
    Optionally, col_labels will define a color strip along the yaxis to show groups"""

    fig = plt.gcf()
    fig.clf()

    if dmat is None and df is None:
        print('Need to provide df or dmat')
        return
    elif df is None:
        rowLabels = dmat.index
        columnLabels = dmat.columns
        dmat = dmat.values
    elif dmat is None:
        dmat = computeDMat(df, metric, minN=minN)
        rowLabels = df.columns
        columnLabels = df.columns

    if cmap is None:
        cmap = palettable.colorbrewer.diverging.RdBu_11_r.mpl_colormap

    if colInd is None:
        colInd = np.arange(dmat.shape[1])
    if rowInd is None:
        rowInd = colInd

    if col_labels is None:
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.05, right=0.78, top=0.85)[0, 0])
        scale_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.87, bottom=0.05, right=0.93, top=0.85)[0, 0])
    else:
        col_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.05, right=0.08, top=0.85)[0, 0])
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.11, bottom=0.05, right=0.78, top=0.85)[0, 0])
        scale_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.87, bottom=0.05, right=0.93, top=0.85)[0, 0])
    
    if vRange is None:
        vmin, vmax = (-1, 1)
        #vmin = dmat.flatten().min()
        #vmax = dmat.flatten().max()
    else:
        vmin, vmax = vRange
    my_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    """Column label colorbar but along the rows"""
    if not col_labels is None:
        col_cbSE = mapColors2Labels(col_labels)
        col_axi = col_cbAX.imshow([[x] for x in col_cbSE.iloc[rowInd].values],
                                  interpolation='nearest',
                                  aspect='auto',
                                  origin='lower')
        clean_axis(col_cbAX)

    """Heatmap plot"""
    axi = heatmapAX.imshow(dmat[rowInd,:][:, colInd],
                           interpolation='nearest',
                           aspect='auto',
                           origin='lower',
                           norm=my_norm,
                           cmap=cmap)
    clean_axis(heatmapAX)

    """Column tick labels along the rows"""
    if tickSz is None:
        heatmapAX.set_yticks([])
        heatmapAX.set_xticks([])
    else:
        heatmapAX.set_yticks(np.arange(dmat.shape[1]))
        heatmapAX.yaxis.set_ticks_position('right')
        heatmapAX.set_yticklabels(rowLabels[colInd], fontsize=tickSz, fontname='Consolas')

        """Column tick labels"""
        heatmapAX.set_xticks(np.arange(dmat.shape[1]))
        heatmapAX.xaxis.set_ticks_position('top')
        xlabelsL = heatmapAX.set_xticklabels(columnLabels[colInd], fontsize=tickSz, rotation=90, fontname='Consolas')

        """Remove the tick lines"""
        for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
            l.set_markersize(0)

    addColorbar(fig, scale_cbAX, axi, label=cbLabel)
    
    """Add title as xaxis label"""
    if not titleStr is None:
        heatmapAX.set_xlabel(titleStr, size='x-large')

def plotHColCluster(df=None, col_dmat=None, method='complete', metric='euclidean', col_labels=None, titleStr=None, vRange=None, tickSz='medium', cmap=None,  minN=1, K=None, labelCmap=None, noColorBar=False, interactive=False):
    """Perform hierarchical clustering on df columns and plot square heatmap of pairwise distances"""
    if col_dmat is None and df is None:
        print('Need to provide df or col_dmat')
        return
    elif df is None:
        columnLabels = col_dmat.columns
        col_dmat = col_dmat.values
        colorbarLabel = ''
        col_plot = col_dmat
    elif col_dmat is None:
        col_dmat = computeDMat(df, metric, minN=minN)
        columnLabels = df.columns

        if metric in ['spearman', 'pearson', 'spearman-signed', 'pearson-signed']:
            """If it's a correlation metric, plot Rho not the dmat"""
      
            colorbarLabel = 'Correlation coefficient'
            if metric in ['spearman-signed', 'pearson-signed']:
                col_plot = df.corr(method=metric.replace('-signed', ''), min_periods=minN).values
            else:
                col_plot = df.corr(method=metric, min_periods=minN).values
        else:
            colorbarLabel = ''
            col_plot = col_dmat
    else:
        col_plot = col_dmat
        columnLabels = df.columns
        colorbarLabel = ''

    nCols = col_dmat.shape[1]

    if cmap is None:
        if metric in ['spearman', 'pearson', 'spearman-signed', 'pearson-signed']:
            cmap = palettable.colorbrewer.diverging.RdBu_11_r.mpl_colormap
        else:
            cmap = palettable.colorbrewer.sequential.YlOrRd_9.mpl_colormap
    
    col_clusters, col_den = computeHCluster(col_dmat, method)

    if col_labels is None and not K is None:
        col_labels = pd.Series(sch.fcluster(col_clusters, K, criterion='maxclust'), index=columnLabels)
    
    if isinstance(col_plot, pd.DataFrame):
        col_plot = col_plot.values

    if vRange is None:
        if metric in ['spearman', 'pearson', 'spearman-signed', 'pearson-signed']:
            vmin, vmax = (-1, 1)
        else:
            vmin = col_plot.min()
            vmax = col_plot.max()
    else:
        vmin, vmax = vRange

    my_norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    fig = plt.gcf()
    fig.clf()
    #heatmapGS = gridspec.GridSpec(1,4,wspace=0.0,width_ratios=[0.25,0.01,2,0.15])
    if col_labels is None and K is None:
        col_denAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.05, right=0.15, top=0.85)[0, 0])
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.16, bottom=0.05, right=0.75, top=0.85)[0, 0])
        if not noColorBar:
            scale_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.94, bottom=0.05, right=0.97, top=0.85)[0, 0])
    else:
        """TODO: work on row_cbAX so that I can have the data labels on the top and left"""
        col_denAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.05, right=0.15, top=0.85)[0, 0])
        col_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.16, bottom=0.05, right=0.19, top=0.85)[0, 0])
        #row_cbAX = fig.add_subplot(GridSpec(1,1,left=0.2,bottom=0.83,right=0.75,top=0.86)[0,0])
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.2, bottom=0.05, right=0.75, top=0.85)[0, 0])
        if not noColorBar:
            scale_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.94, bottom=0.05, right=0.97, top=0.85)[0, 0])
    
    """Column dendrogaram but along the rows"""
    plt.sca(col_denAX)
    col_denD = sch.dendrogram(col_clusters, color_threshold=np.inf, orientation='left')
    colInd = col_denD['leaves']
    clean_axis(col_denAX)

    """Column label colorbar but along the rows"""
    if not col_labels is None:
        col_cbSE = mapColors2Labels(col_labels, cmap=labelCmap)
        col_axi = col_cbAX.imshow([[x] for x in col_cbSE.iloc[colInd].values], interpolation='nearest', aspect='auto', origin='lower')
        clean_axis(col_cbAX)

    """Heatmap plot"""
    axi = heatmapAX.imshow(col_plot[colInd,:][:, colInd],
                           interpolation='nearest',
                           aspect='auto',
                           origin='lower',
                           norm=my_norm,
                           cmap=cmap)
    clean_axis(heatmapAX)

    """Column tick labels along the rows"""
    if tickSz is None:
        heatmapAX.set_yticks(())
        heatmapAX.set_xticks(())
    else:
        heatmapAX.set_yticks(np.arange(nCols))
        heatmapAX.yaxis.set_ticks_position('right')
        heatmapAX.set_yticklabels(columnLabels[colInd], fontsize=tickSz, fontname='Consolas')

        """Column tick labels"""
        heatmapAX.set_xticks(np.arange(nCols))
        heatmapAX.xaxis.set_ticks_position('top')
        xlabelsL = heatmapAX.set_xticklabels(columnLabels[colInd], fontsize=tickSz, rotation=90, fontname='Consolas')

        """Remove the tick lines"""
        for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
            l.set_markersize(0)
    if not noColorBar:
        addColorbar(fig, scale_cbAX, axi, label=colorbarLabel)

    """Add title as xaxis label"""
    if not titleStr is None:
        heatmapAX.set_xlabel(titleStr, size='x-large')

    if interactive and not df is None:
        scatterFig = plt.figure(fig.number + 100)
        ps = PairScatter(df.iloc[:, colInd], heatmapAX, scatterFig.add_subplot(111), method=metric)
        return colInd, ps

    return colInd

def plot1DHClust(distDf, hclusters, labels=None, titleStr=None, vRange=None, tickSz='small', cmap=None, colorbarLabel=None, labelCmap=None, noColorBar=False):
    """Plot hierarchical clustering results (no computation)
    I'm not even sure this is useful..."""
    if cmap is None:
        cmap = palettable.colorbrewer.sequential.YlOrRd_9.mpl_colormap
    fig = plt.gcf()
    fig.clf()

    nCols = distDf.shape[0]
    
    if labels is None:
        col_denAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.05, right=0.15, top=0.85)[0, 0])
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.16, bottom=0.05, right=0.78, top=0.85)[0, 0])
        if not noColorBar:
            scale_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.87, bottom=0.05, right=0.93, top=0.85)[0, 0])
    else:
        col_denAX = fig.add_subplot(GridSpec(1, 1, left=0.05, bottom=0.05, right=0.15, top=0.85)[0, 0])
        col_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.16, bottom=0.05, right=0.19, top=0.85)[0, 0])
        heatmapAX = fig.add_subplot(GridSpec(1, 1, left=0.2, bottom=0.05, right=0.78, top=0.85)[0, 0])
        if not noColorBar:
            scale_cbAX = fig.add_subplot(GridSpec(1, 1, left=0.87, bottom=0.05, right=0.93, top=0.85)[0, 0])
    
    if vRange is None:
        vmin = distDf.values.min()
        vmax = distDf.vlaues.max()
    else:
        vmin, vmax = vRange

    my_norm = mpl.colors.Normalize(vmin = vmin, vmax = vmax)

    """Column dendrogaram but along the rows"""
    plt.axes(col_denAX)

    colInd = hclusters['leaves']
    clean_axis(col_denAX)

    imshowOptions = dict(interpolation = 'nearest', aspect = 'auto', origin = 'lower')

    """Column label colorbar but along the rows"""
    if not labels is None:
        col_cbSE = mapColors2Labels(labels, cmap = labelCmap)
        col_axi = col_cbAX.imshow([[x] for x in col_cbSE.iloc[colInd].values], **imshowOptions)
        clean_axis(col_cbAX)

    """Heatmap plot"""
    axi = heatmapAX.imshow(distDf.values[colInd,:][:, colInd], norm = my_norm, cmap = cmap, **imshowOptions)
    clean_axis(heatmapAX)

    """Column tick labels along the rows"""
    if tickSz is None:
        heatmapAX.set_yticks(())
        heatmapAX.set_xticks(())
    else:
        heatmapAX.set_yticks(np.arange(nCols))
        heatmapAX.yaxis.set_ticks_position('right')
        heatmapAX.set_yticklabels(distDf.columns[colInd], fontsize=tickSz, fontname='Consolas')

        """Column tick labels"""
        heatmapAX.set_xticks(np.arange(nCols))
        heatmapAX.xaxis.set_ticks_position('top')
        xlabelsL = heatmapAX.set_xticklabels(distDf.columns[colInd], fontsize=tickSz, rotation=90, fontname='Consolas')

        """Remove the tick lines"""
        for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
            l.set_markersize(0)
    if not noColorBar:
        addColorbar(fig, scale_cbAX, axi, label=colorbarLabel)

    """Add title as xaxis label"""
    if not titleStr is None:
        heatmapAX.set_xlabel(titleStr, size='x-large')

def plotHCluster(df, method='complete', metric='euclidean', clusterBool=[True, True],row_labels=None, col_labels=None, vRange=None,titleStr=None,xTickSz='small',yTickSz='small',cmap=None,minN=1):
    """Perform hierarchical clustering on df data columns (and rows) and plot results as
    dendrograms and heatmap.

    df - pd.DataFrame(), will use index and column labels as tick labels
    method and metric - parameters passed to scipy.spatial.distance.pdist and scipy.cluster.hierarchy.linkage
    row_labels - pd.Series with index same as df with values indicating groups (optional)
    col_labels - pd.Series with index same as columns in df with values indicating groups (optional)
    vMinMax - optional scaling, [vmin, vmax] can be derived from data
    clusterBool - [row, col] bool indicating whether to cluster along that axis
    """
    if cmap is None:
        cmap = palettable.colorbrewer.diverging.RdBu_11_r.mpl_colormap

    if vRange is None:
        vmin = df.min().min()
        vmax = df.max().max()
    else:
        vmin, vmax = vRange
    my_norm = mpl.colors.Normalize(vmin, vmax)

    fig = plt.gcf()
    fig.clf()
    if clusterBool[1]:
        heatmapGS = gridspec.GridSpec(3, 3, wspace=0.0, hspace=0.0, width_ratios=[0.15, 0.02, 1], height_ratios=[0.15, 0.02, 1])
    else:
        heatmapGS = gridspec.GridSpec(3, 3, wspace=0.0, hspace=0.0, width_ratios=[0.15, 0.02, 1], height_ratios=[0.001, 0.02, 1])

    if clusterBool[0]:
        row_dmat = computeDMat(df.T, metric, minN=minN)
        row_clusters, row_den = computeHCluster(row_dmat, method)

        """Dendrogarams"""
        row_denAX = fig.add_subplot(heatmapGS[2, 0])
        row_denD = sch.dendrogram(row_clusters, color_threshold=np.inf, orientation='left')
        clean_axis(row_denAX)

        rowInd = row_denD['leaves']
    else:
        rowInd = np.arange(df.shape[0])

    """Row colorbar"""
    if not row_labels is None:
        """NOTE: row_labels will not be index aware and must be in identical order as data"""
        row_cbSE = mapColors2Labels(row_labels, 'Set1')
        row_cbAX = fig.add_subplot(heatmapGS[2, 1])

        row_axi = row_cbAX.imshow([[x] for x in row_cbSE.iloc[rowInd].values], interpolation='nearest', aspect='auto', origin='lower')
        clean_axis(row_cbAX)

    if clusterBool[1]:
        col_dmat = computeDMat(df, metric, minN=minN)
        col_clusters, col_den = computeHCluster(col_dmat, method)

        """Dendrogarams"""
        col_denAX = fig.add_subplot(heatmapGS[0, 2])
        col_denD = sch.dendrogram(col_clusters, color_threshold=np.inf)
        clean_axis(col_denAX)

        
        colInd = col_denD['leaves']
    else:
        colInd = np.arange(df.shape[1])

    """Column colorbar"""
    if not col_labels is None:
        col_cbSE = mapColors2Labels(col_labels)
        col_cbAX = fig.add_subplot(heatmapGS[1, 2])
        col_axi = col_cbAX.imshow([list(col_cbSE.iloc[colInd])], interpolation='nearest', aspect='auto', origin='lower')
        clean_axis(col_cbAX)
    
    """Heatmap plot"""
    heatmapAX = fig.add_subplot(heatmapGS[2, 2])
    axi = heatmapAX.imshow(df.iloc[rowInd, colInd], interpolation='nearest', aspect='auto', origin='lower', norm=my_norm, cmap=cmap)
    clean_axis(heatmapAX)
    heatmapAX.grid(False)

    """Row tick labels"""
    heatmapAX.set_yticks(np.arange(df.shape[0]))
    ylabelsL = None
    if not yTickSz is None:
        heatmapAX.yaxis.set_ticks_position('right')
        ylabelsL = heatmapAX.set_yticklabels(df.index[rowInd], fontsize=yTickSz, fontname='Consolas')
    else:
        ylabelsL = heatmapAX.set_yticklabels([])

    """Add title as xaxis label"""
    if not titleStr is None:
        heatmapAX.set_xlabel(titleStr, size='x-large')

    """Column tick labels"""
    heatmapAX.set_xticks(np.arange(df.shape[1]))
    xlabelsL = None
    if not xTickSz is None:
        xlabelsL = heatmapAX.set_xticklabels(df.columns[colInd], fontsize=xTickSz, rotation=90, fontname='Consolas')

    """Remove the tick lines"""
    for l in heatmapAX.get_xticklines() + heatmapAX.get_yticklines(): 
        l.set_markersize(0)

    """Colorbar"""
    scaleGS = gridspec.GridSpec(10, 15, wspace=0., hspace=0.)
    scale_cbAX = fig.add_subplot(scaleGS[:2, 0]) # colorbar for scale in upper left corner
    cb = fig.colorbar(axi, scale_cbAX) # note that we could pass the norm explicitly with norm=my_norm
    cb.set_label('Measurements')
    cb.ax.yaxis.set_ticks_position('left') # move ticks to left side of colorbar to avoid problems with tight_layout
    cb.ax.yaxis.set_label_position('left') # move label to left side of colorbar to avoid problems with tight_layout
    #cb.outline.set_linewidth(0)
    """Make colorbar labels smaller"""
    for t in cb.ax.yaxis.get_ticklabels():
        t.set_fontsize('small')
    scaleGS.tight_layout(fig, h_pad=0.0, w_pad=0.0)

    heatmapGS.tight_layout(fig, h_pad=0.1, w_pad=0.5)

    handles = dict(cb=cb, heatmapAX=heatmapAX, fig=fig, xlabelsL=xlabelsL, ylabelsL=ylabelsL, heatmapGS=heatmapGS)
    return rowInd, colInd, handles

def plotBicluster(df, n_clusters, col_labels=None):
    model = SpectralBiclustering(n_clusters=n_clusters, method='log', random_state=0)
    model.fit(df)
    
    fitDf = df.iloc[np.argsort(model.row_labels_),:]
    fitDf = fitDf.iloc[:, np.argsort(model.column_labels_)]
    plotCorrHeatmap(dmat=fitDf, col_labels=col_labels)
    return fitDf

def normalizeAxis(df,axis=0,useMedian=False):
    """Normalize along the specified axis by
    subtracting the mean and dividing by the stdev.

    Uses df functions that ignore NAs

    Parameters
    ----------
    df : pd.DataFrame
    axis : int
        Normalization along this axis. (e.g. df.mean(axis=axis))

    Returns
    -------
    out : pd.DataFrame"""

    tmp = df.copy()
    retile = ones(len(df.shape))
    retile[axis] = df.shape[axis]
    if useMedian:
        tmp = tmp - tile(tmp.median(axis=axis).values, retile)
    else:
        tmp = tmp - tile(tmp.mean(axis=axis).values, retile)
    tmp = tmp / tile(tmp.std(axis=axis).values, retile)
    return tmp

class PairScatter:
    """Instantiate this class to interactively pair
    a heatmap and a pairwise scatterfit plot in a new figure window."""
    def __init__(self, df, heatmapAx, scatterAx, method):
        self.scatterAx = scatterAx
        self.heatmapAx = heatmapAx
        self.df = df
        self.method = method
        self.cid = heatmapAx.figure.canvas.mpl_connect('button_press_event', self)
    def __call__(self, event):
        if event.inaxes != self.heatmapAx:
            return
        else:
            xind = int(np.floor(event.xdata + 0.5))
            yind = int(np.floor(event.ydata + 0.5))
            plt.sca(self.scatterAx)
            plt.cla()
            scatterfit(self.df.iloc[:, xind], self.df.iloc[:, yind], method = self.method, plotLine = True)
            self.scatterAx.figure.show()

def labeledDendrogram(dmat, labels, method='complete', cmap=None):
    """Perform hierarchical clustering on df columns and plot square heatmap of pairwise distances"""
    """TODO: add tick labels, with sparsity option"""

    Z = sch.linkage(dmat, method=method)
    den = sch.dendrogram(Z, color_threshold=np.inf, no_plot=True)

    figh = plt.gcf()
    figh.clf()

    denAX = figh.add_axes([0.32, 0.05, 0.6, 0.9])
    cbAX =  figh.add_axes([0.25, 0.05, 0.05, 0.9])

    plt.sca(denAX)
    denD = sch.dendrogram(Z, color_threshold=np.inf, orientation='left')
    ind = denD['leaves']
    clean_axis(denAX)
    
    cbSE, lookup = mapColors2Labels(labels, cmap=cmap, returnLookup=True)
    axi = cbAX.imshow([[x] for x in cbSE.iloc[ind].values],
                      interpolation='nearest',
                      aspect='auto',
                      origin='lower')
    clean_axis(cbAX)

    colorLegend(list(lookup.values()), list(lookup.keys()), axh=denAX)
