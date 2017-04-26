import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import polyfit, polyval, stats
import pandas as pd
from mytext import textTL, textTR
import statsmodels.api as sm
from patsy import dmatrices,ModelDesc,Term,LookupFactor
from copy import deepcopy
import itertools
import warnings
import palettable

from adjustwithin import adjustnonnan

__all__ = ['partialcorr',
           'combocorrplot',
           'scatterfit',
           'heatmap',
           'crosscorr',
           'pwpartialcorr',
           'corrheatmap',
           'validPairwiseCounts',
           'removeNARC',
           'permcorr',
           'labeledScatter']

"""Red --> Green colormap with 1024 interpolated values"""
_cdict = {'green' : ((0, 1, 1), (0.5, 0, 0), (1, 0, 0)),
          'red':    ((0, 0, 0), (0.5, 0, 0), (1, 1, 1)),
          'blue' :  ((0, 0, 0), (1, 0, 0))}
#_heatCmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', _cdict, 1024)
_heatCmap = palettable.colorbrewer.diverging.RdBu_11_r.mpl_colormap


def partialcorr(x, y, adjust=[], method='pearson', minN = None):
    """Finds partial correlation of x with y adjusting for variables in adjust

    This function is index aware (i.e. uses index of x, y and adjust for joining).
    Rho and p-value match those from stats.spearmanr, and stats.pearsonr when adjust = [].

    TODO:
        (1) Compute CIs
        (2) Make into its own testable module
        (3) Include partial_corr gist
        (4) Include function to compute whole partial correlation matrix
        (5) Add second method which takes correlation of residuals (should be equivalent, but is nice test)

    Parameters
    ----------
    x,y : pd.Series
        Each contains data for assessing correlation.
    adjust : list of pd.Series objects
        Correlation is assessed between x and y adjusting for all variables in z (default: [])
    method : string
        Method can be 'pearson' (default) or 'spearman', which uses rank-based correlation and adjustment.
    minN : int
        Minimum number of non-nan paired observations. If N < minN then returns pc = nan and p = 1
    
    Returns
    -------
    partial_rho : float
        Partial correlation coefficient between x and y after adjustment.
    pvalue : float
        P-value for the partial correlation coefficient."""

    if not isinstance(x, pd.Series):
        x = pd.Series(x, name = 'X')
    if not isinstance(y, pd.Series):
        y = pd.Series(y, name = 'Y')

    assert x.shape[0] == y.shape[0]
    if x.name == y.name:
        x.name += '_X'
        y.name += '_Y'

    """Make one big DataFrame out of x, y and adjustment variables"""
    tmpDf = pd.concat((x,y), join='inner', axis=1)
    for a in adjust:
        tmpDf = tmpDf.join(a, how='left')

    tmpDf = tmpDf.dropna(axis=0, how='any')

    if not minN is None and tmpDf.shape[0] < minN:
        return np.nan, 1.
    
    m = np.zeros((tmpDf.shape[0], 2+len(adjust)))
    
    if method == 'spearman':
        """Convert data to ranks"""
        m[:,0] = tmpDf[x.name].rank()
        m[:,1] = tmpDf[y.name].rank()
        for i,a in enumerate(adjust):
            m[:,i+2] = tmpDf[a.name].rank()
    else: 
        m[:,0] = tmpDf[x.name]
        m[:,1] = tmpDf[y.name]
        for i,a in enumerate(adjust):
            m[:,i+2] = tmpDf[a.name]
    
    if all(m[:,0] == m[:,1]):
        """Testing for perfect correlation avoids SingularMatrix exception"""
        return 1,0.0
    
    """Take the inverse of the covariance matrix including all variables
    pc = -p_ij / sqrt(p_ii * p_ij)
    where p is the inverse covariance matrix"""
    
    try:
        icv = np.linalg.inv(np.cov(m,rowvar=0))
        pc = -icv[0,1] / np.sqrt(icv[0,0] * icv[1,1])

        n = m.shape[0]
        gn = len(adjust)
        
        statistic = pc * np.sqrt((n-2-gn)/(1-pc**2))
        #pvalue = 2*stats.norm.cdf(-abs(statistic))

        #SAS and pearsonr look the statistic up in a t distribution while R uses the normnal

        pvalue = 2*stats.t.cdf(-np.abs(statistic),n-2-gn)
    except:
        """These were used to check that non-partial rho's and pvalues match those of their scipy equivalents
        They do! Use them if the other fails and warn the caller"""
        if method == 'pearson':
            pc,pvalue = stats.pearsonr(tmpDf[x.name].values,tmpDf[y.name].values)
        else:
            pc,pvalue = stats.spearmanr(tmpDf[x.name].values,tmpDf[y.name].values)
        if len(adjust) > 0:
            warnings.warn('Error computing %s and %s correlation: using scipy equivalent to return UNADJUSTED results'   % (x.name,y.name))
        else:
            warnings.warn('Error computing %s and %s correlation: using scipy equivalent'   % (x.name,y.name))
        #raise
    
    """Below verifies that the p-value for the coefficient in the multivariate model including adjust
    is the same as the p-value of the partial correlation"""
    
    """formula_like=ModelDesc([Term([LookupFactor(y.name)])],[Term([]),Term([LookupFactor(x.name)])]+[Term([LookupFactor(a.name)]) for a in adjust])
    Y, X = dmatrices(formula_like, data=tmpDf, return_type='dataframe')
    model=sm.GLM(Y,X,family=sm.families.Gaussian())
    print model.fit().summary()"""

    return pc, pvalue

def combocorrplot(data,method='spearman',axLimits='variable',axTicks=False,axTicklabels=False,valueFlag=True,ms=2, plotLine = False):
    """Shows correlation scatter plots in combination with a heatmap for small sets of variables.

    Parameters
    ----------
    data : pd.DataFrame
    method : string
        Correlation method, can be 'pearson' or 'spearman'
    axLimits : string
        If 'variable' then allows the limits to be different for each pair of variables.
    axTicks : bool
        Display axis tick marks on each square?
    axTicklabels : bool
        Display axis tick labels on each square?
    valueFlag : bool
        Display correlation coefficient in each square?
    ms : int
        Scatter plot marker size in points.
    plotLine : bool
        Plot fit-line on the subplots?"""

    border = 0.05
    pad = 0.02
    cbwidth = 0.1

    labels = data.columns

    """Use pd.DataFrame method to compute the pairwise correlations"""
    coef = data.corr(method=method)
    n = coef.shape[0]


    axh = np.empty((n,n), dtype=object)
    plth = np.empty((n,n), dtype=object)

    mx = None
    mn = None
    for col in data.columns:
        if mx==None:
            mx = data[col].max()
            mn = data[col].min()
        mx = max(data[col].max(),mx)
        mn = min(data[col].min(),mn)

    plt.clf()
    fh = plt.gcf()
    gs = GridSpec(n,n,
                  left=border,
                  bottom=border,
                  right=1.-border-cbwidth,
                  top=1.-border,
                  wspace=pad,
                  hspace=pad)
    #cbgs=GridSpec(1,1,left=1.-cbwidth,bottom=border,right=1.-border,top=1.-border,wspace=pad,hspace=pad)
    for r in range(n):
        for c in range(n):
            if r == c:
                axh[r,c] = fh.add_subplot(gs[r,c],yticklabels=[],xticklabels=[],xticks=[],yticks=[], axisbg = 'gray')
                plt.text(0,0,'%s' % (data.columns[r]),ha='center',va='center')
                plt.axis([-1,1,-1,1])
            elif r>c:
                if axTicks:
                    if axTicklabels:
                        if r < len(labels)-1 and c>0:
                            axh[r,c] = fh.add_subplot(gs[r,c],xticklabels=[],yticklabels=[])
                        elif r < len(labels)-1 and c==0:
                            axh[r,c] = fh.add_subplot(gs[r,c],xticklabels=[])
                        elif r == len(labels)-1 and c>0:
                            axh[r,c] = fh.add_subplot(gs[r,c],yticklabels=[])
                        elif r == len(labels)-1 and c==0:
                            axh[r,c] = fh.add_subplot(gs[r,c])
                    else:
                        axh[r,c] = fh.add_subplot(gs[r,c],xticklabels=[],yticklabels=[])

                else:
                    axh[r,c] = fh.add_subplot(gs[r,c],xticks=[],yticks=[])
                plotx = data[labels[r]]
                ploty = data[labels[c]]
                validInd = (~np.isnan(plotx)) & (~np.isnan(ploty))
                plotx,ploty = plotx[validInd], ploty[validInd]
                if method == 'pearson' and plotLine:
                    ar,br = polyfit(plotx,ploty,1)
                    xfit = np.array([min(plotx),max(plotx)])
                    yfit = polyval([ar,br],xfit)
                    plt.plot(xfit, yfit, '-', lw=1, color='gray')
                plt.plot(plotx, ploty, 'ok', ms=ms)
                
                if axLimits == 'variable':
                    rmax,rmin = max(plotx), min(plotx)
                    cmax,cmin = max(ploty), min(ploty)
                else:
                    rmax,cmax = mx,mx
                    rmin,cmin = mn,mn

                plt.axis([rmin-0.1*(rmax-rmin), rmax+0.1*(rmax-rmin),cmin-0.1*(cmax-cmin), cmax+0.1*(cmax-cmin)])
            elif r < c:
                axh[r,c] = fh.add_subplot(gs[r,c], yticklabels=[], xticklabels=[], xticks=[], yticks=[])
                val = coef[labels[r]][labels[c]]
                plth[r,c] = plt.pcolor(np.ones((2,2))*val, cmap=_heatCmap, vmin=-1., vmax=1.)
                plt.axis([0,1,0,1])
                if valueFlag:
                    if val<0.75 and val>-0.75:
                        txtcol = 'black'
                    else:
                        txtcol = 'white'
                    plt.text(0.5, 0.5, '%1.2f' % (val),
                             ha='center',
                             va='center',
                             family='monospace',
                             color=txtcol,
                             weight='bold',
                             size='large')
    cbax = fh.add_axes([1.-cbwidth-border/2, border, cbwidth-border-0.02, 1.-2*border])
    cb = plt.colorbar(plth[0,0],cax=cbax)
    method = method[0].upper() + method[1:]
    plt.annotate('%s correlation' % (method),
                 [0.98,0.5],
                 xycoords='figure fraction',
                 ha='right',
                 va='center',
                 rotation='vertical',
                 size='large')

def pwpartialcorr(df, rowVars=None, colVars=None, adjust=[], method='pearson', minN=0, adjMethod='fdr_bh'):
    """Pairwise partial correlation.

    Parameters
    ----------
    df : pd.DataFrame [samples, variables]
        Data for correlation assessment (Nans will be ignored for each column pair)
    rowVars, colVars : lists
        List of column names to incude on heatmap axes.
    adjust : list
        List of column names that will be adjusted for in the pairwise correlations.
    method : string
        Specifies whether a pearson or spearman correlation is performed. (default: 'spearman')
    minN : int
        If a correlation has fewer than minN samples after dropping Nans
        it will be reported as rho = 0, pvalue = 1 and will not be included in the multiplicity adjustment.

    Returns
    -------
    rho : pd.DataFrame [rowVars, colVars]
        Correlation coefficients.
    pvalue : pd.DataFrame [rowVars, colVars]
        Pvalues for pairwise correlations.
    qvalue : pd.DataFrame [rowVars, colVars]
        Multiplicity adjusted q-values for pairwise correlations.
    N : pd.DataFrame [rowVars, colVars]
        Number of non-nan value pairs in the computation."""

    if rowVars is None:
        rowVars = df.columns
    if colVars is None:
        colVars = df.columns

    pvalue = np.zeros((len(rowVars),len(colVars)))
    qvalue = np.nan * np.zeros((len(rowVars),len(colVars)))
    rho = np.zeros((len(rowVars),len(colVars)))
    N = np.zeros((len(rowVars),len(colVars)))

    """Store p-values in dict with keys that are unique pairs (so we only adjust across these)"""
    pairedPvalues = {}
    paireQPvalues = {}
    allColumns = df.columns.tolist()

    for i,rowv in enumerate(rowVars):
        for j,colv in enumerate(colVars):
            if not rowv == colv:
                N[i, j] = df[[rowv,colv]].dropna().shape[0]
                if not N[i, j] < minN:
                    rho[i,j],pvalue[i,j] = partialcorr(df[rowv],df[colv],adjust=[df[a] for a in adjust], method=method)
                else:
                    """Pvalue = nan excludes these from the multiplicity adjustment"""
                    rho[i,j],pvalue[i,j] = np.nan,np.nan
                """Define unique key for the pair by sorting in order they appear in df columns"""
                key = tuple(sorted([rowv,colv], key = allColumns.index))
                pairedPvalues.update({key:pvalue[i,j]})
            else:
                """By setting these pvalues to nan we exclude them from multiplicity adjustment"""
                rho[i,j],pvalue[i,j] = 1,np.nan
            
    """Now only adjust using pvalues in the unique pair dict"""
    keys = pairedPvalues.keys()
    qvalueTmp = adjustnonnan([pairedPvalues[k] for k in keys], method=adjMethod)
    """Build a unique qvalue dict from teh same unique keys"""
    pairedQvalues = {k:q for k,q in zip(keys,qvalueTmp)}
    
    """Assign the unique qvalues to the correct comparisons"""
    for i,rowv in enumerate(rowVars):
        for j,colv in enumerate(colVars):
            if not rowv == colv:
                key = tuple(sorted([rowv,colv], key = allColumns.index))
                qvalue[i,j] = pairedQvalues[key]
            else:
                pvalue[i,j] = 0.
                qvalue[i,j] = 0.
    pvalue = pd.DataFrame(pvalue, index=rowVars, columns=colVars)
    qvalue = pd.DataFrame(qvalue, index=rowVars, columns=colVars)
    rho = pd.DataFrame(rho, index=rowVars, columns=colVars)
    N = pd.DataFrame(N.astype(int), index=rowVars, columns=colVars)
    return rho, pvalue, qvalue, N

def crosscorr(dfA, dfB, method='pearson', minN=0, adjMethod='fdr_bh', returnLong=False):
    """Pairwise correlations between A and B after a join,
    when there are potential column name overlaps.

    Parameters
    ----------
    dfA,dfB : pd.DataFrame [samples, variables]
        DataFrames for correlation assessment (Nans will be ignored in pairwise correlations)
    method : string
        Specifies whether a pearson or spearman correlation is performed. (default: 'spearman')
    minN : int
        If a correlation has fewer than minN samples after dropping Nans
        it will be reported as rho = 0, pvalue = 1 and will not be included in the multiplicity adjustment.
    returnLong : bool
        If True, return one long-form DataFrame with rho, n, pvalue and qvalue as columns.

    Returns
    -------
    rho : pd.DataFrame [rowVars, colVars]
        Correlation coefficients.
    pvalue : pd.DataFrame [rowVars, colVars]
        Pvalues for pairwise correlations.
    qvalue : pd.DataFrame [rowVars, colVars]
        Multiplicity adjusted q-values for pairwise correlations.
    N  : pd.DataFrame [rowVars, colVars]
        Number of non-nan value pairs in the calculation."""
    colA = dfA.columns
    colB = dfB.columns
    dfA = dfA.rename_axis(lambda s: s + '_A', axis=1)
    dfB = dfB.rename_axis(lambda s: s + '_B', axis=1)

    joinedDf = pd.merge(dfA, dfB, left_index=True, right_index=True)

    rho, pvalue, qvalue, N = pwpartialcorr(joinedDf, rowVars=dfA.columns, colVars=dfB.columns, method=method, minN=minN, adjMethod=adjMethod)

    rho.index = colA
    rho.columns = colB

    pvalue.index = colA
    pvalue.columns = colB

    qvalue.index = colA
    qvalue.columns = colB

    N.index = colA
    N.columns = colB

    if returnLong:
        resDf = pd.DataFrame([pair for pair in itertools.product(rho.index, rho.columns)],
                             columns=['A', 'B'])
        resDf.loc[:, 'rho'] = rho.values.ravel()
        resDf.loc[:, 'N'] = N.values.ravel()
        resDf.loc[:, 'pvalue'] = pvalue.values.ravel()
        resDf.loc[:, 'qvalue'] = qvalue.values.ravel()
        return resDf
    else:
        return rho, pvalue, qvalue, N


def corrheatmap(df, rowVars=None, colVars=None, adjust=[], annotation=None, cutoff=None, cutoffValue=0.05, method='pearson', labelLookup={}, xtickRotate=True, labelSize='medium', minN=0, adjMethod='fdr_bh'):
    """Compute pairwise correlations and plot as a heatmap.

    Parameters
    ----------
    df : pd.DataFrame [samples, variables]
        Data for correlation assessment (Nans will be ignored for each column pair)
    rowVars, colVars : lists
        List of column names to incude on heatmap axes.
    adjust : list
        List of column names that will be adjusted for in the pairwise correlations.
    annotation : string
        Specify what is annotated in each square of the heatmap (e.g. pvalue, qvalue, rho, rho2)
    cutoff : str
        Specify how to apply cutoff (e.g. pvalue, qvalue, rho, rho2)
    cutoffValue : float
        Absolute minimum threshold for squares whose color is displayed (color is proportional to rho).
    method : string
        Specifies whether a pearson or spearman correlation is performed. (default: 'spearman')
    labelLookup : dict
        Used to translate column names into appropriate label strings.
    xtickRotate : bool
        Specify whether to rotate the labels along the x-axis
    labelSize : str or int
        Size of x- and y-ticklabels by string (e.g. "large") or points
    minN : int
        If a correlation has fewer than minN samples after dropping Nans
        it will be reported as rho = 0, pvalue = 1 and will not be included in the multiplicity adjustment.

    Returns
    -------
    rho : ndarray [samples, variables]
        Matrix of correlation coefficients.
    pvalue : ndarray [samples, variables]
        Matrix of pvalues for pairwise correlations.
    qvalue : ndarray [samples, variables]
        Matrix of multiplicity adjusted q-values for pairwise correlations."""
    if rowVars is None:
        rowVars = df.columns
    if colVars is None:
        colVars = df.columns
    if cutoff is None:
        cutoff = 'pvalue'

    rho,pvalue,qvalue,N = pwpartialcorr(df, rowVars=rowVars, colVars=colVars, adjust=adjust, method=method, minN=minN)
   
    plt.clf()
    fh = plt.gcf()
    pvalueTxtProp = dict(family='monospace',
                         size='large',
                         weight='bold',
                         color='white',
                         ha='center',
                         va='center')

    axh = fh.add_subplot(111, yticks = np.arange(len(rowVars))+0.5,
                              xticks = np.arange(len(colVars))+0.5)
    if xtickRotate:
        rotation = 'vertical'
    else:
        rotation = 'horizontal'

    _ = axh.set_xticklabels(map(lambda key: labelLookup.get(key,key),colVars),rotation=rotation,size=labelSize)
    _ = axh.set_yticklabels(map(lambda key: labelLookup.get(key,key),rowVars),size=labelSize)

    tmprho = rho.copy()

    if cutoff == 'qvalue':
        criticalValue = qvalue
    elif cutoff == 'pvalue':
        criticalValue = pvalue
    elif cutoff == 'rho':
        criticalValue = np.abs(rho)
    elif cutoff == 'rho2':
        criticalValue = rho**2
        
    tmprho[~(criticalValue <= cutoffValue)] = 0.

    plt.pcolor(tmprho, cmap=_heatCmap, vmin=-1., vmax=1.)
    for i in range(len(rowVars)):
        for j in range(len(colVars)):
            if criticalValue.iloc[i,j] <= cutoffValue and not rowVars[i] == colVars[j]:
                ann = ''
                if annotation == 'pvalue':
                    if pvalue.iloc[i,j] > 0.001:
                        ann = '%1.3f' % pvalue.iloc[i,j]
                    else:
                        ann = '%1.1e' % pvalue.iloc[i,j]
                elif annotation == 'rho':
                    ann = '%1.2f' % rho.iloc[i,j]
                elif annotation == 'rho2':
                    ann = '%1.2f' % (rho.iloc[i,j] ** 2)
                elif annotation == 'qvalue':
                    if qvalue.iloc[i,j]>0.001:
                        ann = '%1.3f' % qvalue.iloc[i,j]
                    else:
                        ann = '%1.1e' % qvalue.iloc[i,j]

                if not ann == '':
                    plt.text(j+0.5, i+0.5, ann, **pvalueTxtProp)

    plt.colorbar(fraction=0.05)
    method = method[0].upper() + method[1:]
    plt.annotate('%s correlation' % method,[0.98,0.5], xycoords='figure fraction', ha='right', va='center', rotation='vertical')
    return rho, pvalue, qvalue

def scatterfit(x, y, method='pearson', adjustVars=[], labelLookup={}, plotLine=True, plotUnity=False, annotateFit=True, annotatePoints=False, returnModel=False, lc='gray', **kwargs):
    """Scatter plot of x vs. y with a fitted line overlaid.

    Expects x and y as pd.Series but will accept arrays.

    Prints covariate unadjusted AND adjusted rho/pvalues on the figure.
    Plots covariate unadjusted data.

    Parameters
    ----------
    x,y : ndarrays or pd.Series
    method : string
        'pearson'
    adjustVars : list
    labelLookup : dict
    plotLine : bool
    annotateFit : bool
    annotatePoints : bool
    returnModel : bool
    kwargs : additional keyword arguments
        Passed to the plot function for the data points.

    Returns
    -------
    model : statsmodels GLM object
        Optionally the fitted model, depending on returnModel."""

    k = kwargs.keys()
    if not 'mec' in k:
        kwargs.update({'mec':'k'})
    if not 'mfc' in k:
        kwargs.update({'mfc':'k'})
    if not 'ms' in k:
        kwargs.update({'ms':5})

    """Try to force X and Y into pandas.Series objects"""
    if not isinstance(x, pd.core.series.Series):
        x = pd.Series(x, name='X')
    if not isinstance(y, pd.core.series.Series):
        y = pd.Series(y, name='Y')

    xlab = x.name
    ylab = y.name
    if xlab == ylab:
        ylab = 'y_'+ylab
        xlab = 'x_'+xlab
        x.name = xlab
        y.name = ylab

    tmpDf = pd.concat((x,y,), axis=1, join='inner')
    for av in adjustVars:
        tmpDf = pd.concat((tmpDf,pd.DataFrame(av)), axis=1)
    
    """Drop any row with a nan in either column"""
    tmpDf = tmpDf.dropna(axis=0, how='any')

    plt.gca().set_xmargin(0.2)
    plt.gca().set_ymargin(0.2)
    
    unrho,unp = partialcorr(tmpDf[xlab],tmpDf[ylab],method=method)

    """Print unadjusted AND adjusted rho/pvalues
    Plot unadjusted data with fit though..."""
    
    if method == 'spearman' and plotLine:
        #unrho,unp=stats.spearmanr(tmpDf[xlab],tmpDf[ylab])
        if unrho > 0:
            plt.plot(sorted(tmpDf[xlab]),sorted(tmpDf[ylab]),'-',color=lc)
        else:
            plt.plot(sorted(tmpDf[xlab]),sorted(tmpDf[ylab],reverse=True),'-',color=lc)
    elif method == 'pearson' and plotLine:
        #unrho,unp=stats.pearsonr(tmpDf[xlab],tmpDf[ylab])
        formula_like = ModelDesc([Term([LookupFactor(ylab)])],[Term([]),Term([LookupFactor(xlab)])])

        Y, X = dmatrices(formula_like, data=tmpDf, return_type='dataframe')
        model = sm.GLM(Y,X,family=sm.families.Gaussian())
        results = model.fit()
        mnmxi = np.array([tmpDf[xlab].idxmin(),tmpDf[xlab].idxmax()])
        plt.plot(tmpDf[xlab][mnmxi],results.fittedvalues[mnmxi],'-',color=lc)

    if plotUnity:
        plt.plot(tmpDf[xlab][mnmxi], tmpDf[xlab][mnmxi], '--', color='white')
    
    plt.plot(tmpDf[xlab],tmpDf[ylab],'o',**kwargs)

    if annotatePoints:
        annotationParams = dict(xytext=(0,5), textcoords='offset points', size='medium')
        for x,y,lab in zip(tmpDf[xlab],tmpDf[ylab],tmpDf.index):
            plt.annotate(lab, xy=(x, y), **annotationParams)

    if annotateFit:
        if unp>0.001:    
            s = 'p = %1.3f\nrho = %1.2f\nn = %d' % (unp, unrho, tmpDf.shape[0])
        else:
            s = 'p = %1.1e\nrho = %1.2f\nn = %d' % (unp, unrho, tmpDf.shape[0])
        textTL(plt.gca(),s,color='black')

        if len(adjustVars) > 0:
            rho,p = partialcorr(tmpDf[xlab], tmpDf[ylab], adjust = adjustVars, method = method)
            if p>0.001:    
                s = 'adj-p = %1.3f\nadj-rho = %1.2f\nn = %d' % (p, rho, tmpDf.shape[0])
            else:
                s = 'adj-p = %1.1e\nadj-rho = %1.2f\nn = %d' % (p, rho, tmpDf.shape[0])

            textTR(plt.gca(),s,color='red')

    plt.xlabel(labelLookup.get(xlab,xlab))
    plt.ylabel(labelLookup.get(ylab,ylab))
    if returnModel:
        return model

def validPairwiseCounts(df, cols=None):
    """Count the number of non-NA data points for
    all pairs of cols in df, as would be needed for
    generating a correlation heatmap.

    Useful for determining a threshold minimum number of
    data pairs for a valid correlation.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list
        Column names to consider

    Returns
    -------
    pwCounts : pd.DataFrame
        DataFrame with columns and index matching cols"""
    if cols is None:
        cols = df.columns
    n = len(cols)
    pwCounts = pd.DataFrame(np.zeros((n,n)), index=cols, columns=cols)
    for colA,colB in itertools.product(cols,cols):
        if colA == colB:
            pwCounts.loc[colA,colA] = df[colA].dropna().shape[0]
        elif colA > colB:
            n = df[[colA,colB]].dropna().shape[0]
            pwCounts.loc[colA,colB] = n
            pwCounts.loc[colB,colA] = n

    return pwCounts

def heatmap(df, colLabels=None, rowLabels=None, labelSize='medium', **kwargs):
    """Heatmap based on values in df

    Parameters
    ----------
    df : pd.DataFrame
        All data in df will be included in heatmap
    colLabels : list
        Strings to replace df column names as x-tick labels
    rowLabels : list
        Strings to replace df index as y-tick labels
    labelSize : fontsize in points or str (e.g. 'large')
    kwargs : dict
        Passed to pcolor()"""
        
    if not 'cmap' in kwargs:
        kwargs['cmap'] = _heatCmap
    
    if colLabels is None:
        colLabels = df.columns
    if rowLabels is None:
        rowLabels = df.index

    plt.clf()
    axh = plt.subplot(111)
    nrows,ncols = df.shape
    plt.pcolor(df.values, **kwargs)
    
    axh.xaxis.tick_top()
    plt.xticks(np.arange(ncols) + 0.5)
    plt.yticks(np.arange(nrows) + 0.5)
    xlabelsL = axh.set_xticklabels(colLabels, size=labelSize, rotation=90, fontname='Consolas')
    ylabelsL = axh.set_yticklabels(rowLabels, size=labelSize, fontname='Consolas')
    plt.ylim((nrows,0))
    plt.xlim((0,ncols))
    plt.colorbar(fraction=0.05)
    plt.tight_layout()
    

def removeNARC(inDf,minRow=1, minCol=1, minFrac=None):
    """Removes all columns and rows that don't have at least
    minX non-NA values. Considers columns then rows iteratively
    until criteria is met or all columns or rows have been removed."""

    def _validCols(df,minCol):
        return [col for col in df.columns if (df.shape[0] - df[col].isnull().sum()) >= minCol]
    def _validRows(df,minRow):
        return [row for row in df.index if (df.shape[1] - df.loc[row].isnull().sum()) >= minRow]

    df = inDf.copy()

    if not minFrac is None:
        minRow = np.round(df.shape[1] * minFrac)
        minCol = np.round(df.shape[0] * minFrac)

    nRows = df.shape[0] + 1
    nCols = df.shape[1] + 1
    
    while (nCols > df.shape[1] or nRows > df.shape[0]) and df.shape[0]>0 and df.shape[1]>0:
        nRows, nCols = df.shape

        df = df[_validCols(df,minCol)]
        df = df.loc[_validRows(df,minRow)]

    return df

def permcorr(a,b,corrFunc, nperms = 10000):
    """Use shuffled permutations of a and b (np.ndarrays or pd.Series)
    to estimate the correlation p-value and rho with CIs (TODO)

    Parameters
    ----------
    a,b : np.ndarray or pd.Series
    corrFunc : function
        Parameters are a and b with return value rho, p-value

    Returns
    -------
    rho : float
    p : float"""

    if isinstance(a,pd.Series):
        a = a.values
    if isinstance(b,pd.Series):
        b = b.values

    rhoShuff = np.zeros(nperms)
    pShuff = np.zeros(nperms)

    rho,pvalue = corrFunc(a,b)

    L = a.shape[0]
    for permi in np.arange(nperms):
        rind = np.floor(np.random.rand(L) * L).astype(int)
        rhoShuff[permi],pShuff[permi] = corrFunc(a,b[rind])

    if rho >= 0:
        p = ((rhoShuff >= rho).sum() + 1)/(nperms + 1)
    else:
        p = ((rhoShuff <= rho).sum() + 1)/(nperms + 1)
    return rho, p

def labeledScatter(x, y, labels, **kwargs):
    """Matplotlib scatter plot with added annotations for each point.

    Parameters
    ----------
    x, y : Passed on to plt.scatter
    labels : list
        Strings to annotate each point in x/y
    kwargs : Passed on to plt.scatter

    Returns
    -------
    scatterH : handles from plt.scatter
    annotateH : handles from plt.annotate"""
    
    axh = plt.scatter(x, y, **kwargs)
    labh = []
    for xx, yy, lab in zip(x,y, labels):        
        h = plt.annotate(lab, xy=(xx,yy),
                             xytext=(3,3),
                             ha='left',
                             va='bottom',
                             textcoords='offset points',
                             size='small')
        labh.append(h)
    return axh, labh
