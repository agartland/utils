from numpy import *
import matplotlib
from pylab import *
from matplotlib.gridspec import GridSpec
from scipy import polyfit, polyval, stats
import pandas as pd
from mytext import *
import statsmodels.api as sm
from patsy import dmatrices,ModelDesc,Term,LookupFactor
from copy import deepcopy
import itertools
import warnings

__all__ = ['partialcorr',
           'combocorrplot',
           'scatterfit',
           'heatmap',
           'corrheatmap',
           'validPairwiseCounts',
           'removeNARC']

"""Red --> Green colormap with 1024 interpolated values"""
_cdict = {'green' : ((0, 1, 1), (0.5, 0, 0), (1, 0, 0)),
          'red':    ((0, 0, 0), (0.5, 0, 0), (1, 1, 1)),
          'blue' :  ((0, 0, 0), (1, 0, 0))}
_heatCmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', _cdict, 1024)

def partialcorr(x, y, adjust=[], method='pearson'):
    """Finds partial correlation of x with y adjusting for variables in adjust

    This function is index aware (i.e. uses index of x, y and adjust for joining).
    Rho and p-value match those from stats.spearmanr, and stats.pearsonr when adjust = [].

    Parameters
    ----------
    x,y : pd.Series
        Each contains data for assessing correlation.
    adjust : list of pd.Series objects
        Correlation is assessed between x and y adjusting for all variables in z (default: [])
    method : string
        Method can be 'pearson' (default) or 'spearman', which uses rank-based correlation and adjustment.
    
    Returns
    -------
    partial_rho : float
        Partial correlation coefficient between x and y after adjustment.
    pvalue : float
        P-value for the partial correlation coefficient."""

    """Make one big DataFrame out of x, y and adjustment variables"""
    tmpDf = pd.concat((x,y), join='inner', axis=1)
    for a in adjust:
        tmpDf = tmpDf.join(a, how='left')

    tmpDf = tmpDf.dropna(axis=0, how='any')
    
    m = zeros((tmpDf.shape[0], 2+len(adjust)))
    
    if method=='spearman':
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

    if all(m[:,0]==m[:,1]):
        """Testing for perfect correlation avoids SingularMatrix exception"""
        return 1,0.0
    
    '''Take the inverse of the covariance matrix including all variables
    pc = -p_ij / sqrt(p_ii * p_ij)
    where p is the inverse covariance matrix'''
    
    try:
        icv = inv(cov(m,rowvar=0))
        pc = -icv[0,1]/sqrt(icv[0,0]*icv[1,1])

        n = m.shape[0]
        gn = len(adjust)
        
        statistic = pc*sqrt((n-2-gn)/(1-pc**2))
        #pvalue = 2*stats.norm.cdf(-abs(statistic))

        #SAS and pearsonr look the statistic up in a t distribution while R uses the normnal

        pvalue = 2*stats.t.cdf(-abs(statistic),n-2-gn)
    except:
        """These were used to check that non-partial rho's and pvalues match those of their scipy equivalents
        They do! Use them if the other fails and warn the caller"""
        if method=='pearson':
            pc,pvalue = stats.pearsonr(x,y)    
        else:
            pc,pvalue = stats.spearmanr(x,y)
        warnings.warn("Error computing %s and %s correlation: using scipy equivalent to return UNADJUSTED results'   % (x.name,y.name)")
    
    """Below verifies that the p-value for the coefficient in the multivariate model including adjust
    is the same as the p-value of the partial correlation"""
    
    """formula_like=ModelDesc([Term([LookupFactor(y.name)])],[Term([]),Term([LookupFactor(x.name)])]+[Term([LookupFactor(a.name)]) for a in adjust])
    Y, X = dmatrices(formula_like, data=tmpDf, return_type='dataframe')
    model=sm.GLM(Y,X,family=sm.families.Gaussian())
    print model.fit().summary()"""

    return pc,pvalue


def combocorrplot(data,method='spearman',axLimits='variable',axTicks=False,axTicklabels=False,valueFlag=True,ms=2):
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
        Scatter plot marker size in points."""

    border = 0.05
    pad = 0.02
    cbwidth = 0.1

    labels = data.columns

    """Use pd.DataFrame method to compute the pairwise correlations"""
    coef = data.corr(method=method)
    n = coef.shape[0]


    axh = empty((n,n),dtype=object)
    plth = empty((n,n),dtype=object)

    mx = None
    mn = None
    for col in data.columns:
        if mx==None:
            mx = data[col].max()
            mn = data[col].min()
        mx = max(data[col].max(),mx)
        mn = min(data[col].min(),mn)

    clf()
    fh = gcf()
    gs = GridSpec(n,n,left=border,bottom=border,right=1.-border-cbwidth,top=1.-border,wspace=pad,hspace=pad)
    #cbgs=GridSpec(1,1,left=1.-cbwidth,bottom=border,right=1.-border,top=1.-border,wspace=pad,hspace=pad)
    for r in xrange(n):
        for c in xrange(n):
            if r==c:
                axh[r,c] = fh.add_subplot(gs[r,c],yticklabels=[],xticklabels=[],xticks=[],yticks=[])
                text(0,0,'%s' % (data.columns[r]),ha='center',va='center')
                axis([-1,1,-1,1])
            elif r>c:
                if axTicks:
                    if axTicklabels:
                        if r<len(labels)-1 and c>0:
                            axh[r,c] = fh.add_subplot(gs[r,c],xticklabels=[],yticklabels=[])
                        elif r<len(labels)-1 and c==0:
                            axh[r,c] = fh.add_subplot(gs[r,c],xticklabels=[])
                        elif r==len(labels)-1 and c>0:
                            axh[r,c] = fh.add_subplot(gs[r,c],yticklabels=[])
                        elif r==len(labels)-1 and c==0:
                            axh[r,c] = fh.add_subplot(gs[r,c])
                    else:
                        axh[r,c] = fh.add_subplot(gs[r,c],xticklabels=[],yticklabels=[])

                else:
                    axh[r,c] = fh.add_subplot(gs[r,c],xticks=[],yticks=[])

                if method=='pearson':
                    ar,br = polyfit(data[labels[r]],data[labels[c]],1)
                    xfit = array([min(data[labels[r]]),max(data[labels[r]])])
                    yfit = polyval([ar,br],xfit)
                    plot(xfit,yfit,'-',lw=1,color='gray')
                    hold(True)
                plot(data[labels[r]],data[labels[c]],'ok',ms=ms)
                
                if axLimits=='variable':
                    rmax,rmin = max(data[labels[r]]),min(data[labels[r]])
                    cmax,cmin = max(data[labels[c]]),min(data[labels[c]])
                else:
                    rmax,cmax = mx,mx
                    rmin,cmin = mn,mn

                axis([rmin-0.1*(rmax-rmin), rmax+0.1*(rmax-rmin),cmin-0.1*(cmax-cmin), cmax+0.1*(cmax-cmin)])
            elif r<c:
                axh[r,c] = fh.add_subplot(gs[r,c],yticklabels=[],xticklabels=[],xticks=[],yticks=[])
                val = coef[labels[r]][labels[c]]
                plth[r,c] = pcolor(ones((2,2))*val,cmap=_heatCmap,vmin=-1.,vmax=1.)
                axis([0,1,0,1])
                if valueFlag:
                    if val<0.5 and val>-0.5:
                        txtcol = 'white'
                    else:
                        txtcol = 'black'
                    text(0.5,0.5,'%1.2f' % (val),ha='center',va='center',family='monospace',color=txtcol)
    cbax = fh.add_axes([1.-cbwidth-border/2,border,cbwidth-border-0.02,1.-2*border])
    cb = colorbar(plth[0,0],cax=cbax)
    method = method[0].upper() + method[1:]
    annotate('%s correlation' % (method),[0.98,0.5],xycoords='figure fraction',ha='right',va='center',rotation='vertical')

def corrheatmap(df,rowVars,colVars,adjust=[],annotation='pvalue',cutoff='pvalue',cutoffValue=0.05,method='spearman',labelLookup={},xtickRotate=False,labelSize='medium',minN=None):
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
    pvalue : ndarray [samples, variables]
        Matrix of pvalues for pairwise correlations.
    qvalue : ndarray [samples, variables]
        Matrix of FDR-adjusted q-values for pairwise correlations.
    rho : ndarray [samples, variables]
        Matrix of correlation coefficients."""
    
    pvalue = zeros((len(rowVars),len(colVars)))
    qvalue = nan*zeros((len(rowVars),len(colVars)))
    rho = zeros((len(rowVars),len(colVars)))

    """Store p-values in dict with keys that are unique pairs (so we only adjust across these)"""
    pairedPvalues = {}
    paireQPvalues = {}
    allColumns = df.columns.tolist()

    for i,rowv in enumerate(rowVars):
        for j,colv in enumerate(colVars):
            if not rowv==colv:
                if not df[[rowv,colv]].dropna().shape[0]<minN:
                    rho[i,j],pvalue[i,j] = partialcorr(df[rowv],df[colv],adjust=[df[a] for a in adjust],method=method)
                else:
                    """Pvalue = nan excludes these from the multiplicity adjustment"""
                    rho[i,j],pvalue[i,j] = 1,nan
                """Define unique key for the pair by sorting in order they appear in df columns"""
                key = tuple(sorted([rowv,colv],key=allColumns.index))
                pairedPvalues.update({key:pvalue[i,j]})
            else:
                """By setting these pvalues to nan we exclude them from multiplicity adjustment"""
                rho[i,j],pvalue[i,j] = 1,nan
            
    """Now only adjust using pvalues in the unique pair dict"""
    keys = pairedPvalues.keys()
    qvalueTmp = _fdrAdjust(array([pairedPvalues[k] for k in keys]))
    """Build a unique qvalue dict from teh same unique keys"""
    pairedQvalues = {k:q for k,q in zip(keys,qvalueTmp)}
    
    """Assign the unique qvalues to the correct comparisons"""
    for i,rowv in enumerate(rowVars):
        for j,colv in enumerate(colVars):
            if not rowv == colv:
                key = tuple(sorted([rowv,colv], key=allColumns.index))
                qvalue[i,j] = pairedQvalues[key]
            else:
                pvalue[i,j] = 0.
                qvalue[i,j] = 0.

    clf()
    fh = gcf()
    
    pvalueTxtProp = dict(family='monospace',size='large',weight='bold',color='white',ha='center',va='center')

    axh = fh.add_subplot(111, yticks = arange(len(rowVars))+0.5,
                              xticks = arange(len(colVars))+0.5)
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
        criticalValue = abs(rho)
    elif cutoff == 'rho2':
        criticalValue = rho**2
        
    tmprho[~(criticalValue <= cutoffValue)] = 0.

    pcolor(tmprho, cmap=_heatCmap,vmin=-1.,vmax=1.)
    for i in xrange(len(rowVars)):
        for j in xrange(len(colVars)):
            if criticalValue[i,j]<=cutoffValue and not rowVars[i]==colVars[j]:
                ann = ''
                if annotation == 'pvalue':
                    if pvalue[i,j]>0.001:
                        ann = '%1.3f' % pvalue[i,j]
                    else:
                        ann = '%1.1e' % pvalue[i,j]
                elif annotation == 'rho':
                    ann = '%1.2f' % rho[i,j]
                elif annotation == 'rho2':
                    ann = '%1.2f' % (rho[i,j]*rho[i,j])
                elif annotation == 'qvalue':
                    if qvalue[i,j]>0.001:
                        ann = '%1.3f' % qvalue[i,j]
                    else:
                        ann = '%1.1e' % qvalue[i,j]

                if not ann == '':
                    text(j+0.5,i+0.5,ann,**pvalueTxtProp)

    colorbar(fraction=0.05)
    method = method[0].upper() + method[1:]
    annotate('%s correlation' % method,[0.98,0.5],xycoords='figure fraction',ha='right',va='center',rotation='vertical')

    return pvalue, qvalue, rho


def scatterfit(x,y,method='pearson',adjustVars=[],labelLookup={},plotLine=True,annotateFit=True,returnModel=False,lc = 'gray', **kwargs):
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

    tmpDf = pd.concat((x,y,),axis=1,join='inner')
    for av in adjustVars:
        tmpDf = pd.concat((tmpDf,pd.DataFrame(av)),axis=1)
    
    """Drop any row with a nan in either column"""
    tmpDf = tmpDf.dropna(axis = 0, how = 'any')

    gca().set_xmargin(0.2)
    gca().set_ymargin(0.2)
    
    unrho,unp = partialcorr(tmpDf[xlab],tmpDf[ylab],method=method)

    """Print unadjusted AND adjusted rho/pvalues
    Plot unadjusted data with fit though..."""
    
    if method == 'spearman' and plotLine:
        #unrho,unp=stats.spearmanr(tmpDf[xlab],tmpDf[ylab])
        if unrho>0:
            plot(sorted(tmpDf[xlab]),sorted(tmpDf[ylab]),'-',color=lc)
        else:
            plot(sorted(tmpDf[xlab]),sorted(tmpDf[ylab],reverse=True),'-',color=lc)
    elif method == 'pearson' and plotLine:
        #unrho,unp=stats.pearsonr(tmpDf[xlab],tmpDf[ylab])
        formula_like = ModelDesc([Term([LookupFactor(ylab)])],[Term([]),Term([LookupFactor(xlab)])])

        Y, X = dmatrices(formula_like, data=tmpDf, return_type='dataframe')
        model = sm.GLM(Y,X,family=sm.families.Gaussian())
        results = model.fit()
        mnmxi = array([tmpDf[xlab].idxmin(),tmpDf[xlab].idxmax()])
        plot(tmpDf[xlab][mnmxi],results.fittedvalues[mnmxi],'-',color=lc)
    
    plot(tmpDf[xlab],tmpDf[ylab],'o',**kwargs)

    if annotateFit:
        if unp>0.001:    
            s = 'p = %1.3f\nrho = %1.2f\nn = %d' % (unp, unrho, tmpDf.shape[0])
        else:
            s = 'p = %1.1e\nrho = %1.2f\nn = %d' % (unp, unrho, tmpDf.shape[0])
        textTL(gca(),s,color='black')

        if len(adjustVars) > 0:
            rho,p = partialcorr(tmpDf[xlab], tmpDf[ylab], adjust = adjustVars, method = method)
            if p>0.001:    
                s = 'adj-p = %1.3f\nadj-rho = %1.2f\nn = %d' % (p, rho, tmpDf.shape[0])
            else:
                s = 'adj-p = %1.1e\nadj-rho = %1.2f\nn = %d' % (p, rho, tmpDf.shape[0])

            textTR(gca(),s,color='red')

    xlabel(labelLookup.get(xlab,xlab))
    ylabel(labelLookup.get(ylab,ylab))
    if returnModel:
        return model

def _fdrAdjust(pvalues):
    """Convenient function for doing FDR adjustment
    Accepts any matrix shape and adjusts across the entire matrix
    Ignores nans appropriately

    1) Pvalues can be DataFrame or Series or array
    2) Turn it into a one-dimensional vector
    3) Qvalues intialized at p to copy nans in the right places
    4) Drop the nans, calculate qvalues, copy to qvalues vector
    5) Reshape qvalues
    6) Return same type as pvalues
    """
    p = array(pvalues).flatten()
    qvalues = deepcopy(p)
    nanInd = isnan(p)
    dummy,q,dummy,dummy = sm.stats.multipletests(p[~nanInd],alpha=0.2,method='fdr_bh')
    qvalues[~nanInd] = q
    qvalues = qvalues.reshape(pvalues.shape)

    if type(pvalues) is pd.core.frame.DataFrame:
        return pd.DataFrame(qvalues,columns=[x+'_q' for x in pvalues.columns],index=pvalues.index)
    elif type(pvalues) is pd.core.series.Series:
        return pd.Series(qvalues,name=pvalues.name+'_q',index=pvalues.index)
    else:
        return qvalues

def validPairwiseCounts(df,cols):
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

    n = len(cols)
    pwCounts = pd.DataFrame(zeros((n,n)), index=cols,columns=cols)
    for colA,colB in itertools.product(cols,cols):
        if colA == colB:
            pwCounts.loc[colA,colA] = df[colA].dropna().shape[0]
        elif colA > colB:
            n = df[[colA,colB]].dropna().shape[0]
            pwCounts.loc[colA,colB] = n
            pwCounts.loc[colB,colA] = n

    return pwCounts

def heatmap(df,colLabels=None,rowLabels=None,labelSize='medium',**kwargs):
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

    clf()
    axh = subplot(111)
    nrows,ncols = df.shape
    pcolor(df.values,**kwargs)
    
    axh.xaxis.tick_top()
    xticks(arange(ncols)+0.5)
    yticks(arange(nrows)+0.5)
    xlabelsL = axh.set_xticklabels(colLabels,size=labelSize,rotation=90,fontname='Consolas')
    ylabelsL = axh.set_yticklabels(rowLabels,size=labelSize,fontname='Consolas')
    ylim((nrows,0))
    xlim((0,ncols))
    colorbar(fraction = 0.05)
    

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
        minRow = round(df.shape[1] * minFrac)
        minCol = round(df.shape[0] * minFrac)

    nRows = df.shape[0] + 1
    nCols = df.shape[1] + 1
    
    while (nCols > df.shape[1] or nRows > df.shape[0]) and df.shape[0]>0 and df.shape[1]>0:
        nRows, nCols = df.shape

        df = df[_validCols(df,minCol)]
        df = df.loc[_validRows(df,minRow)]

    return df