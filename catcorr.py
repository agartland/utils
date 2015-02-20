from __future__ import division
import networkx as nx
import itertools
import brewer2mpl
import plotly.plotly as py
import plotly.graph_objs as pygo
from pylab import *
import pandas as pd
import statsmodels.api as sm

from myfisher import *
from objhist import *
from custom_legends import *

__all__ = ['catcorr',
           'layouts',
           'generateTestData',
           'testEdge',
           'cull_rows']

layouts = ['twopi', 'fdp','circo', 'neato', 'dot']

color2str = lambda col: 'rgb'+str(tuple((array(col)*256).round().astype(int)))

def catcorr(df, layout='fdp', mode='mpl', titleStr='', testSig=0.2, sRange=(50,inf), wRange=(0.1,inf)):
    """Make a network plot showing the correlations among the
    categorical variables in the columns of df.

    Each node is a unique value in one of the columns (Node is specified as a tuple (column, value))
    Node size is proportional to the value's frequency.

    Each edge is a unique pair of values in two columns.
    Edge width is proportional to the frequency of the pairing.

    Parameters
    ----------
    df : pandas.DataFrame
        Nodes will be created for each unique value within each column of this object
    layout : str
        Choose one of ['twopi', 'fdp','circo', 'neato', 'dot']
        to change the layout of the nodes. See Graphviz for details about each layout.
    mode : str
        Specifies whether the resulting plot will be a matplotlib figure (default: 'mpl')
        or if any other value it specifies the filename of a figure to be posted to plot.ly
        (user needs to be logged in previously).
    titleStr : str
        Printed at the top of the plot.
    testSig : float
        If non-zero then testSig is used as the significance cutoff for plotting a highlighted edge.
        For each edge, tests the statistical hypothesis that number of observed pairings
        between values in two columns is significantly different than what one would expect
        based on their marginal frequencies. Note: there is no adjustment for multiple comparisons.
    sRange,wRange : tuples of length 2
        Contains the min and max node sizes or edge widths in points, for scaling

    Examples
    --------
    >>> import plotly.plotly as py
    
    >>> py.sign_in([username], [api_key])

    >>> df = generateTestData()

    >>> catcorr(df, layout = 'neato', mode = 'catcorr_example')

    [Posts a catcorr plot to plot.ly]

    """

    """Compute odds-ratios, p-values and FDR-adjusted q-values for each edge"""
    edgeKeys = []
    pvalueArr = []
    tested = []
    for col1,col2 in itertools.combinations(df.columns,2):
        for val1,val2 in itertools.product(df[col1].unique(),df[col2].unique()):
            w = ((df[col1]==val1) & (df[col2]==val2)).sum()
            pvalue = 1
            if w>0 and testSig:
                OR,pvalue = testEdge(df,(col1,val1),(col2,val2))
                tested.append(True)
            else:
                tested.append(False)
            edgeKeys.append(((col1,val1),(col2,val2)))
            pvalueArr.append(pvalue)
    pvalueArr,tested = array(pvalueArr), array(tested)
    qvalueArr = ones(pvalueArr.shape)
    qvalueArr[tested] = sm.stats.multipletests(pvalueArr[tested],method='fdr_bh')[1]

    g = nx.Graph()
    """Add a node for each unique value in each column with name: col_value"""
    for col in df.columns:
        for val in df[col].unique():
            freq = (df[col]==val).sum()/df.shape[0]
            g.add_node((col,val),freq=freq)
    """Add edges for each unique pair of values
    with edgewidth proportional to frequency of pairing"""
    for col1,col2 in itertools.combinations(df.columns,2):
        for val1,val2 in itertools.product(df[col1].unique(),df[col2].unique()):
            w = ((df[col1]==val1) & (df[col2]==val2)).sum()
            if w>0:
                dat = dict(weight = w/df.shape[0])
                dat['pvalue'] = pvalueArr[edgeKeys.index(((col1,val1),(col2,val2)))]
                dat['qvalue'] = qvalueArr[edgeKeys.index(((col1,val1),(col2,val2)))]
                g.add_edge((col1,val1),(col2,val2),**dat)


    """Compute attributes of edges and nodes"""
    edgewidth = array([d['weight'] for n1,n2,d in g.edges(data=True)])
    nodesize = array([d['freq'] for n,d in g.nodes(data=True)])

    nColors = min(max(len(df.columns),3),9)
    colors = brewer2mpl.get_map('Set1','Qualitative',nColors).mpl_colors
    cmap = {c:color for c,color in zip(df.columns, itertools.cycle(colors))}
    nodecolors = [cmap[n[0]] for n in g.nodes()]
    if layout == 'twopi':
        """If using this layout specify the most common node as the root"""
        freq = {n:d['freq'] for n,d in g.nodes(data=True)}
        pos = nx.graphviz_layout(g,prog=layout, root=max(freq.keys(),key=freq.get))
    else:
        pos = nx.graphviz_layout(g,prog=layout)

    """Use either matplotlib or plot.ly to plot the network"""
    if mode == 'mpl':
        clf()
        figh=gcf()
        axh=figh.add_axes([0.04,0.04,0.92,0.92])
        axh.axis('off')
        figh.set_facecolor('white')

        #nx.draw_networkx_edges(g,pos,alpha=0.5,width=sznorm(edgewidth,mn=0.5,mx=10), edge_color='k')
        #nx.draw_networkx_nodes(g,pos,node_size=sznorm(nodesize,mn=500,mx=5000),node_color=nodecolors,alpha=1)
        ew = szscale(edgewidth,mn=wRange[0],mx=wRange[1])

        for es,e in zip(ew,g.edges_iter()):
            x1,y1=pos[e[0]]
            x2,y2=pos[e[1]]
            props = dict(color='black',alpha=0.4,zorder=1)
            if testSig and g[e[0]][e[1]]['qvalue'] < testSig:
                props['color']='orange'
                props['alpha']=0.8
            plot([x1,x2],[y1,y2],'-',lw=es,**props)

        scatter(x=[pos[s][0] for s in g.nodes()],
                y=[pos[s][1] for s in g.nodes()],
                s=szscale(nodesize,mn=sRange[0],mx=sRange[1]), #Units for scatter is (size in points)**2
                c=nodecolors,
                alpha=1,zorder=2)
        for n in g.nodes():
            annotate(n[1],
                    xy=pos[n],
                    fontname='Consolas',
                    size='medium',
                    weight='bold',
                    color='black',
                    va='center',
                    ha='center')
        colorLegend(labels=df.columns,colors = [c for x,c in zip(df.columns,colors)],loc=0)
        title(titleStr)
    else:
        """Send the plot to plot.ly"""
        data = []
        for es,e in zip(szscale(edgewidth,mn=wRange[0],mx=wRange[1]),g.edges_iter()):
            x1,y1=pos[e[0]]
            x2,y2=pos[e[1]]
            props = dict(color='black',opacity=0.4)
            if testSig and g[e[0]][e[1]]['qvalue'] < testSig:
                props['color']='orange'
                props['opacity']=0.8
            tmp = pygo.Scatter(x=[x1,x2],
                          y=[y1,y2],
                          mode='lines',
                          line=pygo.Line(width=es,**props),
                          showlegend=False)
            data.append(tmp)
        """May need to add sqrt() to match mpl plots"""
        nodesize = szscale(nodesize,mn=sRange[0],mx=sRange[1]) #Units for plotly.Scatter is (size in points)
        for col in cmap.keys():
            ind = [nodei for nodei,node in enumerate(g.nodes()) if node[0]==col]
            tmp = pygo.Scatter(x=[pos[s][0] for nodei,s in enumerate(g.nodes()) if nodei in ind],
                    y=[pos[s][1] for nodei,s in enumerate(g.nodes()) if nodei in ind],
                    mode='markers',
                    name=col,
                    text=[node[1] for nodei,node in enumerate(g.nodes()) if nodei in ind],
                    textposition='middle center',
                    marker=pygo.Marker(size=nodesize[ind],
                                  color=[color2str(nc) for nodei,nc in enumerate(nodecolors) if nodei in ind]))
            data.append(tmp)
        layout = pygo.Layout(title=titleStr,
                        showlegend=True,
                        xaxis=pygo.XAxis(showgrid=False, zeroline=False),
                        yaxis=pygo.YAxis(showgrid=False, zeroline=False))

        fig = pygo.Figure(data=data, layout=layout)
        plot_url = py.plot(fig, filename='catcorr_'+mode)

def generateTestData(nrows=100):
    """Generate a pd.DataFrame() with correlations that can be visualized by catcorr()"""
    testDf = pd.DataFrame(zeros((nrows,3),dtype=object),columns = ['ColA','ColB','ColC'])
    """Use objhist to generate specific frequencies of (0,0,0), (1,0,0) etc. with values 1-4"""
    oh = objhist([])
    oh.update({('X','A','foo'):2,
              ('X','A','bar'):5,
              ('X','B','foo'):1,
              ('X','B','bar'):10,
              ('Y','A','bar'):10,
              ('Y','B','bar'):7})
    for i,v in enumerate(oh.generateRandomSequence(nrows)):
        testDf['ColA'].loc[i] = v[0]
        testDf['ColB'].loc[i] = v[1]
        testDf['ColC'].loc[i] = v[2]
    return testDf

def sznorm(vec,mx=1,mn=0):
    """Normalize values of vec to [mn, mx] interval"""
    vec-=nanmin(vec)
    vec=vec/nanmax(vec)
    vec=vec*(mx-mn)+mn
    vec[isnan(vec)] = mn
    vec[vec<mn] = mn
    return vec

def szscale(vec,mx=inf,mn=1):
    """Normalize values of vec to [mn, mx] interval
    such that sz ratios remain representative."""
    factor = mn/nanmin(vec)
    vec = vec*factor
    vec[vec>mx] = mx
    vec[isnan(vec)] = mn
    return vec    

def cull_rows(df,cols,freq):
    """Remove all rows from df that contain any column
    with a value that is less frequent than freq.

    Parameters
    ----------
    df : pandas.DataFrame
    cols : list
        List of column indices in df
    freq : float
        Frequency threshold for row removal.

    Returns
    -------
    outDf : pandas.DataFrame
        A copy of df with rows removed."""
        
    outDf = df.copy()
    keepers = {}

    for c in cols:
        oh = objhist(df[c]).freq()
        keepers[c] = [v for v in oh.keys() if oh[v]>freq]
        
    """Keep rows that have a value in keepers for each column"""
    for c in cols:
        outDf = outDf.loc[outDf[c].map(lambda v: v in keepers[c])]
    return outDf

def testEdge(df,node1,node2,verbose=False):
    """Test if the occurence of nodeA paired with nodeB is more/less common than expected.

    Parameters
    ----------
    nodeX : tuple (column, value)
        Specify the node by its column name and the value.

    Returns
    -------
    OR : float
        Odds-ratio associated with the 2x2 contingency table
    pvalue : float
        P-value associated with the Fisher's exact test that H0: OR = 1"""
    col1,val1 = node1
    col2,val2 = node2
    
    tab = zeros((2,2))
    tab[0,0] = ((df[col1]!=val1) & (df[col2]!=val2)).sum()
    tab[0,1] = ((df[col1]!=val1) & (df[col2]==val2)).sum()
    tab[1,0] = ((df[col1]==val1) & (df[col2]!=val2)).sum()
    tab[1,1] = ((df[col1]==val1) & (df[col2]==val2)).sum()

    """Add 1 to cells with zero"""
    if any(tab==0):
        if verbose:
            print 'Adding one to %d cells with zero counts.' % (ind.sum())
            print
    tab[tab==0] = 1

    OR,pvalue = fisherTest(tab)

    if verbose:
        print 'Node1: %s, %s' % node1
        print 'Node2: %s, %s' % node2
        print
        print pd.DataFrame(tab,index=['Node1(-)','Node1(+)'],columns = ['Node2(-)','Node2(+)'])
        print '\nOR: %1.2f\nP-value: %1.3f' % (OR,pvalue)
    return OR,pvalue