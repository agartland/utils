from __future__ import division
import networkx as nx
import itertools
import brewer2mpl
import plotly.plotly as py
from plotly.graph_objs import *
from pylab import *
import pandas as pd

from myfisher import *
from objhist import *
from custom_legends import *

__all__ = ['catcorr',
           'layouts',
           'generateTestData']

layouts = ['twopi', 'fdp','circo', 'neato', 'dot', 'sfdp']

color2str = lambda col: 'rgb'+str(tuple((array(col)*256).round().astype(int)))

def catcorr(df, layout='fdp', mode='mpl', titleStr='', testSig=False, sRange=(15,70), wRange=(0.5,15)):
    """Make a network plot showing the correlations among the
    categorical variables in the columns of df.

    Each node is a unique value in one of the columns.
    Node size is proportional to the value's frequency.

    Each edge is a unique pair of values in two columns.
    Edge width is proportional to the frequency of the pairing.

    Parameters
    ----------
    df : pandas.DataFrame
        Nodes will be created for each unique value within each column of this object
    layout : str
        Choose one of ['twopi', 'fdp','circo', 'neato', 'dot', 'sfdp']
        to change the layout of the nodes. See Graphviz for details about each layout.
    mode : str
        Specifies whether the resulting plot will be a matplotlib figure (default: 'mpl')
        or if any other value it specifies the filename of a figure to be posted to plot.ly
        (user needs to be logged in previously).
    titleStr : str
        Printed at the top of the plot.
    testSig : bool
        For each edge, tests the statistical hypothesis that number of observed pairings
        between values in two columns is significantly different than what one would expect
        based on their marginal frequencies.
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

    SPLITCHAR = '@'
    g = nx.Graph()
    """Add a node for each unique value in each column with name: col_value"""
    for col in df.columns:
        for val in df[col].unique():
            freq = (df[col]==val).sum()/df.shape[0]
            g.add_node('%s%s%s' % (col,SPLITCHAR,val),freq=freq)
    """Add edges for each unique pair of values
    with edgewidth proportional to frequency of pairing"""
    for col1,col2 in itertools.combinations(df.columns,2):
        for val1,val2 in itertools.product(df[col1].unique(),df[col2].unique()):
            w = ((df[col1]==val1) & (df[col2]==val2)).sum()
            if w>0:
                print w
                dat = dict(weight = w/df.shape[0])
                if testSig:
                    tab = [[((df[col1]!=val1) & (df[col2]!=val2)).sum(), ((df[col1]==val1) & (df[col2]!=val2)).sum()],
                           [((df[col1]!=val1) & (df[col2]==val2)).sum(),((df[col1]==val1) & (df[col2]==val2)).sum()]]
                    OR,pvalue = fisherTest(tab)
                    dat['pvalue'] = pvalue
                g.add_edge('%s%s%s' % (col1,SPLITCHAR,val1),'%s%s%s' % (col2,SPLITCHAR,val2),**dat)

    """Compute attributes of edges and nodes"""
    edgewidth = array([d['weight'] for n1,n2,d in g.edges(data=True)])
    nodesize = array([d['freq'] for n,d in g.nodes(data=True)])

    nColors = min(max(len(df.columns),3),9)
    colors = brewer2mpl.get_map('Set1','Qualitative',nColors).mpl_colors
    cmap = {c:color for c,color in zip(df.columns, itertools.cycle(colors))}
    nodecolors = [cmap[n.split(SPLITCHAR)[0]] for n in g.nodes()]
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
        ew = sznorm(edgewidth,mn=wRange[0],mx=wRange[1])

        for es,e in zip(ew,g.edges_iter()):
            x1,y1=pos[e[0]]
            x2,y2=pos[e[1]]
            props = dict(color='black',alpha=0.4,zorder=1)
            if testSig and g[e[0]][e[1]]['pvalue'] < testSig:
                props['color']='orange'
                props['alpha']=0.8
            plot([x1,x2],[y1,y2],'-',lw=es,**props)

        scatter(x=[pos[s][0] for s in g.nodes()],
                y=[pos[s][1] for s in g.nodes()],
                s=sznorm(nodesize,mn=sRange[0],mx=sRange[1])**2, #Units for scatter is (size in points)**2
                c=nodecolors,
                alpha=1,zorder=2)
        for n in g.nodes():
            annotate(n.split(SPLITCHAR)[1],
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
        data = []
        for es,e in zip(sznorm(edgewidth,mn=wRange[0],mx=wRange[1]),g.edges_iter()):
            x1,y1=pos[e[0]]
            x2,y2=pos[e[1]]
            props = dict(color='black',opacity=0.4)
            if testSig and g[e[0]][e[1]]['pvalue'] < testSig:
                props['color']='orange'
                props['opacity']=0.8
            tmp = Scatter(x=[x1,x2],
                          y=[y1,y2],
                          mode='lines',
                          line=Line(width=es,**props),
                          showlegend=False)
            data.append(tmp)
        nodesize = sznorm(nodesize,mn=sRange[0],mx=sRange[1])
        for col in cmap.keys():
            ind = [nodei for nodei,node in enumerate(g.nodes()) if node.split(SPLITCHAR)[0]==col]
            tmp = Scatter(x=[pos[s][0] for nodei,s in enumerate(g.nodes()) if nodei in ind],
                    y=[pos[s][1] for nodei,s in enumerate(g.nodes()) if nodei in ind],
                    mode='markers',
                    name=col,
                    text=[node.split(SPLITCHAR)[1] for nodei,node in enumerate(g.nodes()) if nodei in ind],
                    textposition='middle center',
                    marker=Marker(size=nodesize[ind],
                                  color=[color2str(nc) for nodei,nc in enumerate(nodecolors) if nodei in ind]))
            data.append(tmp)
        layout = Layout(title=titleStr,
                        showlegend=True,
                        xaxis=XAxis(showgrid=False, zeroline=False),
                        yaxis=YAxis(showgrid=False, zeroline=False))

        fig = Figure(data=data, layout=layout)
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
