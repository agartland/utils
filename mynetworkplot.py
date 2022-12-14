import matplotlib as mpl
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import itertools
import networkx as nx
import altair as alt

def _make_edge_dataframe(G, pos):
    edge_list = []
    for i, e in enumerate(G.edges):
        tmp = dict(edge=i,
                    source=e[0],
                    target=e[1],
                    pair=e,
                    x=pos[e[0]][0],
                    y=pos[e[0]][1])
        edge_list.append(tmp)
        tmp = dict(edge=i,
                    source=e[0],
                    target=e[1],
                    pair=e,
                    x=pos[e[1]][0],
                    y=pos[e[1]][1])
        edge_list.append(tmp)
    return pd.DataFrame(edge_list)

def plot_network(adj_mat, data=None,
                 layout='neato', # passed to nx.nx_agraph.graphviz_layout(G, prog=layout)
                 remove_isolates=True,
                 node_hue=None, # a categorical variable (TODO: accept continuous variable)
                 node_size=40, # can be a column name or float or vector
                 node_symbol=None, #only implented in altair wth categorical var
                 edge_linewidth=1,
                 edge_color='gray',
                 node_linewidth=0.5,
                 node_alpha=1,
                 hue_order=None, # only applies to mpl backend
                 palette=None, # for mpl its a colormap or list of colors; for altair its the name of a Vega ColorScheme
                 backend='matplotlib', # matplotlib or altair
                 axh=None, # only applies to matplotlib
                 tooltip=[], # only applies to altair, list of columns
                 ):
    if not data is None:
        assert adj_mat.shape[0] == data.shape[0]

    G = nx.from_numpy_array(adj_mat)
    if remove_isolates:
        ind = list(nx.isolates(G))
        G.remove_nodes_from(ind)
        G = nx.convert_node_labels_to_integers(G)
        if not data is None:
            data = data.iloc[np.array([i for i in range(data.shape[0]) if not i in ind])]
        if data.shape[0] == 0:
            return None

    pos = nx.nx_agraph.graphviz_layout(G, prog=layout)

    X = [pos[node][0] for node in pos]
    Y = [pos[node][1] for node in pos]

    plotdf = data.assign(_X=X, _Y=Y)
    subgraphs = np.zeros(plotdf.shape[0])
    for i, cc in enumerate(nx.connected_components(G)):
        ind = np.array([ni for ni in cc])
        subgraphs[ind] = i

    if type(node_size) is str and node_size in plotdf:
        sz = plotdf[node_size]
    elif np.isscalar(node_size):
        sz = np.ones(plotdf.shape[0]) * node_size
    else:
        sz = np.array(node_size)

    plotdf = plotdf.assign(_subgraph=subgraphs,
                           _size=sz)

    if backend == 'matplotlib':
        if hue_order is None and not node_hue is None:
            """Sort alphabetically"""
            # hue_order = sorted(data[hue].unique())
            """Sort based on frequency"""
            hue_order = data[node_hue].value_counts().index.tolist()
        if palette is None and not node_hue is None:
            # palette = sns.color_palette('Set2',  n_colors=data[node_hue].unique().shape[0])
            palette = [c for i,c in zip(range(len(hue_order)), itertools.cycle(mpl.cm.Set1.colors))]
        if not palette is None:
            color_map = {h:c for h,c in zip(hue_order, itertools.cycle(palette))}
        plotdf = plotdf.assign(_color=plotdf[node_hue].map(color_map))

        if not axh is None:
            plt.sca(axh)
        else:
            axh = plt.gca()
        sparams = dict(linewidth=node_linewidth, edgecolor='black', alpha=node_alpha, zorder=3)
        plt.scatter(x='_X', y='_Y', c='_color', s='_size', data=plotdf, **sparams)

        for e1, e2 in G.edges:
            if not e1 == e2:
                plt.plot([X[e1], X[e2]],
                         [Y[e1], Y[e2]], '-', alpha=0.5, color=edge_color, zorder=0, linewidth=edge_linewidth)

        axh.xaxis.set_visible(False)
        axh.yaxis.set_visible(False)
        axh.set_frame_on(False)
        return axh
    elif backend == 'altair':
        if palette is None and not node_hue is None:
            # palette = ['dodgerblue', 'tomato', 'black', 'green', 'eggplant']
            palette = 'category10'
        if not node_hue is None:
            """color_map = {h:c for h,c in zip(hue_order, itertools.cycle(palette))}
            col_dom = [c for c in color_map]
            col_rng = [color_map[c] for c in color_map]
            color_param = alt.Color(field=node_hue, type='nominal', scale=alt.Scale(domain=col_dom, range=col_rng))"""
            color_param = alt.Color(node_hue, scale=alt.Scale(scheme=palette))
        else:
            color_param = alt.Undefined
        if node_symbol is None or not node_symbol in plotdf:
            symbol_param = None
        else:
            symbol_param = node_symbol

        edge_df = _make_edge_dataframe(G, pos)
        marker_attrs = {}

        marker_attrs['strokeWidth'] = edge_linewidth
        marker_attrs['color'] = 'gray'
        marker_attrs['opacity'] = 0.5

        ch_edges = alt.Chart(edge_df).mark_line(**marker_attrs).encode(
                x=alt.X('x', axis=alt.Axis(title='', grid=False, labels=False, ticks=False)),
                y=alt.Y('y', axis=alt.Axis(title='', grid=False, labels=False, ticks=False)),
                detail='edge')
        ch_nodes = alt.Chart(plotdf.reset_index(drop=True)).mark_circle(size=node_size)#.interactive()
        ch_nodes = ch_nodes.encode(x=alt.X('_X'),
                                    y=alt.Y('_Y'),
                                    tooltip=tooltip,
                                    color=color_param)
        if not symbol_param is None:
            ch_nodes.encode(shape=symbol_param)
        ch = alt.layer(ch_edges, ch_nodes)
        return ch
