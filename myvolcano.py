import pandas as pd
import numpy as np
from os.path import join as opj
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

import altair as alt


def altair_scatter(x, y, hue, data, shape=None, tooltip=[], yscale='linear', xscale='linear', palette=None, size=60, stroke=2, fontsize=14, title='', reversex=False, reversey=False):
    # brush = alt.selection(type='single', resolve='global')
    # palette = {'Neither':'black', 'Y only':'gold', 'X only':'blue', 'Both':'red'}
    if palette is None and not hue is None:
        palette = 'tableau10'
        # tmp = sns.color_palette('Set2',  n_colors=data[hue].unique().shape[0])
        """tmp = ['dodgerblue', 'tomato', 'black', 'green', 'eggplant']
        palette = {h:c for h,c in zip(data[hue].unique(), tmp)}

        col_dom = [c for c in palette]
        col_rng = [palette[c] for c in palette]"""

    # brush = alt.selection_single(resolve='global')
    # base = alt.Chart(data).add_selection(brush).mark_point(size=size, strokeWidth=stroke).interactive()
    base = alt.Chart(data).mark_point(size=size, strokeWidth=stroke).interactive()

    if not palette is None:
        #print(palette)
        #color_param = alt.Color(field=hue, type='nominal', scale=alt.Scale(domain=col_dom, range=col_rng))
        color_param = alt.Color(field=hue, type='nominal', scale=alt.Scale(scheme=palette), legend=alt.Legend(orient='right'))
        #color_param = alt.Color(field=hue, type='nominal', scale=alt.Scale(scheme=palette))
        # color_param = alt.condition(brush, tmp_color, alt.ColorValue('gray')),
    else:
        color_param = alt.Undefined

    ch = base.encode(x=alt.X(x, scale=alt.Scale(type=xscale, reverse=reversex)),
                     y=alt.Y(y, scale=alt.Scale(type=yscale, reverse=reversey)),
                     tooltip=tooltip,
                     color=color_param)
    if not shape is None:
        ch.encode(shape=shape)

    ch = ch.properties(title=title)
    # ch = ch.configure_title(fontSize=fontsize).configure_axis(labelFontSize=fontsize-2, titleFontSize=fontsize)
    return ch

def altair_scatter_select(selector, data, scatter_ch):
    selection = alt.selection_multi(fields=[selector])
    #color = alt.condition(selection, alt.Color(field=selector, type='nominal', scale=alt.Scale(scheme='dark2'), legend=None), alt.value('lightgray'))
    color = alt.condition(selection, alt.value('black'), alt.value('lightgray'))
    selector_ch = alt.Chart(data).mark_rect().encode(y=selector, color=color).add_selection(selection)

    ch = scatter_ch.transform_filter(selection)

    return alt.hconcat(selector_ch, ch).resolve_scale(color='independent')


"""
def altair_scatter_select(x, y, hue, data, selector, tooltip=[], yscale='linear', xscale='linear', palette=None, size=60, stroke=2, fontsize=14, title='', reversex=False, reversey=False):
    # brush = alt.selection(type='single', resolve='global')
    # palette = {'Neither':'black', 'Y only':'gold', 'X only':'blue', 'Both':'red'}
    if palette is None and not hue is None:
        # tmp = sns.color_palette('Set2',  n_colors=data[hue].unique().shape[0])
        # tmp = ['dodgerblue', 'tomato', 'black', 'green', 'eggplant']
        tmp = list(mpl.cm.tab10.colors)
        palette = {h:c for h,c in zip(data[hue].unique(), tmp)}

        col_dom = [c for c in palette]
        col_rng = [palette[c] for c in palette]

    # brush = alt.selection_single(resolve='global')
    # base = alt.Chart(data).add_selection(brush).mark_point(size=size, strokeWidth=stroke).interactive()
    base = alt.Chart(data).mark_point(size=size, strokeWidth=stroke).interactive()

    if not palette is None:
        #color_param = alt.Color(field=hue, type='nominal', scale=alt.Scale(domain=col_dom, range=col_rng))
        color_param = alt.Color(field=hue, type='nominal', scale=alt.Scale(scheme='dark2'), legend=alt.Legend(orient='right'))
        # color_param = alt.condition(brush, tmp_color, alt.ColorValue('gray')),
    else:
        color_param = alt.Undefined

    selection = alt.selection_multi(fields=[selector])
    #color = alt.condition(selection, alt.Color(field=selector, type='nominal', scale=alt.Scale(scheme='dark2'), legend=None), alt.value('lightgray'))
    color = alt.condition(selection, alt.value('black'), alt.value('lightgray'))
    selector_ch = alt.Chart(data).mark_rect().encode(y=selector, color=color).add_selection(selection)

    #color_param = alt.Color(field=hue, type='nominal', legend=alt.Legend(orient='right'), scale=alt.Scale(scheme='tab10'))

    ch = base.encode(x=alt.X(x, scale=alt.Scale(type=xscale, reverse=reversex)),
                     y=alt.Y(y, scale=alt.Scale(type=yscale, reverse=reversey)),
                     tooltip=tooltip,
                     color=color_param).properties(title=title).transform_filter(selection)
    # ch = ch.configure_title(fontSize=fontsize).configure_axis(labelFontSize=fontsize-2, titleFontSize=fontsize)

    return alt.hconcat(selector_ch, ch).resolve_scale(color='independent')"""


def plot_volcano_altair(df, pvalue_col, or_col, hue_col, ann_cols=[], censor_or=None):
    if not censor_or is None:
        df = df.copy()
        df.loc[df[or_col] < 1/censor_or, or_col] = 1/censor_or
        df.loc[df[or_col] > censor_or, or_col] = censor_or

    
    tt = ann_cols + [c for c in [or_col, pvalue_col] if not c in ann_cols]

    ch = altair_scatter(x=or_col, y=pvalue_col, hue=hue_col, data=df, tooltip=tt, xscale='log', yscale='log')
    return ch


def plot_volcano(df, pvalue_col, or_col, sig_col, ann_col=None, annotate=None, censor_or=None):
    
    if not censor_or is None:
        df = df.copy()
        df.loc[df[or_col] < 1/censor_or, or_col] = 1/censor_or
        df.loc[df[or_col] > censor_or, or_col] = censor_or

    fc_ticks = [5, 4, 3, 2.5, 2, 1.5]
    xticks = [1/x for x in fc_ticks] + [1] + [x for x in fc_ticks[::-1]]
    xtick_labs = [f'-{x}' for x in fc_ticks] + [1] + [f'{x}' for x in fc_ticks[::-1]]

    p_func = lambda p: -10*np.log10(p)
    sig_ind = df[sig_col]
    
    figh = plt.figure(figsize=(7, 5))
    axh = figh.add_axes([0.15, 0.15, 0.7, 0.7], xscale='log', yscale='log')
    axh.set_axisbelow(True)
    plt.grid(True, linewidth=1)
    axh.yaxis.set_minor_locator(AutoMinorLocator())
    mx_fc = np.exp(np.max(np.abs(np.log(df[or_col]))))
    
    plt.scatter(df.loc[~sig_ind, or_col],
                df.loc[~sig_ind, pvalue_col],
                color='black', alpha=0.3, s=5, zorder=2)

    plt.scatter(df.loc[sig_ind, or_col],
                df.loc[sig_ind, pvalue_col],
                color='r', alpha=0.9, s=5, zorder=3)
    yl = plt.ylim()
    # plt.plot([-1.1*mx_fc, mx_fc*1.1], [p_thresh]*2, '--k')

    if not annotate is None:
        tmp = df.loc[sig_ind].sort_values(by=or_col, ascending=False).set_index(ann_col)
        bottomN = tmp.index[-annotate:].tolist()
        topN = tmp.index[:annotate].tolist()
        tmp = df.loc[sig_ind].sort_values(by=pvalue_col).set_index(ann_col)
        topP = tmp.index[:annotate].tolist()
        tmp = df.loc[sig_ind].set_index(ann_col)
        for i in np.unique(topN + bottomN + topP):
            xy = (tmp[[or_col, pvalue_col]].loc[i]).values
            xy[1] = xy[1]
            xy[0] = xy[0]
            if xy[0] > 1:
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
                         size=6,
                         **p)

    if censor_or is None:
        plt.xlim([1/(1.1*mx_fc), mx_fc*1.1])
    else:
        plt.xlim((1/censor_or, censor_or))

    plt.ylim((1, yl[0]))
    #plt.ylim((1, np.min(df[pvalue_col])))
    plt.ylabel(pvalue_col)
    plt.xlabel(or_col)
    plt.xticks(xticks, xtick_labs)
    return figh

def _test_data():
    df = pd.DataFrame(dict(x=np.random.rand(20),
                            y=np.random.normal(size=20),
                            category=np.random.choice(['A','B', 'C'], size=20)))
    return df