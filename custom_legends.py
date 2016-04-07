import matplotlib.pyplot as plt
import numpy as np

__all__ = ['colorLegend',
           'symbolLegend']
           
def colorLegend(colors, labels, alphas=None, edgecolor='black',loc='best', **legendKwargs):
    """Custom matplotlib legend with colors and labels etc.
    Useful in cases where it is awkward to include labels on the appropriate plot() objects.
    Parameters specify the characteristics of each line in the legend.

    Parameters
    ----------
    colors : list of valid matplotlib colors
    labels : list of strings
    alphas : list of alpha values
    edgecolor : single valid matplotlib color

    All remaining kwargs are passed to legend()
    """

    if alphas is None:
        alphas = np.ones(len(colors))
    circles = (plt.Circle((0,0), fc=c, ec=edgecolor, alpha=a) for c,a in zip(colors,alphas))
    lh = plt.legend(circles,
                    labels,
                    loc=loc,
                    **legendKwargs)
    return lh

def symbolLegend(symbols, labels, facecolors=None, edgecolors=None, alphas=None,loc='best', **legendKwargs):
    """Custom matplotlib legend with lines, symbols and labels etc.
    Useful in cases where it is awkward to include labels on the appropriate plot() objects.
    Parameters specify the characteristics of each line in the legend.

    Parameters
    ----------
    symbols : list of valid matplotlib symbols
        E.g. 'xs^*.<>' or other matplotlib.markers
    labels : list of strings
    facecolors : list of valid matplotlib colors
    edgecolors : list of valid matplotlib colors
    alphas : list of alpha values

    All remaining kwargs are passed to legend()
    """
    if alphas is None:
        alphas = np.ones(len(symbols))
    if edgecolors is None:
        edgecolors = ['black'] * len(symbols)
    if facecolors is None:
        facecolors = ['white'] * len(symbols)

    lh = plt.legend((plt.Line2D([0],[0], ls = '', marker = s, markerfacecolor = mfc, markeredgecolor = ec, alpha = a) for s,mfc,ec,a in zip(symbols,facecolors,edgecolors,alphas)),
                labels,
                loc,
                numpoints=1,
                **legendKwargs)

    return lh
