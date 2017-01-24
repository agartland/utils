import pandas as pd
import numpy as np
import matplotlib.patheffects
import matplotlib.transforms as mtrans
from matplotlib import transforms
import skbio
from objhist import objhist

__all__ = ['computeMotif',
           'plotMotif']

class Scale(matplotlib.patheffects.RendererBase):
    def __init__(self, sx=1., sy=1.):
        self._sx = sx
        self._sy = sy

    def draw_path(self, renderer, gc, tpath, affine, rgbFace):
        affine = affine.identity().scale(self._sx, self._sy) + affine
        renderer.draw_path(gc, tpath, affine, rgbFace)

def parseColor(aa, colorsDf):
    """Used to parse the colors from shapely_aa_colors.tsv"""
    ind = colorsDf['Amino Acids'].map(lambda aminoacids: aa in aminoacids)
    if ind.sum() == 0:
        return [0,0,0]
        # return 'black'
    else:
        # return colorsDf['Color Name'].loc[ind].tolist()[0]
        tmp = colorsDf['RGB Values'].loc[ind].tolist()[0]
        tmp = tmp.replace('[','').replace(']','').split(',')
        arr = [float(a)/255. for a in tmp]
        return arr

def computeMotif(seqs):
    """Compute heights for a sequence logo

    Parameters
    ----------
    seqs : list or pd.Series
        Alignment of AA sequences

    Returns
    -------
    motif : pd.DataFrame
        Heights for each symbol (index, rows) for each column.
        Heights reflect the fraction of total entropy contributed
        by that AA within each column of the alignment."""

    alphabet = skbio.sequence.Protein.alphabet
    
    if not type(seqs) is pd.Series:
        align = pd.Series(seqs)
    else:
        align = seqs

    L = len(align.iloc[0])
    nAA = len(alphabet)

    heights = np.zeros((nAA, L))
    for coli in range(L):
        aaOH = objhist([s[coli] for s in align])
        freq = aaOH.freq()
        p = np.array([freq.get(k, 0) for k in alphabet])
        pNZ = p[p>0]
        totEntropy = np.log2(nAA) - ((-pNZ * np.log2(pNZ)).sum())
        heights[:,coli] = p * totEntropy

    return pd.DataFrame(heights, columns=range(L), index=alphabet)

def plotMotif(x, y, motif, axh=None, fontsize=30, aa_colors='shapely'):
    """Sequence logo of the motif at data coordinates x,y using matplotlib.

    Note: logo will be distorted by zooming after it is plotted

    Inspiration:
    https://github.com/saketkc/motif-logos-matplotlib/blob/master/Motif%20Logos%20using%20matplotlib.ipynb

    Parameters
    ----------
    x,y : float
        Position for the bottom-left corner of the logo in the axes data coordinates
    motif : pd.DataFrame
        Matrix of scores for each AA symbol (index, rows) and each
        column in an alignment (columns). Values will be used to scale
        each symbol linearly.
    axh : matplotlib axes handle
        Will use plt.gca() if None
    fontsize : float
        Pointsize of font passed to axh.text
    aa_colors : str
        Either 'shapely' or a color to use for all symbols."""

    if aa_colors == 'shapely':
        colorsDf = pd.read_csv(GIT_PATH+'utils/shapely_aa_colors.tsv', delimiter='\t')

        aa_colors = {aa:parseColor(aa, colorsDf=colorsDf) for aa in motif.index}
    else:
        """All AAs have the same color, specified by aa_colors"""
        aa_colors = {aa:aa_colors for aa in motif.index}

    if axh is None:
        axh = plt.gca()


    xshift = 0
    for xi in range(motif.shape[1]):
        scores = motif.iloc[:, xi].sort_values()
        yshift = 0
        for yi, (aa, score) in enumerate(scores.iteritems()):
            if score > 0:
                txt = axh.text(x + xshift,
                              y + yshift,
                              aa, 
                              fontsize=30, 
                              color=aa_colors[aa],
                              family='monospace')
                window_ext = txt.get_window_extent(fig.canvas.get_renderer())
                ext_data = ax.transData.inverted().transform(window_ext)

                baseW = (ext_data[1][0] - ext_data[0][0])
                baseH = (ext_data[1][1] - ext_data[0][1])

                txt.set_path_effects([Scale(1., score)])
                yshift += baseH * score
        xshift += baseW
    plt.show()

