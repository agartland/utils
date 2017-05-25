import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects
import matplotlib.transforms as mtrans
from matplotlib import transforms
import skbio
from objhist import objhist
import io

__all__ = ['computeMotif',
           'plotMotif']

_shapely_colors = "Amino Acids,Color Name,RGB Values,Hexadecimal\nD|E,bright red,[230,10,10],E60A0A\nC|M,yellow,[230,230,0],E6E600\nK|R,blue,[20,90,255],145AFF\nS|T,orange,[250,150,0],FA9600\nF|Y,mid blue,[50,50,170],3232AA\nN|Q,cyan,[0,220,220],00DCDC\nG,light grey,[235,235,235],EBEBEB\nL|V|I,green,[15,130,15],0F820F\nA,dark grey,[200,200,200],C8C8C8\nW,pink,[180,90,180],B45AB4\nH,pale blue,[130,130,210],8282D2\nP,flesh,[220,150,130],DC9682"

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
        return [0, 0, 0]
        # return 'black'
    else:
        # return colorsDf['Color Name'].loc[ind].tolist()[0]
        tmp = colorsDf['RGB Values'].loc[ind].tolist()[0]
        tmp = tmp.replace('[', '').replace(']', '').split(',')
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
    
    if not isinstance(seqs, pd.Series):
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
        heights[:, coli] = p * totEntropy

    return pd.DataFrame(heights, columns=list(range(L)), index=alphabet)

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
        Either 'shapely' or a color to use for all symbols.

    Returns
    -------
    bbox : [[x0, y0], [x1, y1]]
        Full extent of the logo in data coodinates."""

    if aa_colors == 'shapely':
        colorsDf = pd.read_csv(io.StringIO(_shapely_colors), delimiter=',')
        aa_colors = {aa:parseColor(aa, colorsDf=colorsDf) for aa in motif.index}
    else:
        """All AAs have the same color, specified by aa_colors"""
        aa_colors = {aa:aa_colors for aa in motif.index}

    if axh is None:
        axh = plt.gca()

    mxy = 0
    xshift = 0
    for xi in range(motif.shape[1]):
        scores = motif.iloc[:, xi].sort_values()
        yshift = 0
        for yi, (aa, score) in enumerate(scores.items()):
            if score > 0:
                txt = axh.text(x + xshift,
                              y + yshift,
                              aa, 
                              fontsize=fontsize, 
                              color=aa_colors[aa],
                              family='monospace')
                # axh.figure.canvas.draw()
                # window_ext = txt.get_window_extent(txt._renderer)
                window_ext = txt.get_window_extent(axh.figure.canvas.get_renderer())
                ext_data = axh.transData.inverted().transform(window_ext)

                baseW = (ext_data[1][0] - ext_data[0][0])
                baseH = (ext_data[1][1] - ext_data[0][1])

                txt.set_path_effects([Scale(1., score)])
                yshift += baseH * score
        if yshift > mxy:
            mxy = yshift
        xshift += baseW * 1.
    plt.show()
    return np.array([[x, y], [xshift, mxy]])