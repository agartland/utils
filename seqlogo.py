import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects
import matplotlib.transforms as mtrans
from matplotlib import transforms
import skbio
import io
import warnings

try:
    from muscle import *
except ImportError:
    print('Importing seqlogo without alignment support from MUSCLE.')

__all__ = ['computeMotif',
           'plotMotif',
           'reduceGaps']

_shapely_colors = 'Amino Acids,Color Name,RGB Values,Hexadecimal\nD|E,bright red,"[230,10,10]",E60A0A\nC|M,yellow,"[230,230,0]",E6E600\nK|R,blue,"[20,90,255]",145AFF\nS|T,orange,"[250,150,0]",FA9600\nF|Y,mid blue,"[50,50,170]",3232AA\nN|Q,cyan,"[0,220,220]",00DCDC\nG,light grey,"[235,235,235]",EBEBEB\nL|V|I,green,"[15,130,15]",0F820F\nA,dark grey,"[200,200,200]",C8C8C8\nW,pink,"[180,90,180]",B45AB4\nH,pale blue,"[130,130,210]",8282D2\nP,flesh,"[220,150,130]",DC9682'

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

def computeMotif(seqs, weights=None, alignFirst=True, gapReduce=None):
    """Compute heights for a sequence logo

    Parameters
    ----------
    seqs : list or pd.Series
        Alignment of AA sequences
    weights : np.array
        Weights for each sequence

    Returns
    -------
    motif : pd.DataFrame
        Heights for each symbol (index, rows) for each column.
        Heights reflect the fraction of total entropy contributed
        by that AA within each column of the alignment."""

    alphabet = sorted([aa for aa in skbio.sequence.Protein.alphabet if not aa in '*.'])

    if weights is None:
        weights = np.ones(len(seqs))
    
    if alignFirst:
        align = muscle_align(seqs)
    else:
        if not isinstance(seqs, pd.Series):
            align = pd.Series(seqs)
        else:
            align = seqs
    if not gapReduce is None:
        align = reduceGaps(align, thresh=gapReduce)

    L = len(align.iloc[0])
    nAA = len(alphabet)
    
    freq = np.zeros((nAA, L,))
    for coli in range(L):
        for si,s in enumerate(align):
            freq[alphabet.index(s[coli]), coli] += weights[si]
    freq = freq / freq.sum(axis=0, keepdims=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        totEntropy = np.log2(nAA) - np.nansum(-freq * np.log2(freq), axis=0, keepdims=True)
    heights = freq * totEntropy

    return pd.DataFrame(heights, columns=list(range(L)), index=alphabet)

def plotMotif(x, y, motif, axh=None, fontsize=16, aa_colors='black', highlightAAs=None, highlightColor='red'):
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
    highlightAAs : list of lists
        Lists AAs in each position that should be plotted in highlight color.
    highlightColor : string or RGB
        Valid color for highlighted AAs

    Returns
    -------
    letters : list of lists of tuples (Text, height)
        List of lists of the Text objects in the order they are plotted
        (lower left to upper right). Useful for putting annotations relative
        to specific objects in the motif.
    bbox : [[x0, y0], [x1, y1]]
        Full extent of the logo in screen coodinates.
    bbox_data : [[x0, y0], [x1, y1]]
        Full extent of the logo in data coodinates."""

    if aa_colors == 'shapely':
        colorsDf = pd.read_csv(io.StringIO(_shapely_colors), delimiter=',')
        aa_colors = {aa:parseColor(aa, colorsDf=colorsDf) for aa in motif.index}
    else:
        """All AAs have the same color, specified by aa_colors"""
        aa_colors = {aa:aa_colors for aa in motif.index}

    if axh is None:
        axh = plt.gca()
  
    trans_offset = transforms.offset_copy(axh.transData, 
                                      fig=axh.figure, 
                                      x=0, y=0, 
                                      units='dots')
    first = None
    xshift = 0
    mxy = 0
    letters = [[] for i in range(motif.shape[1])]
    for xi in range(motif.shape[1]):
        scores = motif.iloc[:, xi].sort_values()
        yshift = 0
        for yi, (aa, score) in enumerate(scores.items()):
            if score > 0:
                if aa in highlightAAs[xi]:
                    color = highlightColor
                else:
                    color = aa_colors[aa]
                txt = axh.text(x, 
                              y, 
                              aa, 
                              transform=trans_offset,
                              fontsize=fontsize, 
                              color=color,
                              ha='left',
                              family='monospace')
                txt.set_path_effects([Scale(1.5, score)])
                axh.figure.canvas.draw()
                window_ext = txt.get_window_extent(txt._renderer)
                letters[xi].append((txt, window_ext.height * score))
                if first is None:
                    first = window_ext

                if yshift == 0:
                    xshift += window_ext.width * 1.5
                    bottom_offset = transforms.offset_copy(txt._transform, 
                                                          fig=axh.figure,
                                                          x=window_ext.width * 1.5,
                                                          units='dots')

                yshift += window_ext.height * score
                
                # print(xi, aa, '%1.1f' % score, '%1.0f' % window_ext.height, '%1.0f' % window_ext.width, '%1.1f' % yshift)
                trans_offset = transforms.offset_copy(txt._transform, 
                                                      fig=axh.figure,
                                                      y=window_ext.height * score,
                                                      units='dots')

        trans_offset = bottom_offset
        if yshift > mxy:
            mxy = yshift

    plt.show()
    bbox = np.array([[first.x0, first.y0], [window_ext.x1, window_ext.y0 + window_ext.height * score]])
    bbox_data = axh.transData.inverted().transform(bbox)
    return letters, bbox, bbox_data

def reduceGaps(align, thresh=0.7):
    """Identifies columns with thresh fraction of gaps,
    removes the sequences that have AAs there and
    removes the column from the alignment"""

    removePos = []
    for pos in range(len(align.iloc[0])):
        if np.mean(align.map(lambda seq: seq[pos] == '-')) > thresh:
            removePos.append(pos)
    for pos in removePos:
        align = align.loc[align.map(lambda seq: seq[pos] == '-')]

    return align.map(lambda seq: ''.join([aa for pos, aa in enumerate(seq) if not pos in removePos]))