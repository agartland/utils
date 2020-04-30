import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects
import matplotlib.transforms as mtrans
from matplotlib import transforms
import skbio
import io
import warnings

import parasail

from svg_logo import svg_logo as SVGMotif

try:
    from muscle import *
except ImportError:
    print('Importing seqlogo without alignment support from MUSCLE.')

__all__ = ['compute_motif',
           'plot_motif',
           'reduce_gaps',
           'computePALMotif',
           'SVGMotif',
           'uniprot_frequency',
           'compute_relative_motif']

_shapely_colors = 'Amino Acids,Color Name,RGB Values,Hexadecimal\nD|E,bright red,"[230,10,10]",E60A0A\nC|M,yellow,"[230,230,0]",E6E600\nK|R,blue,"[20,90,255]",145AFF\nS|T,orange,"[250,150,0]",FA9600\nF|Y,mid blue,"[50,50,170]",3232AA\nN|Q,cyan,"[0,220,220]",00DCDC\nG,light grey,"[235,235,235]",EBEBEB\nL|V|I,green,"[15,130,15]",0F820F\nA,dark grey,"[200,200,200]",C8C8C8\nW,pink,"[180,90,180]",B45AB4\nH,pale blue,"[130,130,210]",8282D2\nP,flesh,"[220,150,130]",DC9682'

uniprot_frequency = {'A': 8.25,
                     'R': 5.53,
                     'N': 4.06,
                     'D': 5.45,
                     'C': 1.37,
                     'Q': 3.93,
                     'E': 6.75,
                     'G': 7.07,
                     'H': 2.27,
                     'I': 5.96,
                     'L': 9.66,
                     'K': 5.84,
                     'M': 2.42,
                     'F': 3.86,
                     'P': 4.70,
                     'S': 6.56,
                     'T': 5.34,
                     'W': 1.08,
                     'Y': 2.92,
                     'V': 6.87}

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


def compute_motif(seqs, reference_freqs=None, weights=None, align_first=True, gap_reduce=None, alphabet=None):
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

    if alphabet is None:
        alphabet = sorted([aa for aa in skbio.sequence.Protein.alphabet if not aa in '*.'])

    if weights is None:
        weights = np.ones(len(seqs))
    
    if align_first:
        align = muscle_align(seqs)
    else:
        if not isinstance(seqs, pd.Series):
            align = pd.Series(seqs)
        else:
            align = seqs
    if not gap_reduce is None:
        align = reduce_gaps(align, thresh=gap_reduce)

    L = len(align.iloc[0])
    nAA = len(alphabet)

    freq = _get_frequencies(align, alphabet, weights)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        if not reference_freqs is None:
            reference_freqs = np.array([reference_freqs.get(aa, 0) for aa in alphabet])
            reference_freqs = reference_freqs / np.sum(reference_freqs)
            reference_freqs = np.tile(reference_freqs[:, None], (1, L))
            heights = freq * np.log2(freq / reference_freqs)
            heights[np.isnan(heights)] = 0
        else:
            tot_entropy = np.log2(nAA) - np.nansum(-freq * np.log2(freq), axis=0, keepdims=True)
            heights = freq * tot_entropy

    return pd.DataFrame(heights, columns=list(range(L)), index=alphabet)

def reduce_gaps(align, thresh=0.7):
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

def pairwise_alignment_frequencies(centroid, seqs, gopen=3, gextend=3, matrix=parasail.blosum62):
    alphabet = sorted([aa for aa in skbio.sequence.Protein.alphabet if not aa in '*.'])
    centroid_query = parasail.profile_create_stats_sat(centroid, matrix=matrix)
    
    seq_algn = np.zeros((len(centroid), len(alphabet)))
    for s in seqs:
        # a = parasail.nw_trace(centroid, s, open=3, extend=3, matrix=parasail.blosum62)
        a = parasail.nw_trace_scan_profile_sat(centroid_query, s, open=gopen, extend=gextend)
        
        pos = 0
        for ref_aa, q_aa in zip(a.traceback.ref, a.traceback.query):
            if not q_aa == '-':
                seq_algn[pos, alphabet.index(ref_aa)] = seq_algn[pos, alphabet.index(ref_aa)] + 1
                pos += 1
    return pd.DataFrame(seq_algn, index=list(centroid), columns=alphabet)

def _get_frequencies(seqs, alphabet, weights, add_one=False):
    L = len(seqs[0])
    nAA = len(alphabet)
    
    freq = np.zeros((nAA, L))
    for coli in range(L):
        for si, s in enumerate(seqs):
            freq[alphabet.index(s[coli]), coli] += weights[si]
    if add_one:
        freq = (freq + 1) / (freq + 1).sum(axis=0, keepdims=True)
    else:
        freq = freq / freq.sum(axis=0, keepdims=True)
    return freq

def compute_relative_motif(seqs, refs):
    """Use log-OR scores indicating how likely it was to see the AA
    in the seqs vs. the refs. Seqs and refs must have equal length.

    Parameters
    ----------
    seqs : list
        List of AA sequences that are all similar to each other and the centroid
    refs : list
        List of all other sequences in the experiment as a reference.

    Returns
    -------
    A : pd.DataFrame [AA alphabet x position]
        A matrix of log-OR scores that can be used directly with the svg_logo function"""

    alphabet = sorted([aa for aa in skbio.sequence.Protein.alphabet if not aa in '*.'])

    """
    p_i is reference
    q_i is observed

    A = q * np.log2(q / p) - c(N)

    """
    """Adding 1 to avoid inf's, but this should be studied more carefully"""
    p = _get_frequencies(refs, alphabet, np.ones(len(refs)), add_one=True)
    q = _get_frequencies(seqs, alphabet, np.ones(len(seqs)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        A = q * np.log2(q/p)
    A[np.isnan(A)] = 0
    A = pd.DataFrame(A, index=alphabet)
    return A

def computePALMotif(centroid, seqs, refs, gopen=3, gextend=3, matrix=parasail.blosum62):
    """Compute pairwise alignments between the centroid and all sequences in seqs and refs. The motif
    will have the same length as the centroid with log-OR scores indicating how likely it was to see the AA
    in the seqs vs. the refs.

    Parameters
    ----------
    centroid : str
        Amino-acid sequence that is also among the seqs
    seqs : list
        List of AA sequences that are all similar to each other and the centroid
    refs : list
        List of all other sequences in the experiment as a reference.
    gopen : int
        Gap open penalty for parasail
    gextend : int
        Gap extend penalty for parasail
    matrix : substitution matrix
        Matrix from parasail for the alignment

    Returns
    -------
    A : pd.DataFrame [AA alphabet x position]
        A matrix of log-OR scores that can be used directly with the plotPALLogo function"""

    seq_algn = pairwise_alignment_frequencies(centroid, seqs, gopen=gopen, gextend=gextend, matrix=matrix)
    ref_algn = pairwise_alignment_frequencies(centroid, refs, gopen=gopen, gextend=gextend, matrix=matrix)

    """
    p_i is reference
    q_i is observed

    A = q * np.log2(q / p) - c(N)

    """
    """Adding 1 to avoid inf's, but this should be studied more carefully"""
    p = (ref_algn.values + 1) / (ref_algn.values + 1).sum(axis=1, keepdims=True)
    q = (seq_algn.values) / (seq_algn.values).sum(axis=1, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        A = q * np.log2(q/p)
    A[np.isnan(A)] = 0
    A = pd.DataFrame(A, index=ref_algn.index, columns=ref_algn.columns)
    #pdf = pd.DataFrame(p, index=ref_algn.index, columns=ref_algn.columns)
    #qdf = pd.DataFrame(q, index=ref_algn.index, columns=ref_algn.columns)
    return A.T

def _extend_bbox(bbox, ext):
    if ext.x0 < bbox[0, 0]:
        bbox[0, 0] = ext.x0
    if ext.y0 < bbox[0, 1]:
        bbox[0, 1] = ext.y0
    if ext.x1 > bbox[1, 0]:
        bbox[1, 0] = ext.x1
    if ext.y1 < bbox[1, 1]:
        bbox[1, 1] = ext.y1
    return bbox

def plot_motif(x, y, motif, axh=None, fontsize=16, aa_colors='black', highlightAAs=None, highlightColor='red'):
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

    if highlightAAs is None:
        highlightAAs = ['' for i in range(motif.shape[1])]
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
    bottom_offset = trans_offset
    neg_trans_offset = trans_offset
    first_pos = None
    last_neg = None
    bbox = np.array([[x, y], [x, y]])
    mxy = y 
    mny = y
    letters = [[] for i in range(motif.shape[1])]
    for xi in range(motif.shape[1]):
        scores = motif.iloc[:, xi]
        pos_scores = scores[scores > 0].sort_values()
        neg_scores = (-scores[scores < 0]).sort_values()
        yshift = 0
        neg_trans_offset = trans_offset
        for yi, (aa, score) in enumerate(pos_scores.items()):
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
            
            #bbox_data = axh.transData.inverted().transform(np.array([[window_ext.x0, window_ext.y0], [window_ext.x1, window_ext.y1]]))
            #bbox_data = window_ext.inverse_transformed(axh.transData)
            #plt.plot(bbox_data.corners()[:, 0], bbox_data.corners()[:, 1], '-k')

            bbox = _extend_bbox(bbox, window_ext)
            letters[xi].append((txt, window_ext.height * score))
            if first_pos is None:
                first_pos = window_ext

            if yshift == 0:
                bottom_offset = transforms.offset_copy(txt._transform, 
                                                      fig=axh.figure,
                                                      x=window_ext.width * 1.5,
                                                      units='dots')

            yshift += window_ext.height * score
            
            # print(xi, aa, window_ext.x0, window_ext.y0, '%1.1f' % score, '%1.0f' % window_ext.height, '%1.0f' % window_ext.width, '%1.1f' % yshift)
            trans_offset = transforms.offset_copy(txt._transform, 
                                                  fig=axh.figure,
                                                  y=window_ext.height * score,
                                                  units='dots')
        trans_offset = bottom_offset
        if yshift > mxy:
            mxy = yshift

    trans_offset = transforms.offset_copy(axh.transData, 
                                          fig=axh.figure, 
                                          x=0, y=0, 
                                          units='dots')    
    for xi in range(motif.shape[1]):
        scores = motif.iloc[:, xi]
        pos_scores = scores[scores > 0].sort_values()
        neg_scores = (-scores[scores < 0]).sort_values()
        yshift = 0
        for yi, (aa, score) in enumerate(neg_scores.items()):
            if aa in highlightAAs[xi]:
                color = highlightColor
            else:
                color = aa_colors[aa]
            txt = axh.text(x, 
                          y, 
                          aa, 
                          transform=trans_offset,
                          fontsize=fontsize, 
                          color='r',
                          ha='left',
                          va='top',
                          family='monospace')
            txt.set_path_effects([Scale(1.5, -score)])
            axh.figure.canvas.draw()
            window_ext = txt.get_window_extent(txt._renderer)
            bbox = _extend_bbox(bbox, window_ext)
            letters[xi].append((txt, window_ext.height * score))
            last_neg = window_ext

            if yshift == 0:
                bottom_offset = transforms.offset_copy(txt._transform, 
                                                      fig=axh.figure,
                                                      x=window_ext.width * 1.5,
                                                      units='dots')
            yshift -= window_ext.height * score
            
            # print(xi, aa, '%1.1f' % score, '%1.0f' % window_ext.height, '%1.0f' % window_ext.width, '%1.1f' % yshift)
            trans_offset = transforms.offset_copy(txt._transform, 
                                                      fig=axh.figure,
                                                      y=-window_ext.height * score,
                                                      units='dots')
        trans_offset = bottom_offset
        if yshift < mny:
            mny = yshift
        

    if last_neg is None:
        last_neg = first_pos
    plt.show()
    
    #bbox = np.array([[first_pos.x0, last_neg.y0], [window_ext.x1, window_ext.y0 + window_ext.height * score]])
    bbox_data = axh.transData.inverted().transform(bbox)
    return letters, bbox, bbox_data

