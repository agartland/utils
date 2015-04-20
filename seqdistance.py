from __future__ import division
from functools import *
import itertools
import operator
from Bio import SeqIO, pairwise2
from Bio.SubsMat.MatrixInfo import blosum90, ident, blosum62
from copy import deepcopy
from numpy import *
import sys
import numpy as np
import numba as nb


from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA,KernelPCA
from sklearn import cluster
from sklearn.manifold import Isomap
import tsne
import pytsne

__all__ = ['BADAA',
           'AALPHABET',
           'AA2CODE',
           'CODE2AA',
           'isvalidpeptide',
           'removeBadAA',
           'hamming_distance',
           'trunc_hamming',
           'dichot_hamming',
           'seq2vec',
           'hamming_distance_vec',
           'nanGapScores',
           'nanZeroGapScores',
           'binGapScores',
           'blosum90GapScores',
           'binarySubst',
           'addGapScores',
           'seq_similarity',
           'seq_distance',
           'seq_similarity_old',
           'unalign_similarity',
           '_test_seq_similarity',
           'calcDistanceMatrix',
           'calcDistanceRectangle',
           'blosum90',
           'ident',
           'blosum62',
            'embedDistanceMatrix']


BADAA = '-*BX#Z'
FULL_AALPHABET = 'ABCDEFGHIKLMNPQRSTVWXYZ'
AALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA2CODE = {aa:i for i,aa in enumerate(FULL_AALPHABET)}
AA2CODE.update({'-':23})
CODE2AA = {i:aa for i,aa in enumerate(FULL_AALPHABET)}
CODE2AA.update({23:'-'})

def subst2mat(subst,alphabet = FULL_AALPHABET):
    """Converts a substitution dictionary
    (like those from Bio) into a numpy 2d substitution matrix"""
    mat = nan * zeros((len(alphabet),len(alphabet)), dtype = float64)
    for (aa1,aa2),v in subst.items():
        mat[alphabet.index(aa1),alphabet.index(aa2)] = v
    return mat

"""Many different ways of handling gaps. Remember that these are SIMILARITY scores"""
nanGapScores={('-','-'):nan,
              ('-','X'):nan,
              ('X','-'):nan}

nanZeroGapScores={('-','-'):nan,
                   ('-','X'):0,
                   ('X','-'):0}
"""Default for addGapScores()"""
binGapScores={('-','-'):1,
              ('-','X'):0,
              ('X','-'):0}
"""Arbitrary/reasonable values (extremes for blosum90 I think)"""
blosum90GapScores={('-','-'):5,
                   ('-','X'):-11,
                   ('X','-'):-11}

binarySubst = {(aa1,aa2):float(aa1==aa2) for aa1,aa2 in itertools.product(FULL_AALPHABET,FULL_AALPHABET)}

identMat = subst2mat(ident)
blosum90Mat = subst2mat(blosum90)
blosum62Mat = subst2mat(blosum62)
binaryMat = subst2mat(binarySubst)


def isvalidpeptide(mer,badaa=None):
    """Test if the mer contains an BAD amino acids in global BADAA
    typically -*BX#Z"""
    if badaa is None:
        badaa=BADAA
    if not mer is None:
        return not re.search('[%s]' % badaa,mer)
    else:
        return False
def removeBadAA(mer,badaa=None):
    """Remove badaa amino acids from the mer, default badaa is -*BX#Z"""
    if badaa is None:
        badaa=BADAA
    if not mer is None:
        return re.sub('[%s]' % badaa,'',mer)
    else:
        return mer

def hamming_distance(str1, str2, **kwargs):
    """Hamming distance between str1 and str2.
    Only finds distance over the length of the shorter string.
    **kwargs are so this can be plugged in place of a seq_distance() metric"""
    if isinstance(str1,basestring):
        str1 = string2byte(str1)

    if isinstance(str2,basestring):
        str2 = string2byte(str2)
    return nb_hamming_distance(str1, str2)

def aamismatch_distance(seq1,seq2, **kwargs):
    if isinstance(seq1,basestring):
        seq1 = seq2vec(seq1)

    if isinstance(seq2,basestring):
        seq2 = seq2vec(seq2)
    dist12 = nb_seq_similarity(seq1, seq2, substMat = binaryMat, normed = False, asDistance = True)
    return dist12

def string2byte(s):
    """Convert string to byte array since numba can't handle strings"""
    if is_string_like(s):
        s = array(s)
    dtype = s.dtype
    if dtype is numpy.dtype('byte'):
        return s # it's already a byte array
    shape = list(s.shape)
    n = dtype.itemsize
    shape.append(n)
    return s.ravel().view(dtype='byte').reshape(shape)

@nb.jit(nb.int32(nb.char[:],nb.char[:]), nopython = True)
def nb_hamming_distance(str1,str2):
    tot = 0
    for s1,s2 in zip(str1,str2):
        if s1 != s2:
            tot += 1
    return tot

def trunc_hamming(seq1,seq2,maxDist=2,**kwargs):
    """Truncated hamming distance
    d = hamming() if d<maxDist else d = maxDist"""
    d = hamming_distance(seq1,seq2)
    return maxDist if d>=maxDist else d

def dichot_hamming(seq1,seq2,mmTolerance=1,**kwargs):
    """Dichotamized hamming distance.
    hamming <= mmTolerance is 0 and all others are 1"""
    d = hamming_distance(seq1,seq2)
    return 1 if d>mmTolerance else 0

def seq2vec(seq):
    """Convert AA sequence into numpy vector of integers for fast comparison"""
    vec = zeros(len(seq), dtype=int32)
    for aai,aa in enumerate(seq):
        vec[aai] = AA2CODE[aa]
    return vec

def addGapScores(subst, gapScores = None, minScorePenalty = False, returnMat = False):
    """Add gap similarity scores for each AA (Could be done once for a set of sequences to improve speed)
    if gapScores is None then it will use defaults:
        gapScores={('-','-'):1,
                   ('-','X'):0,
                   ('X','-'):0}
    OR for blosum90 default is:
        blosum90GapScores={('-','-'):5,
                   ('-','X'):-11,
                   ('X','-'):-11}
    """
    if minScorePenalty:
        gapScores = {('-','-') : 1,
                     ('-','X') : min(subst.values()),
                     ('X','-') : min(subst.values())}
    elif gapScores is None:
        if subst is binarySubst:
            print 'Using default binGapScores for binarySubst'
            gapScores = binGapScores
        elif subst is blosum90:
            print 'Using default blosum90 gap scores'
            gapScores = blosum90GapScores
        else:
            raise Exception('Cannot determine which gap scores to use!')
    su = deepcopy(subst)
    uAA = unique([k[0] for k in subst.keys()])
    su.update({('-',aa) : gapScores[('-','X')] for aa in uAA})
    su.update({(aa,'-') : gapScores[('X','-')] for aa in uAA})
    su.update({('-','-') : gapScores[('-','-')]})

    if returnMat:
        return subst2mat(su)
    return su

@nb.jit(nb.float64(nb.int32[:],nb.int32[:],nb.float64[:,:],nb.boolean,nb.boolean), nopython = True)
def nb_seq_similarity(seq1, seq2, substMat, normed, asDistance):
    """Computes sequence similarity based on the substitution matrix."""
    if seq1.shape[0] != seq2.shape[0]:
        raise IndexError

    if normed or asDistance:
        sim12 = 0.
        siteN = 0.
        sim11 = 0.
        sim22 = 0.
        for i in range(seq1.shape[0]):
            cur12 = substMat[seq1[i],seq2[i]]
            cur11 = substMat[seq1[i],seq1[i]]
            cur22 = substMat[seq2[i],seq2[i]]
            if not np.isnan(cur12):
                sim12 += cur12
                siteN += 1.
            if not np.isnan(cur11):
                sim11 += cur11
            if not np.isnan(cur22):
                sim22 += cur22
        sim12 = 2*sim12/((sim11/siteN) + (sim22/siteN))
    else:
        sim12 = 0.
        siteN = 0.
        for i in range(seq1.shape[0]):
            if not np.isnan(substMat[seq1[i],seq2[i]]):
                sim12 += substMat[seq1[i],seq2[i]]
                siteN += 1.

    if asDistance:
        if normed:
            sim12 = (siteN - sim12)/siteN
        else:
            sim12 = siteN - sim12
    return sim12

def np_seq_similarity(seq1, seq2, substMat, normed, asDistance):
    """Computes sequence similarity based on the substitution matrix."""
    if seq1.shape[0] != seq2.shape[0]:
        raise IndexError, "Sequences must be the same length (%d != %d)." % (seq1.shape[0],seq2.shape[0])

    """Similarity between seq1 and seq2 using the substitution matrix subst"""
    sim12 = substMat[seq1,seq2]

    if normed or asDistance:
        siteN = (~isnan(sim12)).sum()
        sim11 = np.nansum(substMat[seq1,seq1])/siteN
        sim22 = np.nansum(substMat[seq1,seq1])/siteN
        tot12 = np.nansum(2*sim12)/(sim11+sim22)
    else:
        tot12 = np.nansum(sim12)

    if asDistance:
        """Distance between seq1 and seq2 using the substitution matrix subst
            because seq_similarity returns a total similarity with max of siteN (not per site), we use
                d = siteN - sim
            which is a total normed distance, not a per site distance"""
        if normed:
            tot12 = (siteN - tot12)/siteN
        else:
            tot12 = siteN - tot12
    return tot12

def seq_similarity(seq1, seq2, subst = None, normed = True, asDistance = False):
    """Compare two sequences and return the similarity of one and the other
    If the seqs are of different length then it raises an exception

    FOR HIGHLY DIVERGENT SEQUENCES THIS NORMALIZATION DOES NOT GET TO [0,1] BECAUSE OF EXCESS NEGATIVE SCORES!
    Consider normalizing the matrix first by adding the min() so that min = 0 (but do not normalize per comparison)
    
    Return a nansum of site-wise similarities between two sequences based on a substitution matrix
        [0, siteN] where siteN ignores nan similarities which may depend on gaps
        sim12 = nansum(2*sim12/(nanmean(sim11) + nanmean(sim22))
    Optionally specify normed = False:
        [0, total raw similarity]

    This returns a score [0, 1] for binary and blosum based similarities
        otherwise its just the sum of the raw score out of the subst matrix"""

    if subst is None:
        print 'Using default binarySubst matrix with binGaps for seq_similarity'
        subst = addGapScores(binarySubst, binGapScores)

    if isinstance(subst,dict):
        subst = subst2mat(subst)

    if isinstance(seq1,basestring):
        seq1 = seq2vec(seq1)

    if isinstance(seq2,basestring):
        seq2 = seq2vec(seq2)

    result = np_seq_similarity(seq1, seq2, substMat = subst, normed = normed, asDistance = asDistance)
    return result


def seq_similarity_old(seq1,seq2,subst=None,normed=True):
    """Compare two sequences and return the similarity of one and the other
    If the seqs are of different length then it raises an exception
    FOR HIGHLY DIVERGENT SEQUENCES THIS NORMALIZATION DOES NOT GET TO [0,1] BECAUSE OF EXCESS NEGATIVE SCORES!
    Consider normalizing the matrix first by adding the min() so that min = 0 (but do not normalize per comparison)
    
    Return a nansum of site-wise similarities between two sequences based on a substitution matrix
        [0, siteN] where siteN ignores nan similarities which may depend on gaps
        sim12 = nansum(2*sim12/(nanmean(sim11) + nanmean(sim22))
    Optionally specify normed = False:
        [0, total raw similarity]
    This returns a score [0, 1] for binary and blosum based similarities
        otherwise its just the sum of the raw score out of the subst matrix
    
    For a hamming similarity when there are no gaps use subst=binarySubst
        and performance is optimized underneath using hamming_distance"""

    assert len(seq1)==len(seq2), "len of seq1 (%d) and seq2 (%d) are different" % (len(seq1),len(seq2))

    if subst is binarySubst:
        dist=hamming_distance(seq1,seq2)
        sim=len(seq1)-dist
        if normed:
            sim=sim/len(seq1)
        return sim

    if subst is None:
        print 'Using default binarySubst matrix with binGaps for seq_similarity'
        subst=addGapScores(binarySubst,binGapScores)

    """Distance between seq1 and seq2 using the substitution matrix subst"""
    sim12=array([i for i in itertools.imap(lambda a,b: subst.get((a,b),subst.get((b,a))), seq1, seq2)])

    if normed:
        siteN=sum(~isnan(sim12))
        sim11=seq_similarity_old(seq1,seq1,subst=subst,normed=False)/siteN
        sim22=seq_similarity_old(seq2,seq2,subst=subst,normed=False)/siteN
        sim12=nansum(2*sim12/(sim11+sim22))
    else:
        sim12=nansum(sim12)
    return sim12
    
def seq_distance(seq1, seq2, subst = None, normed = True):
    """Compare two sequences and return the distance from one to the other
    If the seqs are of different length then it raises an exception

    Returns a scalar [0, siteN] where siteN ignores nan similarities which may depend on gaps
    Optionally returns normed = True distance:
        [0, 1]

    Note that either way the distance is "normed", its either per site (True) or total normed (False):
        [0, siteN]"""
    return seq_similarity(seq1, seq2, subst = subst, normed = normed, asDistance = True)


def unalign_similarity(seq1,seq2,subst=None):
    """Compare two sequences by aligning them first with pairwise alignment
       and return the distance from one to the other"""
    
    if subst is None:
        subst=blosum90

    res=pairwise2.align.globaldx(seq1,seq2,subst)
    return res[0][2]

def _test_seq_similarity(subst=None,normed=True):
    def test_one(s,sname,n,seq1,seq2):
        print seq1
        print seq2
        try:
            sim = seq_similarity(seq1,seq2,subst=s,normed=n)
            print 'Similarity: %f' % sim
        except:
            print 'Similarity: %s [%s]' % (sys.exc_info()[0],sys.exc_info()[1])
        
        #dist = seq_distance(seq1,seq2,subst=s)
        try:
            dist = seq_distance(seq1,seq2,subst=s)
            print 'Distance: %f' % dist
        except:
            print 'Distance: %s [%s]' % (sys.exc_info()[0],sys.exc_info()[1])
        print

    seqs = ['AAAA',
            'AAAA',
            'BBBB',
            'AABA',
            '-AAA',
            '-A-A']
    if subst is None:
        subst = [addGapScores(binarySubst,binGapScores),
                 addGapScores(binarySubst,nanZeroGapScores),
                 addGapScores(blosum90,blosum90GapScores),
                 addGapScores(blosum90,nanGapScores)]
        names = ['addGapScores(binarySubst,binGapScores)',
                 'addGapScores(binarySubst,nanZeroGapScores)',
                 'addGapScores(blosum90,blosum90GapScores)',
                 'addGapScores(blosum90,nanGapScores)']
        for s,sname in zip(subst,names):
            print 'Using %s normed = %s' % (sname,normed)
            for seq1,seq2 in itertools.combinations(seqs,2):
                test_one(s,sname,normed,seq1,seq2)
    else:
        for seq1,seq2 in itertools.combinations(seqs,2):
            test_one(subst,'supplied subst',normed,seq1,seq2)

def calcDistanceMatrix(seqs,normalize=False,symetric=True,metric=None,**kwargs):
    """Returns a square distance matrix with rows and columns of the unique sequences in seqs
    By default will normalize by subtracting off the min() to at least get rid of negative distances
    However, I don't really think this is the best option. 
    If symetric is True then only calculates dist[i,j] and assumes dist[j,i] == dist[i,j]
    Additional kwargs are passed to the distanceFunc (e.g. subst, gapScores, normed)

    Parameters
    ----------
    seqs : list/iterator
        Genetic sequences to compare.
    normalize : bool
        If true (default: False), subtracts off dist.min() to eliminate negative distances
        (Could be improved/expanded)
    symetric : bool
        If True (default), then it assumes dist(A,B) == dist(B,A) and speeds up computation.
    metric : function with params seq1, seq2 and possibly additional kwargs
        Function will be called to compute each pairwise distance.
    kwargs : additional keyword arguments
        Will be passed to each call of metric()

    Returns
    -------
    dist : ndarray of shape [len(seqs), len(seqs)]
        Contains all pairwise distances for seqs.
    """
    return calcDistanceRectangle(seqs,seqs,normalize=normalize,symetric=symetric,metric=metric,**kwargs)

def calcDistanceRectangle(row_seqs,col_seqs,normalize=False,symetric=False,metric=None,convertToNP=False,**kwargs):
    """Returns a rectangular distance matrix with rows and columns of the unique sequences in row_seqs and col_seqs
    By default will normalize by subtracting off the min() to at least get rid of negative distances
    However, I don't really think this is the best option. 
    If symetric is True then only calculates dist[i,j] and assumes dist[j,i] == dist[i,j]
    
    Additional kwargs are passed to the distanceFunc (e.g. subst, gapScores, normed)

    Parameters
    ----------
    row_seqs : list/iterator
        Genetic sequences to compare.
    col_seqs : list/iterator
        Genetic sequences to compare.
    normalize : bool
        If true (default: False), subtracts off dist.min() to eliminate negative distances
        (Could be improved/expanded)
    symetric : bool
        If True (default), then it assumes dist(A,B) == dist(B,A) and speeds up computation.
    metric : function with params seq1, seq2 and possibly additional kwargs
        Function will be called to compute each pairwise distance.
    convertToNP : bool (default: False)
        If True then strings are converted to np.arrays for speed,
        but metric will also need to accomodate the arrays as opposed to strings
    kwargs : additional keyword arguments
        Will be passed to each call of metric()

    Returns
    -------
    dist : ndarray of shape [len(row_seqs), len(col_seqs)]
        Contains all pairwise distances for seqs.
    """
    if not 'normed' in kwargs.keys():
        kwargs['normed'] = False
    if metric is None:
        metric = seq_distance

    """Only compute distances on unique sequences. De-uniquify with inv_uniqi later"""
    row_uSeqs,row_uniqi,row_inv_uniqi = unique(row_seqs,return_index=True,return_inverse=True)
    col_uSeqs,col_uniqi,col_inv_uniqi = unique(col_seqs,return_index=True,return_inverse=True)

    if convertToNP:
        R = [seq2vec(s) for s in row_uSeqs]
        C = [seq2vec(s) for s in col_uSeqs]
    else:
        R = row_uSeqs
        C = col_uSeqs

    dist = zeros((len(row_uSeqs),len(col_uSeqs)))
    for i,j in itertools.product(range(len(row_uSeqs)),range(len(col_uSeqs))):
        if not symetric:
            """If not assumed symetric, compute all distances"""
            dist[i,j] = metric(R[i],C[j],**kwargs)
        else:
            if j<i:
                tmp = metric(R[i],C[j],**kwargs)
                dist[i,j] = tmp
                dist[j,i] = tmp
            elif j>i:
                pass
            elif j==i:
                dist[i,j] = metric(R[i],C[j],**kwargs)

    if normalize:
        dist = dist-dist.min()
    """De-uniquify such that dist is now shape [len(seqs), len(seqs)]"""
    dist = dist[row_inv_uniqi,:][:,col_inv_uniqi]
    return dist

def embedDistanceMatrix(dist,method='tsne'):
    """MDS embedding of sequence distances in dist, returning Nx2 x,y-coords: tsne, isomap, pca, mds, kpca"""
    if method=='tsne':
        xy=tsne.run_tsne(dist,no_dims=2)
        #xy=pytsne.run_tsne(adist,no_dims=2)
    elif method=='isomap':
        isoObj=Isomap(n_neighbors=10,n_components=2)
        xy=isoObj.fit_transform(dist)
    elif method=='mds':
        mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=15,
                           dissimilarity="precomputed", n_jobs=1)
        xy = mds.fit(dist).embedding_
        rot = PCA(n_components=2)
        xy = rot.fit_transform(xy)
    elif method=='pca':
        pcaObj = PCA(n_components=2)
        xy = pcaObj.fit_transform(1-dist)
    elif method=='kpca':
        pcaObj = KernelPCA(n_components=2,kernel='precomputed')
        xy = pcaObj.fit_transform(1-dist)
    elif method=='lle':
        lle = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2,method='standard')
        xy = lle.fit_transform(dist)
    return xy