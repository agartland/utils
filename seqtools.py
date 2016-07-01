from __future__ import division
from functools import *
import pandas as pd
import itertools
from Bio import SeqIO, pairwise2, Phylo
import dendropy
from dendropy import treecalc
from copy import deepcopy
from HLAPredCache import *
import subprocess
import tempfile
import os
from aacolors import hydrophobicity, chemistry, taylor
from pylab import *
from corrplot import fdrAdjust
from utilHelpers import mutual_information
from sympy import binomial
import sys
import numpy as np
import re

from objhist import objhist

from seqdistance import hamming_distance, seq_distance
from seqdistance.matrices import addGapScores, binarySubst, nanGapScores

"""Utility functions that I sometimes depend on for sequence analysis. Most are old dependencies.
If you can't find something, it may still be in the SVN repo 'scripts/util/seqtools_old.py' file."""

__all__ = ['BADAA',
           'AALPHABET',
           'AA2CODE',
           'CODE2AA',
           'isvalidpeptide',
           'cleanAlign',
           'cleanDf',
           'removeBadAA',
           'padAlignment',
           'consensus',
           'identifyMindist',
           'peptideSetCoverage',
           'fasta2df',
           'df2fasta',
           'align2fasta',
           'align2mers',
           'align2mers_tracked',
           'fasta2align',
           'sliceAlign',
           'kmerSlice',
           'alignmentEntropy',
           'generateAlignment',
           'fasta2seqs',
           'seqs2fasta',
           'catAlignments',
           'mynorm',
           'aalogoheights',
           'computeAlignmentLogoHeights',
           'pairwiseDiversity',
           'pairwiseDiversityInterGroup',
           '_PD',
           '_PD_hamming',
           'pairwiseMutualInformation',
           'seqmat2align',
           'align2mat',
           'align2aamat',
           'condenseGappyAlignment',
           'nx2sif',
           'kmerConsensus',
           'pepComp',
           'tree2pwdist',
           'overlappingKmers']


BADAA = '-*BX#Z'
AALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
AA2CODE = {aa:i for i,aa in enumerate(AALPHABET)}
AA2CODE.update({'-':21})
CODE2AA = {i:aa for i,aa in enumerate(AALPHABET)}
CODE2AA.update({21:'-'})

def isvalidpeptide(mer,badaa=None):
    """Test if the mer contains an BAD amino acids in global BADAA
    typically -*BX#Z"""
    if badaa is None:
        badaa=BADAA
    if not mer is None:
        return not re.search('[%s]' % badaa,mer)
    else:
        return False
def cleanAlign(align,badaa=None):
    """Remove all invalid sequences (containing badaa) from
    the alignment
    badaa is '-*BX#Z' by default"""
    return align.loc[[isvalidpeptide(s,badaa) for s in align]]
def cleanDf(df,badaa=None):
    """Remove all invalid sequences (containing badaa) from
    the alignment
    badaa is '-*BX#Z' by default"""
    return df.loc[[isvalidpeptide(s,badaa) for s in df.seq]]
def removeBadAA(mer,badaa=None):
    """Remove badaa amino acids from the mer, default badaa is -*BX#Z"""
    if badaa is None:
        badaa=BADAA
    if not mer is None:
        return re.sub('[%s]' % badaa,'',mer)
    else:
        return mer
def _seq2vec(seq):
    """Convert AA sequence into numpy vector of integers for fast comparison"""
    vec = zeros(len(seq),dtype=int)
    for aai,aa in enumerate(seq):
        vec[aai] = AA2CODE[aa]
    return vec

def padAlignment(align, applyPadding=True):
    """Given an iterator of sequences, convert to pd.Series
    Remove * or # from the end and pad sequences of different length with gaps
    There is a warning if gaps are used for padding

    Returns the align obj as pd.Series"""
    if type(align) in [dict, np.ndarray, list]:
        align = pd.Series(align)

    """Replace *  and #  with - and - """
    for ind in align.index:
        if '*' in align[ind]:
            align[ind] = align[ind].replace('*','-')
        if '#' in align[ind]:
            align[ind] = align[ind].replace('#','-')
    """Pad with gaps if the lengths are all the same"""
    if applyPadding:
        L = align.map(len).unique()
        if len(L) > 1:
            #print 'Sequences have different lengths (pading with gaps): %s' % L
            L = L.max()
            for ind in align.index:
                if len(align[ind]) < L:
                    align[ind] = align[ind].ljust(L,'-')
        else:
            L = L.max()
    return align

def consensus(align, ignoreGaps = True):
    """Return a consensus sequence from the sequences in seqs
    seqs can be a dict or a pd.Series of sequence strings

    ignoresGaps unless all AA are gaps"""
    align = padAlignment(align)
    L = len(align[align.index[0]])

    cons = ''
    for aai in arange(L):
        counts = objhist([seq[aai] for seq in align])
        if ignoreGaps and len(counts)>1:
            droppedGaps = counts.pop('-',0)
        cons += max(counts.keys(), key=counts.get)
    return cons

def identifyMindist(align, ignoreGaps=True):
    """Compute a consensus sequence and return the sequence
    in the alignment with the smallest (hamming) distance

    Parameters
    ----------
    align : list or pd.Series
        Sequence alignment.
    ignoreGaps : bool
        Passed to consensus, specifies whether gap
        characters are ignored for computing consensus.

    Returns
    -------
    seq : str
        One of the sequences in align."""
    align = padAlignment(align)
    cons = consensus(align, ignoreGaps)
    dist = align.map(partial(hamming_distance, cons))
    return align[dist.argmin()]


def peptideSetCoverage(peptides1,peptides2,mmTolerance=1):
    """Returns a dict that reports the fraction of peptides in peptides2 that are covered
    by each peptide in peptides1 by matching within a tolerance of mmTolerance.
    Key 'tot' indicates the cumulative coverage that peptides1 provide of peptides2 as a fraction of peptides2
    
    Call with unique(peptides2) if Q is the fraction of unique peptides or not if Q is fraction of representative population.
    Note: Can be used as a non-symetric distance between two peptide sets"""

    oh1=objhist(peptides1)
    coveredPeps={k:[] for k in set(oh1.keys())}
    coveredPeps.update({'tot':[]})
    cache={}
    for pep2 in peptides2:
        anyCover=False
        for pep1 in set(peptides1):
            try:
                dist = cache[(pep1,pep2)]
            except KeyError:
                dist = seq_distance(pep1,pep2,subst=binarySubst,normed=False)
                cache.update({(pep1,pep2):dist,(pep2,pep1):dist})
            if dist<=mmTolerance:
                coveredPeps[pep1].append(pep2)
                anyCover=True
        if anyCover:
            coveredPeps['tot'].append(pep2)
    coverage={k:len(v)/len(peptides2) for k,v in coveredPeps.items()}
    return coverage


def fasta2seqs(fn):
    return [str(r.seq) for r in SeqIO.parse(open(fn,'r'),'fasta')]
def seqs2fasta(seqs,fn):
    with open(fn,'w') as fh:
        for i,s in enumerate(seqs):
            fh.write('>seq%d\n' % i)
            fh.write('%s\n' % s)
def fasta2df(fn,sep='.',columns=['clade','country','year','name','seqid'],index=None,uniqueIndex=True):
    """Read in a fasta file and turn it  into a Pandas DataFrame

    Defaults parse the HIV LANL fasta alignments.

    Parameters
    ----------
    sep : str
        Separator in the description field.
    columns : list
       List of the sep delimited column names in-order.
    index : str
       Column to use as the DataFrame index (default: None)

    Returns
    -------
    seqDf : pd.DataFrame
        All sequences from the fasta file with a seq column containing the sequences."""
    
    with open(fn,'r') as fh:
        records = SeqIO.parse(fh,'fasta')
        sDict = {'seq':[]}
        sDict.update({k:[] for k in columns})
        for r in records:
            sDict['seq'].append(str(r.seq))

            info = r.description.split(sep)
            for i in arange(len(columns)):
                if i < len(info):
                    sDict[columns[i]].append(info[i])
                else:
                    sDict[columns[i]].append('')

    seqDf = pd.DataFrame(sDict)
    if not index is None:
        if seqDf.shape[0]==seqDf[index].unique().shape[0] or not uniqueIndex:
            """If the index is unique fine, otherwise make a unique index by appending _%d"""
            seqDf = seqDf.set_index(index)
        else:
            tmp = seqDf[index].copy()
            for i,ind in enumerate(tmp.index):
                tmp[ind] = '%d_%s' % (i,tmp[ind])
            seqDf = seqDf.set_index(tmp)
    return seqDf
def df2fasta(df,fn,sep='.',columns=None):
    """Writes the Df from fasta2df back to a FASTA file"""
    if columns is None:
        columns=list(df.columns)
    if 'seq' in columns:
        columns.remove('seq')
    with open(fn,'w') as fh:
        for ind,row in df.iterrows():
            label='>%s' % ind
            for col in columns:
                label+='%s%s' % (sep,row[col])
            fh.write('%s\n' % label)
            fh.write('%s\n' % row['seq'])

def align2fasta(align, fn, applyPadding = True):
    """Write align to a FASTA file where align is a dict or pd.Series of sequences"""
    align = padAlignment(align, applyPadding)

    with open(fn,'w') as fh:
        for i in arange(align.shape[0]):
            ind = align.index[i]
            fh.write('>%s\n' % ind)
            fh.write('%s\n' % align.iloc[i])
def align2mers(align,fn=None,nmers=[9]):
    """Compute all nmers in align and write to a mers file for prediction"""
    align=padAlignment(align)
    mers=[]
    for seq in align:
        mers.extend(getMers(re.sub('[%s]' % BADAA,'',seq),nmers))
    mers=set(mers)
    if not fn is None:
        with open(fn,'w') as fh:
            for pep in mers:
                fh.write('%s\n' % pep)
    else:
        return list(mers)
def align2mers_tracked(align,nmers=[9],firstOnly=True):
    """Return a df of all nmers in the alignment along with start position and seq index"""
    align = padAlignment(align)
    cols = ['peptide','starti','seqi','L','count']
    outD = {k:[] for k in cols}
    for k in nmers:
        for seqi,seq in enumerate(align):
            for starti in range(len(seq)-k+1):
                mer = grabKmer(seq,starti,k)[1]
                if not mer is None:
                    if not firstOnly or not mer in outD['peptide']:
                        outD['peptide'].append(mer)
                        outD['starti'].append(starti)
                        outD['seqi'].append(align.index[seqi])
                        outD['L'].append(k)
                        outD['count'].append(1)
                    else:
                        ind = outD['peptide'].index(mer)
                        outD['count'][ind] += 1
    return pd.DataFrame(outD)[cols]

def fasta2align(fn,uniqueIndex=True):
    """Read sequences from a FASTA file and store in a pd.Series object indexed by the description"""
    return fasta2df(fn,sep=None,columns=['name'],index='name',uniqueIndex=uniqueIndex).seq

def sliceAlign(align,region,sites=False):
    """Return a region of the alignment where region is (start, end)
    OR if sites is True then include all sites in region (not range)"""
    if region is None:
        return align
    elif sites:
        return align.map(lambda seq: ''.join([seq[r] for r in region]))
    else:
        return align.map(lambda seq: seq[region[0]:region[1]])

def kmerSlice(align,starti,k,gapped=True):
    """Return a slice of an alignment specified by kmer start position.
    Uses grabKmer to return "gapped" or "non-gapped" kmers.

    Note: Using non-gapped slices can return None when kmer begins with a gap
          or if is near the end and there are insufficient non-gap characters"""

    if gapped:
        grabKmerFlag = 0
    else:
        grabKmerFlag = 1
    return align.map(lambda s: grabKmer(s,starti,k)[grabKmerFlag])

def alignmentEntropy(align, statistic = 'absolute', removeGaps = False, k = 1, logFunc = np.log):
    """Calculates the entropy in bits of each site (or kmer) in a sequence alignment.

    Also can compute:
        - "uniqueness" which I define to be the fraction of unique sequences
        - "uniquenum" which is the number of unique sequences
    
    Parameters
    ----------
    align : pd.Series() or list
        Alignment of sequences.
    statistic : str
        Statistic to be computed: absolute, uniqueness
        Uniqueness is the fraction of unique sequences.
        Uniquenum is the number of unique AA at each position.
    removeGaps : bool
        Remove from the alignment at each position, kmers that start with a gap character.
        Also use "non-gapped kmers" (ie skipping gaps)
    k : int
        Length of the kmer to consider at each start position in the alignment.
        (default 1 specifies site-wise entropy)
    logFunc : function
        Default is natural log, returning nats. Can also use log2 for bits.


    Return
    ------
    out : float
        Output statistic."""
    if removeGaps:
        grabKmerFlag = 1
    else:
        grabKmerFlag = 0

    align = padAlignment(align)
    L = len(align[align.index[0]])
    nKmers = L - k + 1

    entropy = zeros(nKmers,dtype=float64)
    for aai in arange(nKmers):
        kmers = [grabKmer(seq,aai,k)[grabKmerFlag] for seq in align]
        """kmers that start with a gap or that are at the end and are of insufficent length, will be None"""
        kmers = [mer for mer in kmers if not mer is None]
        oh = objhist(kmers)
        
        if statistic == 'absolute':
            entropy[aai] = oh.entropy()
        elif statistic == 'uniqueness':
            entropy[aai] = oh.uniqueness()
        elif statistic == 'uniquenum':
            entropy[aai] = len(oh.keys())
    return entropy

def generateAlignment(seqs):
    """Use MUSCLE to align the seqs.

    muscle -in new_seqs.fa -out new_seqs.afa
    
    Parameters
    ----------
    seqs : list

    Return
    ------
    align : pd.Series()
        Aligned sequences.
    """
    """Create temporary file for MUSCLE"""
    inFn=tempfile.mktemp(prefix='tmp_align',suffix='.fasta',dir=None)
    outFn=tempfile.mktemp(prefix='tmp_align',suffix='.fasta',dir=None)
    
    
    """Creates an align object or pd.Series() with indexing to preserve order but does not appyl padding"""
    align = padAlignment(seqs, applyPadding=False)
    """Put alignments in the tempfiles"""
    align2fasta(seqs,inFn,applyPadding=False)

    muscleCommand = ['muscle','-in',inFn,'-out',outFn]
    result = subprocess.call(muscleCommand)

    """If MUSCLE was successful"""
    if not result:
        outAlign = fasta2align(outFn)
    else:
        print "Error in MUSCLE!"
        raise Exception("MUSCLEError")
    """Remove the temporary files"""
    os.remove(inFn)
    os.remove(outFn)
        
    """MUSCLE seqs need to be reorderd using the original index"""
    outAlign = outAlign.loc[[str(i) for i in align.index]]
    """Index was str() through FASTA files so reset index with original index"""
    outAlign.index = align.index
    
    """Check that all seqs are being returned in the correct order"""
    badSeqs = 0
    if not len(seqs) == len(outAlign):
        print 'Different number of output seqs!'
        badSeqs+=1

    for i,s1,s2 in zip(arange(len(seqs)),seqs,outAlign):
        if not s1.replace('-','') == s2.replace('-',''):
            print '%d: %s != %s' % (i,s1,s2)
            badSeqs+=1
    if badSeqs>0:
        raise Exception('Output seqs are different than input seqs! (%d)' % badSeqs)

    return outAlign

def catAlignments(alignA,alignB):
    """
    Take two dict or pd.Series as alignments and combine using MUSCLE
    Return a pd.Series of all aligned sequences indexed by original seq keys
    (keys are suffixed with A or B if neccessary)

    From MUSCLE documentation:
       To align one sequence to an existing alignment:
            muscle -profile -in1 existing_aln.afa -in2 new_seq.fa -out combined.afa

        If you have more than one new sequences, you can align them first then add them, for example:
            muscle -in new_seqs.fa -out new_seqs.afa
            muscle -profile -in1 existing_aln.afa -in2 new_seqs.afa -out combined.afas
    """

    """Create temporary files for MUSCLE to work on the two alignments"""
    aFn=tempfile.mktemp(prefix='tmp_align',suffix='.fasta',dir=None)
    bFn=tempfile.mktemp(prefix='tmp_align',suffix='.fasta',dir=None)
    outFn=tempfile.mktemp(prefix='tmp_align',suffix='.fasta',dir=None)

    
    """Make sure alignments have the same length and are Series objects"""
    alignA=padAlignment(alignA)
    alignB=padAlignment(alignB)

    """Put alignments in the tempfiles"""
    align2fasta(alignA,aFn)
    align2fasta(alignB,bFn)

    muscleCommand=['muscle','-profile','-in1',aFn,'-in2',bFn,'-out',outFn]
    result=subprocess.call(muscleCommand)

    """If MUSCLE was successful"""
    if not result:
        outAlign=fasta2align(outFn)
    else:
        print "Error in MUSCLE!"
        raise Exception("MUSCLEError")
   
    """
    except:
        pass
        os.remove(aFn)
        os.remove(bFn)
        os.remove(outFn)
        raise
    """
    """Remove the temporary files"""
    os.remove(aFn)
    os.remove(bFn)
    os.remove(outFn)

    return outAlign

def mynorm(vec,mx=1,mn=0):
    """Normazlize values of vec in-place to [mn, mx] interval"""
    vec-=nanmin(vec)
    vec=vec/nanmax(vec)
    vec=vec*(mx-mn)+mn
    return vec

def aalogoheights(aahistObj,N=20):
    """For a objhist of AA frequencies, compute the heights
    of each AA for a logo plot"""
    aahistObj=deepcopy(aahistObj)
    keys=aahistObj.keys()
    for aa in BADAA:
        if aa in keys:
            dummy=aahistObj.pop(aa)
    keys=[aa for aa in aahistObj.sortedKeys(reverse=False)]
    freq=aahistObj.freq()
    p=array([freq[k] for k in keys])
    #err=(1/log(2))*((N-1)/(2*aahistObj.sum()))
    #totEntropy=log2(N)-((-p*log2(p)).sum()+err)
    totEntropy=log2(N)-((-p*log2(p)).sum())
    heights=p*totEntropy
    return keys,heights

def computeAlignmentLogoHeights(fullAlign,region=None):
    """Compute heights for a sequence logo plot of relative entropy
    Returns a vector of heights"""
    fullAlign=padAlignment(fullAlign)
    align=sliceAlign(fullAlign,region)
    L=len(align[align.index[0]])
    
    tot=zeros(L)

    for sitei in arange(L):
        aaHist=objhist([seq[sitei] for seq in align])
        aaKeys,entropy=aalogoheights(aaHist)
        tot[sitei]=entropy.sum()
    return tot

def pairwiseDiversity(fullAlign,region=None,subst=None,bySite=True):
    """Calculate sitewise pairwise diversity for an alignment
    By default it will use a "hamming" substitution matrix
    All gap comparisons are nan
    if bySite is False then compute single PD based on whole-sequence distances
    Return the fraction of valid (non-gap) pairwise comparisons at each site that are AA matched"""

    fullAlign=padAlignment(fullAlign)
    align=sliceAlign(fullAlign,region)
    L=len(align[align.index[0]])

    if subst is None:
        _PD_hamming(align,None,subst,bySite,True)
    
    return _PD(align,None,subst,bySite,True)

def pairwiseDiversityInterGroup(align1,align2,region=None,subst=None,bySite=True):
    """Calculate pairwise diversity between two alignments
    By default it will use a "hamming" substitution matrix
    All gap comparisons are nan
    if bySite is False then compute single PD based on whole-sequence distances
    Return the fraction of valid (non-gap) pairwise comparisons at each site that are AA matched"""

    """Does not perform "padding" so alignments must have same sequence lengths"""
    align1=sliceAlign(align1,region)
    align2=sliceAlign(align2,region)
    L=len(align[align.index[0]])

    if subst is None:
        _PD_hamming(align1,align2,subst,bySite,False)
    
    return _PD(align1,align2,subst,bySite,False)

def _PD(alignA,alignB,subst,bySite,withinA):
    """Computation for pairwise diversity"""
    L=len(alignA.iloc[0])

    """Dist will be 1 where equal, 0 where not and nan if one is a gap"""
    if withinA:
        dist=zeros((int(binomial(len(alignA),2)),L))
        allPairs = itertools.combinations(alignA,2)
    else:
        dist=zeros((len(alignA)*len(alignB),L))
        allPairs = itertools.product(alignA,alignB)
    j=0
    for seq1,seq2 in allPairs:
        """This line is the bottleneck. I should try some optimization here. This would help with all distance functions"""
        dist[j,:]=array([i for i in itertools.imap(lambda a,b: subst.get((a,b),subst.get((b,a))), seq1, seq2)])
        j+=1
    
    """Actually, pairwise diversity is a distance, not a similarity so identical AA should be counted as 0"""
    dist=1-dist
    if not bySite:
        dist=nanmean(dist,axis=1)
    return nanmean(dist,axis=0)

def _PD_hamming(alignA,alignB,subst,bySite,withinA,ignoreGaps=True):
    """Computation for pairwise diversity using a vector optimized hamming distance.
    Optionally ignoreGaps treats gap comparisons as Nan"""
    L=len(alignA.iloc[0])
    gapCode = AA2CODE['-']

    """Convert alignments into integer arrays first to speed comparisons"""
    matA = zeros((len(alignA),L))
    for seqi,s in enumerate(alignA):
        matA[seqi,:] = _seq2vec(s)
    if not withinA:
        matB = zeros((len(alignB),L))
        for seqi,s in enumerate(alignB):
            matB[seqi,:] = _seq2vec(s)

    """Dist will be 1 where equal, 0 where not and nan if one is a gap"""
    if withinA:
        dist=zeros((int(binomial(len(alignA),2)),L))
        allPairs = itertools.combinations(arange(len(alignA)),2)
        for j,(seqi1,seqi2) in enumerate(allPairs):
            dist[j,:] = matA[seqi1,:]!=matA[seqi2,:]
            if ignoreGaps:
                gapInd = (matA[seqi1,:]==gapCode) | (matA[seqi2,:]==gapCode)
                dist[j,gapInd] = nan
    else:
        dist=zeros((len(alignA)*len(alignB),L))
        allPairs = itertools.product(arange(len(alignA)),arange(len(alignB)))
        for j,(seqiA,seqiB) in enumerate(allPairs):
            dist[j,:] = matA[seqiA,:]!=matB[seqiB,:]
            if ignoreGaps:
                gapInd = (matA[seqiA,:]==gapCode) | (matB[seqiB,:]==gapCode)
                dist[j,gapInd] = nan

    if not bySite:
        dist=nanmean(dist,axis=1)
    return nanmean(dist,axis=0)

def pairwiseMutualInformation(align,nperms=1e4):
    """Compute the pairwise mutual information of all sites in the alignment
    Return matrix of M and p-values"""
    L=len(align[align.index[0]])
    columns=[align.map(lambda s: s[i]) for i in arange(L)]
    M=nan*zeros((L,L))
    p=nan*zeros((L,L))
    Mstar=nan*zeros((L,L))
    for xi,yi in itertools.combinations(arange(L),2):
        freqx=objhist(columns[xi])
        freqy=objhist(columns[yi])

        tmpM,tmpMstar,tmpp,Hx,Hy,Hxy=mutual_information(columns[xi],columns[yi],logfunc=log2,nperms=nperms)
       
        """We wouldn't need to test invariant sites or a site with itself"""
        if len(freqx)==1 or len(freqy)==1:
            tmpp=nan
        elif xi==yi:
            tmpp=nan

        M[xi,yi]=tmpM
        p[xi,yi]=tmpp
        Mstar[xi,yi]=tmpMstar
    q=fdrAdjust(p)

    return M,Mstar,p,q

def seqmat2align(smat,index=None):
    """Convert from an array of dtype=S1 to alignment"""
    if index is None:
        index = np.arange(smat.shape[0])
    return pd.Series([''.join(smat[seqi,:]) for seqi in np.arange(smat.shape[0])], name='seq', index=index)

def align2mat(align, k=1, gapped=True):
    """Convert an alignment into a 2d numpy array of kmers [nSeqs x nSites/nKmers]
    If gapped is True, returns kmers with gaps included.
    If gapped is False, returns "non-gapped" kmers and each kmer starting with a gap is '-'*k 
    See grabKmer() for definition of non-gapped kmer."""
    tmp = padAlignment(align)
    L = len(tmp.iloc[0])
    Nkmers = L-k+1

    if gapped:
        """Slightly faster, but not as flexible"""
        out = np.array([[s[i:i+k] for i in range(Nkmers)] for s in tmp], dtype='S%d' % k)
    else:
        out = np.empty((L,Nkmers), dtype='S%d' % k)
        for seqi,seq in enumerate(tmp):
            for starti in range(Nkmers):
                #out[seqi,starti] = seq[starti:starti+k]
                full, ng = grabKmer(seq, starti, k=k)
                if ng is None:
                    ng = '-'*k
                out[seqi,starti] = ng
    return out

def align2aamat(align):
    """Convert an alignment into a 3d boolean numpy array [nSeqs x nSites x nAAs]"""
    for seq in align:
        L = len(seq)
        break
    aaMat = align2mat(align)
    aaFeat = np.zeros((len(align), L, len(AALPHABET)))
    for seqi,sitei in itertools.product(xrange(aaFeat.shape[0]), range(aaFeat.shape[1])):
        try:
            aai = AALPHABET.index(aaMat[seqi,sitei])
            aaFeat[seqi,sitei,aai] = 1.
        except ValueError:
            """If AA is not in AALPHABET then it is ignored"""
            continue
    return aaFeat

def condenseGappyAlignment(a, thresh=0.9):
    """Find sites with more than thresh percent gaps.
    Then remove any sequences with non-gaps at these sites
    and remove the sites from the alignment."""

    a = padAlignment(a)
    smat = align2mat(a)
    gapSiteInd = mean(smat == '-', axis=0) >= thresh
    keepSeqInd = np.all(smat[:,gapSiteInd] == '-', axis=1)
    print 'Removing %d of %d sites and %d of %d sequences from the alignment.' % (gapSiteInd.sum(),smat.shape[1],(~keepSeqInd).sum(),smat.shape[0])

    smat = smat[keepSeqInd,:]
    smat = smat[:,~gapSiteInd]
    
    return seqmat2align(smat, index=a.index[keepSeqInd])
def nx2sif(fn,g):
    """Write Networkx Graph() to SIF file for BioFabric or Cytoscape visualization"""
    with open(fn,'w') as fh:
        for e in g.edges_iter():
            fh.write('%s pp %s\n' % (e[0],e[1]))
def generateSequences(a,N=1,useFreqs=True):
    """Generate new sequences based on those in alignment a
    The AA at each position are chosen independently from the
    observed AAs and may or may not be chosen based on their frequency.
    If useFreqs is True then returns exactly N sequences that are not neccessarily unique.
    Else returns N unique sequences or as many as possible, printing an error if actualN < N"""
    
    a = padAlignment(a)
    L = len(a.iloc[0])

    if useFreqs:
        smat = empty((N,L),dtype='S1')
        for i in arange(L):
            oh = objhist(sliceAlign(a,(i,i+1)))
            smat[:,i] = oh.generateRandomSequence(N,useFreqs=True)
    else:
        chunkN = int(ceil(N/10))
        smat = None
        counter = 0
        actualN = 0
        while actualN < N and counter < N*100:
            tmpmat = empty((chunkN,L),dtype='S1')
            for i in arange(L):
                oh = objhist(sliceAlign(a,(i,i+1)))
                tmpmat[:,i] = oh.generateRandomSequence(chunkN,useFreqs=False)
            if smat is None:
                smat = tmpmat
            else:
                smat = concatenate((smat,tmpmat),axis=0)
            smat = unique_rows(smat)
            actualN = smat.shape[0]
            counter += 1

        outAlign = seqmat2align(smat[:actualN,:])
        if actualN<N:
            print "Could not create N = %d unique sequences with %d attempts" % (N,counter*10)
            smat = smat[:actualN,:]
    outAlign = seqmat2align(smat)
    return outAlign

def kmerConsensus(align,k=9,verbose=False):
    """From an alignment of sequences create a k-mer consensus sequence
    by identifying the most common whole k-mer at each start position
    and using those residues as the consensus. This will result in more
    than one consensus amino acid at many sites.

    [What is the object that is returned?]

    Parameters
    ----------
    align : list or pd.Series
        Alignment of amino acid sequences all with the same length.
    k : int
        Width of the kmer window.

    Returns
    -------
    con : str
        Consensus sequence taking the mode at each position
    full : list of dicts, len(full) == len(con)
        Each element of the list is a position in the alignment.
        Each dict contains keys/values of the consesnus residues
        and their number at each position.

    Example
    -------
    >>> seqs = ['ABCDE',
                'ABCDE',
                'ABCDE',
                'ABCDE',
                'ABCIE',
                'ABCIE',
                'ABFIE',
                'ABFIE',
                'ABFIE',
                'ABFIE']
    >>> kcon,full = kmerConsensus(seqs,k=3,verbose=True)
    
    ABC
     BCD
      CDE
    Seq1: true consensus
    Seq2: 3mer consensus
    Pos 1 - 5
    A B C I E
          |  
    A B C D E

    Seq1 (5) and Seq2 (5) are 80.0% similar

    >>> print full
    
    [{'A': 1}, {'B': 2}, {'C': 3}, {'D': 2}, {'E': 1}]

    """

    align = padAlignment(align)
    L = len(align.iloc[0])
    Nkmers = L-k+1

    """Get a 2D array of alignment [nSeqs x nSites]"""
    mat = align2mat(align)

    full = [dict() for i in arange(L)]
    for starti in arange(Nkmers):
        """Create a temporary alignment of the ith kmer"""
        tmpA = seqmat2align(mat[:,starti:starti+k])
        """Pick off the most common kmer at that start position"""
        top1 = objhist(tmpA).topN(n=2)[0][0]
        if verbose:
            print ' '*starti + top1
            #print ' '*starti + objhist(tmpA).topN(n=2)[1][0]
        """Add each AA in the most frequent kmer to the consensus"""
        for j,startj in enumerate(arange(starti,starti+k)):
            try:
                full[startj][top1[j]] += 1
            except KeyError:
                full[startj][top1[j]] = 1
    """Consensus is the mode AA at each position in full"""
    con = ''.join([max(pos.keys(),key=pos.get) for pos in full])
    if verbose:
        print 'Seq1: true consensus'
        print 'Seq2: %dmer consensus' % k
        compSeq(consensus(align),con)
    return con, full
def pepComp(align,useConsensus=True):
    """Return align with mix of upper and lower case
    AA residues depending on whether they match or 
    mismatch the consensus or mindist sequence."""
    if useConsensus:
        ref = consensus(align)
    else:
        ref = identifyMindist(align)
    out = []
    for seq in align:
        out.append(''.join([aa.upper() if aa.upper()==refaa.upper() else aa.lower() for aa,refaa in zip(seq,ref)]))
    return out

def tree2pwdist(tree):
    """Compute pairwise distances between every leaf on the phylogenetic tree.

    Can use either a Bio.Phylo object or a dendropy.Tree object (much faster).

    Parameters
    ----------
    tree : obj
        A phylogenetic tree object.

    Returns
    -------
    pwdist : pd.DataFrame
        Symmetric table of all pairwise distances with node labels as columns and index."""
    if type(tree) == type(Phylo.BaseTree.Tree()):
        N = len(tree.get_terminals())
        names = [node.name for node in tree.get_terminals()]
        pwdist = zeros((N,N))
        for i,node1 in enumerate(tree.get_terminals()):
            for j,node2 in enumerate(tree.get_terminals()):
                """Compute half of these and assume symmetry"""
                if i==j:
                    pwdist[i,j] = 0
                elif i<j:
                    pwdist[i,j] = tree.distance(node1,node2)
                    pwdist[j,i] = pwdist[i,j]
    elif type(tree) == type(dendropy.Tree()):
        pdm = dendropy.treecalc.PatristicDistanceMatrix(tree)
        taxon_set = [n.taxon for n in tree.leaf_nodes()]
        N = len(taxon_set)
        names = [taxa.label for taxa in taxon_set]
        pwdist = zeros((N,N))
        for i,t1 in enumerate(taxon_set):
            for j,t2 in enumerate(taxon_set):
                """Compute half of these and assume symmetry"""
                if i==j:
                    pwdist[i,j] = 0
                elif i<j:
                    pwdist[i,j] = pdm(t1,t2)
                    pwdist[j,i] = pwdist[i,j]
    else:
        print 'Tree type does not match Phylo.BaseTree.Tree or dendropy.Tree'
        return
    return pd.DataFrame(pwdist,index = names, columns = names)

def overlappingKmers(s, k=15, overlap=11, includeFinalPeptide=True, returnStartInds=False):
    """Create a list of overlapping kmers from a single sequence

    Params
    ------
    s : sequence (sliceable object)
    k : int
        Length of each mer
    overlap : int
        Overlap between each consecutive kmer
    includeFinalPeptide : bool
        If True, include a peptide of length k that covers the end of the sequence.
    returnStartInds : bool
        If True, return start indices for each peptide.

    Returns
    -------
    mers : list of kmers
    inds : list of indices (optional)"""
    inds = [i for i in range(0, len(s), k-overlap) if i+k < len(s)]

    if includeFinalPeptide and not s[-k:] == s[inds[-1]:inds[-1]+k]:
        inds.append(len(s)-k)

    mers = [s[i:i+k] for i in inds]

    if returnStartInds:
        return mers, inds
    else:
        return mers

def compSeq(s1, s2, lineL=50):
    """Print two sequences showing mismatches.

    Parameters
    ----------
    s1, s2 : str
        Strings representing aligned AA or NT sequences
    lineL : int
        Wrap line at lineL"""
    lineN = int(np.ceil(min(len(s1), len(s2))/lineL))
    count = 0
    samecount = 0
    outStr = ''
    for linei in range(lineN):
        if (linei+1) * lineL < min(len(s1), len(s2)):
            end = (linei+1) * lineL
        else:
            end = min(len(s1), len(s2))
        outStr += 'Pos %d - %d\n' % (linei*lineL+1, end-1+1)
        for sitei in range(linei*lineL, end):
            outStr += s1[sitei]
        outStr += '\n'
        for sitei in range(linei*lineL, end):
            out = ' ' if s1[sitei] == s2[sitei] else '|'
            outStr += out
            count += 1
            samecount += 1 if s1[sitei]==s2[sitei] else 0
        outStr += '\n'
        for sitei in range(linei*lineL, end):
            out = '.' if s1[sitei] == s2[sitei] else s2[sitei]
            outStr += s2[sitei]
        outStr += '\n\n'
    outStr += 'Seq1 (%d) and Seq2 (%d) are %1.1f%% similar\n\n' % (len(s1),len(s2),1e2*samecount/count)
    print outStr
