import pandas as pd
import tempfile
import skbio
from skbio.sequence import Sequence
import os
import subprocess
import numpy as np
import re

__all__ = ['nwalign',
           'align2skbio',
           'skbio2align']

def align2skbio(align):
    return skbio.TabularMSA([Sequence(s, metadata=dict(id=str(i))) for i,s in align.iteritems()])

def skbio2align(seqColl):
    return pd.Series({s.metadata['id']:''.join(s.values) for s in seqColl})

def align2fasta(align, fn):
    with open(fn, 'w') as fh:
        for i in range(align.shape[0]):
            fh.write('>%s\n' % str(align.index[i]))
            fh.write('%s\n' % str(align.iloc[i]))

def needleall(seqsA, seqsB=None, gop=5, gep=2):
    """Use EMBOSS needleall to globally align 
    all pairs of AA sequences in seqA with seqB
    (or all seqs in seqA to eachother).

    needleall -auto -asequence a.fasta -bsequence b.fasta -gapopen 5 -gapextend 2 -outfile results.needleall
    
    Parameters
    ----------
    seqsA, seqsB : list or pd.Series
    gop,gep : int
        Gap open and extend penalties (positive integers)

    Return
    ------
    pwsim : pd.DataFrame()
        Matrix of similarity scores.
    resDf : pd.DataFrame()
        Frame of more detailed results."""

    """Create temporary files"""
    inAFn = tempfile.mktemp(prefix='tmp_seqsA', suffix='.fasta', dir='.')
    if not seqsB is None:
        inBFn = tempfile.mktemp(prefix='tmp_seqsB', suffix='.fasta', dir='.')
    else:
        inBFn = inAFn
    outFn = tempfile.mktemp(prefix='tmp_results', suffix='.needleall', dir='.')
        
    if not type(seqsA) is pd.Series:
        alignA = pd.Series(seqsA)
    else:
        alignA = seqsA

    if not seqsB is None and not type(seqsB) is pd.Series:
        alignB = pd.Series(seqsB)
    elif not seqsB is None:
        alignB = seqsB
        
    """Put sequences in the tempfiles"""
    align2fasta(alignA, inAFn)
    if not seqsB is None:
        align2fasta(alignB, inBFn)

    command = ['needleall',
               '-auto',
               '-asequence', inAFn,
               '-bsequence', inBFn,
               '-gapopen', str(gop),
               '-gapextend', str(gep),
               '-aformat', 'markx10',
               '-outfile', outFn]

    result = subprocess.call(command, startupinfo=None)

    """If EMBOSS was successful"""
    if not result:
        #scores = pd.read_csv(outFn, sep=' ')
        #print scores
        with open(outFn, 'r') as fh:
            s = fh.read()
        resDf = _parseEMBOSSmarkx10(s)
    else:
        print "Error in EMBOSS needleall!"
        raise Exception("EMBOSSError")
    
    """Remove the temporary files"""
    os.remove(inAFn)
    if not seqsB is None:
        os.remove(inBFn)
    os.remove(outFn)
        
    """Check that all scores are there"""
    if not seqsB is None:
        newShape = (alignB.shape[0], alignA.shape[0])
        index = alignB.values
        columns = alignA.values
    else:
        newShape = (alignA.shape[0], alignA.shape[0])
        index = alignA.values
        columns = alignA.values
    
    pwsim = pd.DataFrame(resDf['Score'].values.reshape(newShape),
                         index=index,
                         columns=columns)

    return pwsim, resDf

def _parseEMBOSSmarkx10(s):
    allSeqs = np.array(re.findall(r'^([\w-]+)', s, flags=re.MULTILINE))

    gseqA = allSeqs[range(0, len(allSeqs), 2)]
    gseqB = allSeqs[range(1, len(allSeqs), 2)]
    nameA = re.findall('# 1: (.+)', s)
    nameB = re.findall('# 2: (.+)', s)
    length = np.array(re.findall('# Length: (.+)', s), dtype=int)
    ident = np.array(re.findall('# Identity: (.+)/', s), dtype=int)
    sim = np.array(re.findall('# Similarity: (.+)/', s), dtype=int)
    gaps = np.array(re.findall('# Gaps: (.+)/', s), dtype=int)
    score = np.array(re.findall('# Score: (.+)', s), dtype=float)

    df = pd.DataFrame(dict(AlgnSeqA=gseqA,
                           AlgnSeqB=gseqB,
                           NameA=nameA,
                           NameB=nameB,
                           Length=length,
                           Identity=ident,
                           Similarity=sim,
                           Gaps=gaps,
                           Score=score))
    return df



