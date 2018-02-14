import pandas as pd
import tempfile
import skbio
from skbio.sequence import Sequence
import os
import subprocess
import platform
import numpy as np

if platform.system() == 'Windows':
    import _subprocess  # @bug with python 2.7 ?
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= _subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = _subprocess.SW_HIDE
else:
    startupinfo = None

__all__ = ['muscle_align',
           'align2skbio',
           'skbio2align']

def align2skbio(align):
    return skbio.TabularMSA([Sequence(s, metadata=dict(id=str(i))) for i, s in align.items()])

def skbio2align(seqColl):
    return pd.Series({s.metadata['id']:''.join([c.decode('utf-8') for c in s.values]) for s in seqColl})

def align2fasta(align, fn):
    with open(fn, 'w') as fh:
        for i in range(align.shape[0]):
            fh.write('>%s\n' % str(align.index[i]))
            fh.write('%s\n' % str(align.iloc[i]))

def muscle_align(seqs):
    """Use MUSCLE to align AA seqs.

    muscle -in new_seqs.fa -out new_seqs.afa
    
    Parameters
    ----------
    seqs : list or pd.Series

    Return
    ------
    align : pd.Series()
        Aligned sequences."""

    """Create temporary file for MUSCLE"""
    inFn = tempfile.mktemp(prefix='tmp_align', suffix='.fasta', dir=None)
    outFn = tempfile.mktemp(prefix='tmp_align', suffix='.fasta', dir=None)
        
    if not isinstance(seqs, pd.Series):
        align = pd.Series(seqs)
    else:
        align = seqs
        
    newIndex = np.arange(align.shape[0])
    oldIndex = align.index

    align.index = newIndex

    """Put alignment in the tempfiles"""
    align2fasta(align, inFn)
    # align2skbio(align).write(inFn, format='fasta')
    # skbio.write(obj=, into=inFn, format='fasta')

    muscleCommand = ['muscle',
                     '-in', inFn,
                     '-out', outFn]
    
    result = subprocess.call(muscleCommand, startupinfo=startupinfo)

    """If MUSCLE was successful"""
    if not result:
        outAlign = skbio2align(skbio.read(outFn, format='fasta'))
        # outAign = skbio2align(skbio.TabularMSA.read(outFn, format='fasta'))
    else:
        print("Error in MUSCLE!")
        raise Exception("MUSCLEError")
    
    """Remove the temporary files"""
    os.remove(inFn)
    os.remove(outFn)
        
    """MUSCLE seqs need to be reorderd using the original index"""
    outAlign = outAlign.loc[[str(i) for i in align.index]]
    
    """Index was str() through FASTA files so reset index with original index"""
    outAlign.index = oldIndex
    
    """Check that all seqs are being returned in the correct order"""
    badSeqs = 0
    if not len(seqs) == len(outAlign):
        print('Different number of output seqs!')
        badSeqs += 1

    for i, s1, s2 in zip(list(range(len(seqs))), seqs, outAlign):
        if not s1.replace('-', '') == s2.replace('-', ''):
            print('%d: %s != %s' % (i, s1, s2))
            badSeqs += 1
    if badSeqs > 0:
        raise Exception('Output seqs are different than input seqs! (%d)' % badSeqs)

    return outAlign