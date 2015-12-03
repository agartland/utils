#!/usr/bin/env python
from Bio import SeqIO
import argparse
import re

BADAA='-B#X*?'

def main():
    """Read in a FASTA file and spit out a txt file with one unique 9-mer per line.
    Can be imported as a function or run as a script with arguments."""

    #parse the command line arguments
    parser = argparse.ArgumentParser(prog='seq2nmers',description='Open a FASTA file and produce a list of all unique 9-mers, one per line in an output file.')
    parser.add_argument('fasta_filename', type=str, help='FASTA file with sequences')
    parser.add_argument('out_filename', type=str, help='output file')
    parser.add_argument('-L', metavar='peptide_length', default='9', type=int, help='length of each peptide (default: 9)')
    
    args = parser.parse_args()
    
    print 'Processing sequences',
    allMers = seq2nmers(args.fasta_filename,args.L)
    print 'done'
    print 'Filtering out bad mers...',
    allMers=filter(lambda s: not re.search('[%s]' % BADAA,s),allMers)
    print 'done\nWriting mer file...',
    
    with open(args.out_filename,'w') as f:
        for s in sorted(allMers):
            f.write('%s\n' % s)
    print 'done'
    
def seq2nmers(filename,nmer=9):
    """Takes a fasta file and chops it up into a list of unique kmers"""
    inputFile=SeqIO.parse(filename,'fasta')
    allMers=set()
    for i,record in enumerate(inputFile):
        if i % 20 == 0:
            print '.',
        """Get rid of gaps since those will never be useful mers"""
        allMers.update(set(aa2nmers(str(record.seq).replace('-',''),nmer)))
    return list(allMers)

def aa2nmers(seq,nmer=9):
    """Takes an AA seq and generates a list of the kmers"""
    return [seq[i:i+nmer] for i in range(len(seq)-nmer+1)]

if __name__ == '__main__':
    main()
