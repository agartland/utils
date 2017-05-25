import pandas as pd
import argparse
import re
import skbio
from copy import deepcopy
import skbio
from skbio.alignment import local_pairwise_align_ssw, make_identity_substitution_matrix
from skbio.sequence import Protein
ident = make_identity_substitution_matrix(match_score=1, mismatch_score=0, alphabet=skbio.sequence.Protein.alphabet)

def assembleOverlappingPeptides(pepArr,overlap=11):
    """This is a work in progress, but the idea was
    to be able to rebuild the sequence from the set of
    overlapping 15mers..."""
    assembled = [pep for pep in pepArr]
    while len(assembled)>1:
        for pepi1, pepi2 in itertools.combinations(arange(len(assembled)), 2):
            pep1, pep2 = assembled[pepi1], assembled[pepi2]
            res = pairwise2.align.globalxs(pep2, pep1, -4, 0)[0]
            #print res[2]
            if res[2]>=overlap-8:
                #print res[0]
                #print res[1]

                _ = assembled.pop(pepi2)
                assembled[pepi1] = ''.join([aa1 if not aa1=='-' else aa2 for aa1, aa2 in zip(res[0], res[1])])
                #print assembled[pepi1]
                #print
                break
    return assembled[0]

def _assembleTwo(seq1, seq2):
    """This only works if two sequences share a significant identical overlap"""
    if len(seq2) <= len(seq1) and re.search(seq2, seq1):
        return seq1
    elif len(seq1) <= len(seq2) and re.search(seq1, seq2):
        return seq2
    else:
        msa = local_pairwise_align_ssw(Protein(seq1),
                                       Protein(seq2),
                                       substitution_matrix=ident)
        if msa[1] >= 8:
            try:
                (s1, e1), (s2, e2) = msa[-1]
            except:
                print(msa)

            if s1 >= s2:
                return seq1 + seq2[e2+1:]
            else:
                return seq2 + seq1[e1+1:]
            return out
        else:
            print('No significant overlap')
            raise

def _extendForward(seq, peptides, overlapL=8):
    ol = seq[-overlapL:]
    for pep in peptides:
        match = re.search(pep, ol)
        if match and match.start() > 0:
            return pep[:match.start()] + seq, pep

def _extendBackward(seq, peptides, overlapL=8):
    ol = seq[:overlapL]
    for pep in peptides:
        match = re.search(pep, ol)
        if match and match.end() < mat.endpos():
            return seq + pep[match.start():], pep

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Assemble overlapping 15mers')
    parser.add_argument('filename', metavar='FILENAME', type=str, help='Peptide set from Peptide DB')
    parser.add_argument('--useInds', action='store_true', help='Use the indices provided.')
    parser.add_argument('--useOverlap', action='store_true', help='Use the overlap to assemble, assuming 8 AA are unique')
    args = parser.parse_args()

    outFn = args.filename[:-4] + '.fasta'
    
    df = pd.read_excel(args.filename)
    if args.useInds:
        out = list(range(1, df['AA End'].max() + 1))
        for i, row in df.iterrows():
            pep = row['Peptide Sequence']
            errors = 0
            for aai, aa in zip(list(range(row['AA Start'], row['AA Start'] + len(pep))), pep):
                # print '(%s)%d%s' % (out[aai - 1], aai, aa)
                if out[aai - 1] == aai:
                    out[aai - 1] = aa
                elif not out[aai - 1] == aa:
                    print('Error in assembly at %s%d%s' % (out[aai - 1], aai, aa))
                    errors += 1
            if errors > 0:
                print(row['Peptide Id'], pep)
        for i, aa in enumerate(out):
            if isinstance(aa, int):
                out[i] = '-'
        fullSeq = ''.join(out)
    elif args.useOverlap:
        """Start with one peptide and add to it."""
        peptides = df['Peptide Sequence'].tolist()
        seq = peptides[0]
        peptides.remove(seq)
        while len(peptides) > 0 or len(peptides) < len(oldPeptides):
            oldPeptides = deepcopy(peptides)
            startOL = seq[:8]
            endOL = seq[-8:]
            middleOL = seq[8:-8]
            for pep in oldPeptides:
                if re.search(pep, middleOL):
                    peptides.remove(pep)
                    print('Found %s in middle of %s' % (pep, seq))
                    break
                elif re.search(pep, startOL) or re.search(pep, endOL):
                    print('Found %s in edge of %s' % (pep, seq))
                    seq = _assembleTwo(seq, pep)
                    print(seq)
                    peptides.remove(seq)
                    break
        fullSeq = seq
    else:
        pass

    fastaStr = '>%s\n%s\n' % (df['Peptide Group Name'].unique()[0], fullSeq)
    print(fastaStr)
    with open(outFn, 'w') as fh:
        fh.write(fastaStr)