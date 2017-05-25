
import pandas as pd
import numpy as np
from copy import deepcopy
import argparse

import itertools
import operator
from functools import partial

# from HLAPredCache import *

__all__ = ['responseClass',
            'epitopeClass',
            'hamming',
            'overlap',
            'sharedCoords',
            'findResponseIslands',
            'findEpitopes',
            'identicalRule',
            'overlapRule',
            'hlaRule',
            'findpeptide']

class responseClass(object):
    def __init__(self, protein, peptideset, pepID, seq, start, end):
        self.protein = protein
        self.peptideset = peptideset
        self.pepID = pepID
        self.seq = seq
        self.start = start
        self.end = end
        self.L = len(seq)
    def __str__(self):
        return self.seq
    def __eq__(self, other):
        return self.seq == other.seq
    def __lt__(self, other):
        return self.seq < other.seq
    def __gt__(self, other):
        return self.seq > other.seq
    def __key__(self):
        return (self.protein, self.pepID, self.seq, self.start)
    def __hash__(self):
        return hash(self.__key__())
    @property
    def coords(self):
        return list(range(self.start, self.start + self.L))
    def todict(self):
        return dict(protein=self.protein,
                    pepID=self.pepID,
                    seq=self.seq,
                    start=self.start,
                    L=self.L)

class epitopeClass(responseClass):
    def __init__(self, parents, seq, start, end, hlas=[]):
        self.parents = list(parents)
        self.protein = self.parents[0].protein
        self.seq = seq
        self.start = start
        self.end = end
        self.L = len(seq)
        self.hlas = hlas
    def __key__(self):
        return (self.protein, self.seq, self.start)
    def todict(self):
        return dict(protein=self.protein,
                    seq=self.seq,
                    start=self.start,
                    L=self.L,
                    hlas='-'.join(self.hlas))

def hamming(str1, str2):
    """Hamming distance between two strings"""
    return sum([i for i in map(operator.__ne__, str1, str2)])

def overlap(response1, response2):
    """Any overlap between two responses?"""
    if response1.protein == response2.protein:
        coord2 = response2.coords
        for i in response1.coords:
            if i in coord2:
                return True

    return False

def sharedCoords(island):
    """Find the coordinates that are shared by ALL responses in the island"""
    if len(island) > 0:
        sc = set(island[0].coords)
        for response in island:
            sc.intersection_update(response.coords)
        return sorted(sc)
    else:
        return []

def findResponseIslands(responses):
    """Return a list of response islands, where each island is a group of overlapping responses"""
    responses = sorted(responses)
    islandList = []
    for r in responses:
        anyFound = False
        for islandi, island in enumerate(islandList):
            """Add the response to each existing island"""
            for islandResp in island:
                if overlap(r, islandResp):
                    islandList[islandi].append(r)
                    anyFound = True
                    break
        if not anyFound:
            """Start a new island"""
            islandList.append([r])
    return islandList

def findEpitopes(responses, sharedRule, **kwargs):
    """Given a list of responses find the minimum
    set of epitopes that "explain" all responses."""

    """Start with all possible sets of responses (i.e. combinations)"""
    sharedSets = []
    for N in range(1, len(responses) + 1):
        for inds in itertools.combinations(list(range(len(responses))), N):
            sharedSets.append(inds)

    """Remove any sets that don't have one unique shared response/epitope"""
    sharedSets = [ss for ss in sharedSets if sharedRule([responses[i] for i in ss], **kwargs)[0]]

    """Remove any sets that are a subset of another set"""
    sharedSetsCopy = deepcopy(sharedSets)
    for s1, s2 in itertools.product(sharedSetsCopy, sharedSetsCopy):
        if (not s1 is s2) and set(s1).issubset(set(s2)):
            try:
                sharedSets.remove(s1)
            except ValueError:
                """It has already been removed"""
                pass

    """Remove any sets whose members are all members of other sets, starting with shorter sets"""
    """Could help remove a few redundant responses, but may not be neccesary if we use the HLA rule"""
    anyRemoved = True
    while anyRemoved:
        anyRemoved = False
        newSharedSets = deepcopy(sharedSets)
        for s1 in sorted(newSharedSets, key=len):
            """Fore each set of responses"""
            eachExplained = []
            for respInd in s1:
                """For each response index in the set"""
                explained = False
                for s2 in newSharedSets:
                    """For each set of responses"""
                    if (not s2 is s1) and respInd in s2:
                        """If respInd is explained by another, larger set"""
                        explained = True
                eachExplained.append(explained)
            if all(eachExplained):
                sharedSets.remove(s1)
                anyRemoved = True
                """Start for s1 again with updated set of responses"""
                break
    epitopeIslands = [[responses[i] for i in inds] for inds in sharedSets]
    epitopes = [sharedRule([responses[i] for i in inds], **kwargs)[1] for inds in sharedSets]
    return epitopeIslands, epitopes

"""
Each rule returns True or False,
    if True it also returns the unique epitope representative of the response island

Simple "shared rule" implementation
(1) Define a rule to determine if a set of responses share a single response in common
    (8 AA overlap, 6 AA matching rule or HLA-based rule)
(2) Begin with all possible sets of responses (including sets of size N, N-1, N-2, ... 1)
(3) Eliminate sets that do not all share a common response
    (i.e. not all pairwise responses share a response)
(4) Eliminate sets that are a subset of another set
(5) Eliminate sets whose members are all members of other sets (starting with shorter sets)
(5) Remaining subsets are the true sets of responses (i.e. breadth)
"""

def identicalRule(island):
    s = set(island)
    if len(s) == 1:
        """If there's just one response"""
        ep = epitopeClass(parents=[island[0]], seq=island[0].seq, start=island[0].start, end=island[0].end)
        return True, ep
    else:
        return False, None

def overlapRule(island, minOverlap=8, minSharedAA=6):
    """If all responses share an overlap by at least minOverlap and 
    at least minSharedAA amino acid residues are matched in all responses
    then the responses are considered 'shared'"""
    
    """Find the shared peptide for responses"""
    sharedInds = sharedCoords(island)

    nOverlap = 0
    nMatching = 0
    seq = ''
    for si in sharedInds:
        aas = {resp.seq[si - resp.start] for resp in island}
        if '-' in aas:
            seq += '-'
        else:
            nOverlap += 1
            if len(aas) == 1:
                nMatching += 1
                seq += list(aas)[0]
            else:
                seq += 'X'
    if nOverlap >= minOverlap and nMatching >= minSharedAA:
        ep = epitopeClass(parents=island, seq=seq, start=sharedInds[0], end=sharedInds[-1])
        return True, ep
    else:
        return False, None

def hlaRule(island, hlaList, ba, topPct=0.1, nmer=[8, 9, 10]):
    """Determine overlap region common amongst all responses in the island
    Predict HLA binding for all mers within the overlap region, for all response sequences (putative epitopes)
    For each mer (defined by start position and length),
        If there is a mer within the topPct for every response then the responses are shared
        Return the mer whose avg rank is best amongst all the responses
    If none of the mers are ranked topPct in all responses then then return None

    TODO: untested and wil need to handle gaps"""
    
    """Find the shared peptide for responses"""
    sharedInds = sharedCoords(island)

    """Indices of all the possible epitopes in the island"""
    possibleEpitopeInds = getMers(sharedInds, nmer=nmer)
    rankPctMat = np.ones((len(island), len(possibleEpitopeInds), len(hlaList)))

    for ri, r in enumerate(island):
        """Get the predictions for all mers in this response"""
        ranks, sorti, kmers, ic50, hla = rankEpitopes(ba, hlaList, r.seq, nmer=nmer, peptideLength=None)
        for ii, inds in enumerate(possibleEpitopeInds):
            """Find the part of each epitope that overlaps with this response"""
            curMer = ''.join([r.seq[si-r.L] for si in inds])
            if len(curMer) in nmer:
                for hi, h in enumerate(hlaList):
                    curIC50 = ba[(h, curMer)]
                    rankPctMat[ri, ii, hi] = (ic50 < curIC50).sum() / len(ic50)

    """Are any of the possible HLA:mer pairs in the topPct for all responses?"""
    if (rankPctMat<=topPct).any(axis=2).any(axis=1).all():
        """Among HLA:mer pairs that are in the topPct, which one is the lowest on avg across responses?"""
        rankPctMat[rankPctMat > topPct] = np.nan
        tmp = np.nanmean(rankPctMat, axis=0)
        avgRank = np.nanmin(tmp)
        ii, hi = np.unravel_index(np.nanargmin(tmp), dims=tmp.shape)
        epitopeInds = possibleEpitopeInds[ii]
        hla = hlaList[hi]

        seq = ''
        for si in epitopeInds:
            aas = {resp.seq[si - resp.start] for resp in island}
            if len(aas) == 1:
                seq += list(aas)[0]
            else:
                seq += 'X'
        ep = epitopeClass(parents=island, seq=seq, start=epitopeInds[0], hlas=[hla], end=epitopeInds[-1])
        return True, ep
    else:
        return False, None

def findpeptide(pep, seq):
    """Find pep in seq ignoring gaps but returning a start position that counts gaps
    pep must match seq exactly (otherwise you should be using pairwise alignment)

    seq[startPos:endPos] = pep

    Parameters
    ----------
    pep : str
        Peptide to be found in seq.
    seq : str
        Sequence to be searched.
    returnEnd : bool
        Flag to return the end position such that:
        
    Returns
    -------
    startPos : int
        Start position (zero-indexed) of pep in seq or -1 if not found
    endPos : int
        Start position (zero-indexed) of pep in seq or -1 if not found"""

    ng = seq.replace('-', '')
    ngInd = ng.find(pep)
    ngCount = 0
    pos = 0
    """Count the number of gaps prior to the non-gapped position. Add them to it to get the gapped position"""
    while ngCount < ngInd or seq[pos] == '-':
        if not seq[pos] == '-':
            ngCount += 1
        pos += 1
    startPos = ngInd + (pos - ngCount)

    if startPos == -1:
        endPos = -1
    else:
        count = 0
        endPos = startPos
        while count < len(pep):
            if not seq[endPos] == '-':
                count += 1
            endPos += 1
    return startPos, endPos

