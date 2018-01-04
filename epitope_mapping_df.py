
import pandas as pd
import numpy as np
from copy import deepcopy
import argparse

import itertools
import operator
from functools import partial

"""Rewrite of epitope_mapping.py that avoids the use of objects,
thereby permitting easy writing/reading of epitope mapping
results to/from files.

response : pd.Series
    A single row from the response DataFrame
    Row is indexed by attributes of the object
island : pd.DataFrame
    A subset of the whole response DataFrame
    containing responses from one PTID that overlap
epitope : pd.Series
    A row from the DataFrame of epitopes
    Has an epitope ID that is specific to the PTID
    Can be merged with the respons DataFrame using the epitopeID"""

# from HLAPredCache import *

__all__ = [ 'hamming',
            'overlap',
            'sharedCoords',
            'findResponseIslands',
            'findEpitopes',
            'identicalRule',
            'overlapRule',
            'hlaRule',
            'findpeptide']



def hamming(str1, str2):
    """Hamming distance between two strings"""
    return sum([i for i in map(operator.__ne__, str1, str2)])

def _coords(r):
    return list(range(int(r.start), int(r.start) + r.L))

def overlap(response1, response2):
    """Any overlap between two responses?"""
    if response1.protein == response2.protein:
        coord2 = _coords(response2)
        for i in _coords(response1):
            if i in coord2:
                return True

    return False

def sharedCoords(island):
    """Find the coordinates that are shared by ALL responses in the island"""
    if island.shape[0] > 0:
        sc = set(_coords(island.iloc[0]))
        for ri in range(island.shape[0]):
            response = island.iloc[ri]
            sc.intersection_update(_coords(response))
        return sorted(sc)
    else:
        return []

def assignResponseIslands(responses):
    """Return a pd.Series with an island ID that can be joined with
    the responses pd.DataFrame"""
    
    """Contains lists of row indexes of overlapping responses"""
    islandList = []
    for ri in range(responses.shape[0]):
        r = responses.iloc[ri]
        anyFound = False
        for islandi, island in enumerate(islandList):
            """Add the response to each existing island"""
            for islandRespi in island:
                if overlap(r, responses.iloc[islandRespi]):
                    islandList[islandi].append(ri)
                    anyFound = True
                    break
        if not anyFound:
            """Start a new island"""
            islandList.append([ri])
    
    """Return an island column indexed like responses"""
    islandColumn = [''] * responses.shape[0]
    for islandi,island in enumerate(islandList):
        for i in island:
            islandColumn[i] = 'I%d' % islandi
    out = pd.DataFrame({'IslandID': islandColumn})
    out.index = responses.index
    return out

def findEpitopes(responses, sharedRule, **kwargs):
    """Given a list of responses find the minimum
    set of epitopes that "explain" all responses."""

    """Start with all possible sets of responses (i.e. combinations)"""
    sharedSets = []
    for N in range(1, responses.shape[0] + 1):
        for inds in itertools.combinations(list(range(responses.shape[0])), N):
            sharedSets.append(list(inds))

    """Remove any sets that don't have one unique shared response/epitope"""
    sharedSets = [ss for ss in sharedSets if sharedRule(responses.iloc[ss], **kwargs)[0]]

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
            """For each set of responses"""
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

    """Return epitope columns indexed like responses"""
    epitopes = [None] * responses.shape[0]
    for epitopei, inds in enumerate(sharedSets):
        ep = sharedRule(responses.iloc[inds], **kwargs)[1]
        ep['EpID'] = 'E%d' % epitopei
        for i in inds:
            epitopes[i] = ep
        
    epitopesDf = pd.DataFrame(epitopes)
    epitopesDf.index = responses.index
    return epitopesDf

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
    s = set(island.seq)
    if len(s) == 1:
        """If there's just one response"""
        ep = dict(EpSeq=island.iloc[0].seq, EpStart=island.iloc[0].start, EpEnd=island.iloc[0].end)
        return True, ep
    else:
        return False, {}

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
        aas = {respSeq[si - int(respStart)] for respSeq, respStart in zip(island.seq, island.start)}
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
        ep = dict(EpSeq=seq, EpStart=sharedInds[0], EpEnd=sharedInds[-1])
        return True, ep
    else:
        return False, {}

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

    for ri, (rSeq, rL) in enumerate(zip(island.seq, island.L)):
        """Get the predictions for all mers in this response"""
        ranks, sorti, kmers, ic50, hla = rankEpitopes(ba, hlaList, rSeq, nmer=nmer, peptideLength=None)
        for ii, inds in enumerate(possibleEpitopeInds):
            """Find the part of each epitope that overlaps with this response"""
            curMer = ''.join([rSeq[si-rL] for si in inds])
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
            aas = {respSeq[si - respStart] for respSeq, respStart in zip(island.seq, island.start)}
            if len(aas) == 1:
                seq += list(aas)[0]
            else:
                seq += 'X'
        ep = dict(EpSeq=seq, EpStart=epitopeInds[0], EpEnd=epitopeInds[-1], hlas=[hla])
        return True, ep
    else:
        return False, {}

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


def plotIsland(ptid, island, ruleFunc=overlapRule, ruleKwargs={}, toHXB2=None):
    subIslands, epitopes = findEpitopes(island, ruleFunc, **ruleKwargs)
    #subIslands, epitopes = findEpitopes(island, overlapRule)

    """Build an x-vector and AA vector"""
    sitex = []
    immunogens = []
    for ep in epitopes:
        sitex.extend(ep.coords)
        for r in ep.parents:
            sitex.extend(r.coords)
            immunogens.append(r.peptideset)

    sitex = np.array(sorted(set(sitex)))
    xx = np.arange(len(sitex))
    sitex2xx = {i:j for i, j in zip(sitex, xx)}

    immunogens = set(immunogens)
    colors = {p:col for p, col in zip(immunogens, palettable.colorbrewer.qualitative.Set1_3.mpl_colors)}

    plt.clf()
    ss=[]
    y=1
    for r in island:
        col = colors[r.peptideset]

        plt.plot([sitex2xx[r.start], sitex2xx[r.end]], [y, y], '-s', lw=2, mec='gray', color=col)
        for xoffset, aa in enumerate(r.seq):
            plt.annotate(aa, xy=(sitex2xx[r.start] + xoffset, y - 0.1), ha='center', va='top', size='small')
        
        """if not hlaList is None and not ba is None:
            ranks,sorti,kmers,ic50,hla=rankKmers(ba,hlaList,r[2],nmer=[8,9,10,11],peptideLength=None)
            rankList=ranks.tolist()
            mers,inds=getMerInds(r[2],nmer=[8,9,10,11])
            notes={}
            for curRank in arange(5):
                curi=where(ranks==curRank)[0][0]
                xcoord=sitex2xx[inds[mers.index(kmers[curi])][0]+r[3]]
                if not notes.has_key(xcoord) or notes[xcoord][1] > ic50[curi]:
                    if AList.issuperset(kmers[curi]):
                        notes.update({xcoord:(hla[curi][:4],ic50[curi],len(kmers[curi]),'%d|A' % (curRank+1))})
                    elif BList.issuperset(kmers[curi]):
                        notes.update({xcoord:(hla[curi][:4],ic50[curi],len(kmers[curi]),'%d|B' % (curRank+1))})
                    else:
                        notes.update({xcoord:(hla[curi][:4],ic50[curi],len(kmers[curi]),'%d' % (curRank+1))})
            for k,v in notes.items():
                annotate('%s\n%1.1f\nL%d\nR%s' % v,
                         xy=(k,y+0.08),
                         xytext=(-1,0),textcoords='offset points',
                         ha='left',va='bottom',size='x-small')"""

        ss += [r.start, r.end]
        y += 1

    y = 0
    for e in epitopes:
        xvec = [sitex2xx[e.start] - 0.3, sitex2xx[e.end] + 0.3]
        plt.fill_between(xvec, [y, y], len(island), color='k', edgecolor='None', alpha=0.2)
        for xoffset, aa in enumerate(e.seq):
            plt.annotate(aa, xy=(sitex2xx[e.start] + xoffset, y + 0.1), ha='center', va='bottom', size='medium', weight='bold')
        ss += [e.start, e.end]
        y -= 1

    plt.title('PUB-ID: %s (%d epitopes)' % (ptid, len(epitopes)))

    plt.yticks([])
    ss = np.unique(ss)
    if not toHXB2 is None:
        plt.xticks([sitex2xx[sx] for sx in ss], [toHXB2[aai] for aai in ss], size='x-large')
        plt.xlabel('%s HXB2 coordinate' % r.protein.title(), fontsize='x-large')
    else:
        plt.xticks([sitex2xx[sx] for sx in ss], ss.astype(int), size='x-large')
        plt.xlabel('%s coordinate' % r.protein.title(), fontsize='x-large')
    plt.xlim((-1, len(xx)))
    plt.ylim((-len(epitopes), len(island)+1))
    
    # plt.ylabel('Responses and epitopes',fontsize='x-large')
    handles = [mpl.patches.Patch(facecolor=colors[p], edgecolor='k') for p in immunogens]
    plt.legend(handles, immunogens, loc='best', title=None)
