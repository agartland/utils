import pandas as pd
import numpy as np
import parasail
import itertools

__all__ = ['computePWDist',
           'CachedNWDistance',
           'nw_metric']

def members2seqs(members, ssDf, indexCol, gbCol, countCol, gbValues=None):
    if gbValues is None:
        gbValues = sorted(ssDf[gbCol].unique())

    uIndices = ssDf[indexCol].dropna().unique()
    cnts = ssDf.groupby([indexCol, gbCol])[countCol].agg(np.sum).unstack(gbCol, fill_value=0)[gbValues]
    cnts = cnts.loc[uIndices[list(members)]]
    
    seqs = cnts.index.tolist()
    weights = cnts.sum(axis=1).values
    return seqs, weights

def computeCDR3PWDist(seqs, gap_open=3, gap_extend=3, matrix=parasail.blosum62, useIdentity=False):
    """Compute paiwrwise distances among all sequences using global NW alignment"""
    cache = CachedNWDistance(seqs, matrix=matrix, gap_open=gap_open, gap_extend=gap_extend, useIdentity=useIdentity)

    indices = cache.indices()
    L = indices.shape[0]
    pwdist = np.nan * np.zeros((L, L))
    
    for i, j in itertools.product(range(L), range(L)):
        
        if i <= j:
            d = cache.metric(indices[i], indices[j])
            pwdist[i, j] = d
            pwdist[j, i] = d

    pwdist = pd.DataFrame(pwdist, columns=cache.elements, index=cache.elements)
    return pwdist

def nw_metric(self, s1, s2):
    """May or may not produce a true metric. Details in:
        E. Halpering, J. Buhler, R. Karp, R. Krauthgamer, and B. Westover.
            Detecting protein sequence conservation via metric embeddings.
            Bioinformatics, 19(Suppl. 1):i122–i129, 2003"""
    xx = parasail.nw_stats(s1, s1, open=3, extend=3, matrix=parasail.blosum62).score
    yy = parasail.nw_stats(s2, s2, open=3, extend=3, matrix=parasail.blosum62).score
    xy = parasail.nw_stats(s1, s2, open=3, extend=3, matrix=parasail.blosum62).score
    D = xx + yy - 2 * xy
    return D

class CachedNWDistance:
    def __init__(self, elements, gap_open=3, gap_extend=3, matrix=parasail.blosum62, useIdentity=False):
        self.sim_cache = {}
        self.elements = elements
        self.e2i = {e:i[0] for e,i in zip(elements, self.indices())}
        self.i2e = {i[0]:e for e,i in zip(elements, self.indices())}
        self.matrix = matrix
        self.gap_extend = gap_extend
        self.gap_open = gap_open

        if useIdentity:
            self.matrix = parasail.matrix_create(alphabet='ACDEFGHIKLMNPQRSTVWXY', match=1, mismatch=0)
        self.paraParams = dict(open=self.gap_open, extend=self.gap_extend, matrix=self.matrix)

    def indices(self):
        return np.arange(len(self.elements), dtype=np.float)[:, None]
    
    def _try_cache(self, e):
        try:
            xx = self.sim_cache[(e, e)]
        except KeyError:
            xx = parasail.nw_stats(e, e, **self.paraParams).score
            self.sim_cache[(e, e)] = xx
        return xx

    def metric(self, i1, i2):
        """sklearn specifies that function will receive
        two rows as parameters and return one value as distance"""
        xx = self._try_cache(self.i2e[i1[0]])
        yy = self._try_cache(self.i2e[i2[0]])

        """Don't need to cache the xy similarity because it doesn't have other uses"""
        xy = parasail.nw_stats(self.i2e[i1[0]], self.i2e[i2[0]], **self.paraParams).score

        D = xx + yy - 2 * xy
        return D