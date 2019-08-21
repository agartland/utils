import pandas as pd
import numpy as np
import itertools

import parasail

__all__ = ['CachedNWDistance',
           'computeCDR3PWDist',
           'aligned_mm_metric',
           'nw_metric']

def computeCDR3PWDist(seqs, gap_open=3, gap_extend=3, matrix=parasail.blosum62, useIdentity=False, cache=None):
    """Compute paiwrwise distances among all sequences using global NW alignment"""
    if cache is None:
        cache = CachedNWDistance(seqs, matrix=matrix, gap_open=gap_open, gap_extend=gap_extend, useIdentity=useIdentity)

    metric = cache.single_chain_metric
    
    indices = cache.indices()
    L = indices.shape[0]
    pwdist = np.nan * np.zeros((L, L))
    
    for i, j in itertools.product(range(L), range(L)):
        if i <= j:
            d = metric(indices[i], indices[j])
            pwdist[i, j] = d
            pwdist[j, i] = d

    pwdist = pd.DataFrame(pwdist, columns=cache.elements, index=cache.elements)
    return pwdist

def aligned_mm_metric(s1, s2, open=3, extend=3, matrix=parasail.blosum62):
    res = parasail.nw_stats(s1, s2, open=open, extend=extend, matrix=matrix)
    return res.length - res.matches

def nw_metric(s1, s2):
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
    
    def _try_cache(self, e1, e2):
        try:
            xx = self.sim_cache[(e1, e2)]
        except KeyError:
            xx = parasail.nw_stats(e1, e2, **self.paraParams).score
            self.sim_cache[(e1, e2)] = xx
        return xx

    def single_chain_metric_simple(self, e1, e2):
        xx = self._try_cache(e1, e1)
        yy = self._try_cache(e2, e2)

        """Don't need to cache the xy similarity because it doesn't have other uses"""
        xy = self._try_cache(e1, e2)
        D = xx + yy - 2 * xy
        return D

    def single_chain_metric(self, i1, i2):
        """sklearn specifies that function will receive
        two rows as parameters and return one value as distance"""
        xx = self._try_cache(self.i2e[i1[0]], self.i2e[i1[0]])
        yy = self._try_cache(self.i2e[i2[0]], self.i2e[i2[0]])

        """Don't need to cache the xy similarity because it doesn't have other uses"""
        xy = self._try_cache(self.i2e[i1[0]], self.i2e[i2[0]])

        D = xx + yy - 2 * xy
        return D