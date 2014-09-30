"""
Package that first attempts to load a cython version of the Fisher's exact test:
    Fast Fisher's Exact Test (haibao tang, brent pedersen)
    https://pypi.python.org/pypi/fisher/
But falls back to the scipy test if it cannot be found
"""

from __future__ import division

__all__ = ['fisherTest']

try:
    """Attempt to use the fisher library (cython) if available (100x speedup)"""
    import fisher
    def fisherTest(tab,alternative='two-sided'):
        res = fisher.pvalue(tab[0][0],tab[0][1],tab[1][0],tab[1][1])
        OR = (tab[0][0] * tab[1][1]) / (tab[0][1] * tab[1][0])

        if alternative == 'two-sided':
            return (OR,res.two_tail)
        elif alternative == 'less':
            return (OR,res.left_tail)
        elif alternative == 'greater':
            return (OR,res.right_tail)
    print "Using Cython-powered Fisher's exact test"
except ImportError:
    from scipy import stats
    print "Using scipy.stats Fisher's exact test (slow)"
    fisherTest = stats.fisher_exact
