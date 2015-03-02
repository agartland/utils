"""
Package that first attempts to load a cython version of the Fisher's exact test:
    Fast Fisher's Exact Test (Haibao Tang, Brent Pedersen)
    https://pypi.python.org/pypi/fisher/
But falls back to the scipy test if it cannot be found
"""

from __future__ import division
import numpy as np

__all__ = ['fisherTest','fisherTestVec']

try:
    """Attempt to use the fisher library (cython) if available (100x speedup)"""
    import fisher
    def fisherTest(tab,alternative='two-sided'):
        """Fisher's exact test on a 2x2 contingency table.

        Wrapper around fisher.pvalue found in:
        Fast Fisher's Exact Test (Haibao Tang, Brent Pedersen)
        https://pypi.python.org/pypi/fisher/

        Test is performed in C (100x speed-up)

        Parameters
        ----------
        tab : list of lists or 2x2 ndarray
            Each element should contain counts
        alternative : string
            Specfies the alternative hypothesis (similar to scipy.fisher_exact)
            Options: 'two-sided', 'less', 'greater' """
        
        res = fisher.pvalue(tab[0][0],tab[0][1],tab[1][0],tab[1][1])
        OR = (tab[0][0] * tab[1][1]) / (tab[0][1] * tab[1][0])

        if alternative == 'two-sided':
            return (OR,res.two_tail)
        elif alternative == 'less':
            return (OR,res.left_tail)
        elif alternative == 'greater':
            return (OR,res.right_tail)

    def fisherTestVec(a,b,c,d,alternative='two-sided'):
        """Vectorized Fisher's exact test performs n tests
        on 4 n-dimensional numpy vectors a, b, c, and d representing
        the 4 elements of a 2x2 contigency table.

        Wrapper around fisher.pvalue_npy found in:
        Fast Fisher's Exact Test (Haibao Tang, Brent Pedersen)
        https://pypi.python.org/pypi/fisher/

        Loop and test are performed in C (100x speed-up)

        Parameters
        ----------
        a,b,c,d : shape (n,) ndarrays
            Vector of counts (will be cast as uint8 for operation)
        alternative : string
            Specfies the alternative hypothesis (similar to scipy.fisher_exact)
            Options: 'two-sided', 'less', 'greater' """

        res = fisher.pvalue_npy(a.astype(np.uint),b.astype(np.uint),c.astype(np.uint),d.astype(np.uint))
        OR = (a*d)/(b*c)

        if alternative == 'two-sided':
            return (OR,res[2])
        elif alternative == 'less':
            return (OR,res[0])
        elif alternative == 'greater':
            return (OR,res[1])

    print "Using Cython-powered Fisher's exact test"

except ImportError:
    from scipy import stats
    print "Using scipy.stats Fisher's exact test (slow)"
    fisherTest = stats.fisher_exact
