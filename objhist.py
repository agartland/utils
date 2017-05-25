
import numpy as np
from numpy.random import permutation, randint
from scipy import stats, special

try:
    from matplotlib.pyplot import plot, xticks, is_numlike, bar
except ImportError:
    print('Imported objhist without matplotlib.')

__all__ = ['objhist',
           'countdict']

def objhist(x=[], keys=None):
    """Count unique objects in x and return a dict of counts
    with added functionality (see countdict)

    Paramaters
    ----------
    x : iterator (e.g. list, string, ndarray, pd.Series)
        List of objects to be counted.
        Objects must be hashable as they will become keys in a dictionary.
    keys : optional, list
        Supply a set of required categories that will be set to 0 if not present in x.
    
    Returns
    -------
    out : countdict (subclass of dict)
        Unique objects in x are keys, with counts as values.
        Additional methods include: sum, freq, entropy, etc. (see countdict)

    Examples
    --------
    >>> a = randint(5,size=50)

    >>> oh = objhist(a)

    >>> print objhist(a)
    {0: 5, 1: 10, 2: 10, 3: 13, 4: 12}

    >>> print oh.freq()
    {0: 0.1, 1: 0.2, 2: 0.2, 3: 0.26, 4: 0.24}

    >>> print oh.topN(2)
    [(3, 13), (4, 12)]
    
    >>> print oh.generateRandomSequence(10,useFreqs = True)
    [3, 4, 0, 1, 1, 4, 4, 1, 3, 3]
    
    """
    out = countdict()
    if not keys is None:
        out.update({k:0 for k in keys})
    out.add(x)
    return out

class countdict(dict):
    """Subclass of dict to represent a histogram of discrete frequency distribution.
    Used by objhist() to generate a histogram of (hashable) objects.
    Adds methods for a few common operations on distributions of counts, but could be expanded..."""

    def sum(self):
        """Return the total counts over all categories"""
        return np.sum(list(self.values()))
    def freq(self):
        """Return the fraction of the total counts for each category"""
        tot = float(self.sum())
        return {k:self.get(k)/tot for k in list(self.keys())}
    def entropy(self,logFunc=np.log2):
        """Compute the entropy of the discrete distribution"""
        return -np.array([p*logFunc(p) for p in list(self.freq().values())]).sum()
    def simpsons_index(self, variant='D'):
        """Simpson's Index (D)
        Measures the probability that two individuals randomly selected from
        a sample will belong to the same species. With this index, 0
        represents infinite diversity and 1, no diversity.

        Simpson's Index of Diversity (1-D)
        The value of this index also ranges between 0 and 1, but now, the greater
        the value, the greater the sample diversity. The index represents the
        probability that two individuals randomly selected from a sample will
        belong to different species.

        Simpson's Reciprocal Index (1/D)
        Ranges from 1 to the number of species. The higher the value,
        the greater the diversity."""

        tot = float(self.sum())
        p = np.array([self[k]/tot for k in list(self.keys())])
        D = (p * p).sum()

        if variant == 'D':
            pass    
        elif variant == '1-D':
            D = 1 - D
        elif variant == '1/D':
            D = 1/D
        return D
        
    def relative_entropy(self,reference,log_func=np.log2):
        """Compute the relative entropy between the frequencies
        in this countdict object and those in reference.

        The Kullback-Leibler divergence is the negative sum of these values.

        Parameters
        ----------
            reference : dict
                Another objhist object with keys for each key in the calling object.

            log_func : function
                Function for computing log(). Allows for specification of the base to use.
        Returns
        -------
             : ndarray"""

        keys = list(self.keys())
        freq = self.freq()
        p = np.array([freq[k] for k in keys])
        q = np.array([reference.freq()[k] for k in keys])
        divergence = -p*log_func(p/q)
        return {k:v for k, v in zip(keys, divergence)}
        
    def jensen_shannon_divergence(self, b):
        """Compute Jensen-Shannon divergence between self and b (also an objhist).
        If keys from self are missing in b assume 0 counts."""

        keys = np.unique(list(self.keys()) + list(b.keys()))

        avec = np.array([self[k] if k in self else 0 for k in keys])
        bvec = np.array([b[k] if k in b else 0 for k in keys])

        return _jensen_shannon_divergence(avec, bvec)

    def morisita_horn_overlap(self, b):
        keys = np.unique(list(self.keys()) + list(b.keys()))

        avec = np.array([self[k] if k in self else 0 for k in keys])
        bvec = np.array([b[k] if k in b else 0 for k in keys])

        return _morisita_horn_index(avec, bvec)


    def uniqueness(self):
        return len(self)/self.sum()
    def sortedKeys(self,reverse=False):
        """Returns a list of the keys sorted ascending by frequency"""
        return sorted(list(self.keys()), key=self.get, reverse=reverse)

    def topN(self,n=5,reverse=True,returnFreq=False):
        """Returns a list of the top N most frequent keys/values as a list of tuples.

        Parameters
        ----------
        n : int
            Number of keys/values to return
        reverse : bool
            True (default) returns keys in descending order.
        returnFreq : bool
            True returns frequencies instead of counts.

        Returns
        -------
        out : list of tuples
            Ordered list of tuples e.g. [(k1,v1), (k2,v2)]             
        """

        if returnFreq:
            return [(k, self.freq()[k]) for i, k in zip(np.arange(n), self.sortedKeys(reverse=reverse))]
        else:
            return [(k, self[k]) for i, k in zip(np.arange(n), self.sortedKeys(reverse=reverse))]

    def add(self, newIter):
        """Add items in newIter to the existing frequency object.
        Object is updated in-place."""
        for k in newIter:
            try:
                self[k] += 1
            except KeyError:
                self[k] = 1
    def subset(self, newkeys):
        """Returns a copy of the countdict with only a subset of the keys remaining."""
        return countdict({k:self[k] for k in newkeys})

    def plot(self, color='gray', normed=True, barPlot=True):
        """Uses matplotlib to generate a minimalist histogram.

        Parameters
        ----------
        color : any valid matplotlib color (e.g. 'red', 'LightBrown' or (0.5,0.1,0.9) )

        normed : bool
            A normed histogram has fractional frequencies as heights.
        barPlot : bool
            True (default) produces a bar plot as opposed to a line with markers.

        Returns
        -------
        axh : matplotlib axes handle
        """
        if all([is_numlike(k) for k in list(self.keys())]):
            """If keys are numbers then use the x-axis scale"""
            if all([round(k)==k for k in list(self.keys())]):
                xvec = [int(k) for k in sorted(self.keys())]
            else:
                xvec = sorted(self.keys())
            xlab = xvec
        else:
            xlab = sorted(self.keys())
            xvec = np.arange(len(xlab))
        
        if normed:
            yDict = self.freq()
        else:
            yDict = self

        if barPlot:
            for x, k in zip(xvec, xlab):
                bar(x, yDict[k], align = 'center', color=color)
        else:
            plot(xvec, [yDict[k] for k in xlab], 's-', color=color)
        xticks(xvec, xlab)

    def generateRandomSequence(self, n=1, useFreqs=True):
        """Generate a random sequence of the objects in keys.
        Frequencies are optionally based on the observed frequencies.
        Returns a list of length n."""

        keys = list(self.keys())
        if useFreqs:
            freqDict = self.freq()
            """Ensure that it sums to 1 for stats.rv_discrete()"""
            freqArr = np.round(np.array([freqDict[k] for k in keys]), decimals=7)
            freqArr = freqArr/freqArr.sum()

            gridint = np.arange(len(keys))
            arbdiscrete = stats.rv_discrete(values=(gridint, freqArr), name='arbdiscrete')
            indices = arbdiscrete.rvs(size=n)
        else:
            indices = randint(len(keys), size=n)
        out = [keys[i] for i in indices]
        return out
    def returnList(self):
        """Return a list of objs that correspond exactly to the observed counts."""
        out = []
        for k in list(self.keys()):
            out.extend([k for i in arange(self[k])])
        return out


def _jensen_shannon_divergence(a, b):
    """Compute Jensen-Shannon Divergence

    Lifted from github/scipy:
    https://github.com/luispedro/scipy/blob/ae9ad67bfc2a89aeda8b28ebc2051fff19f1ba4a/scipy/stats/stats.py

    Parameters
    ----------
    a : array-like
        possibly unnormalized distribution
    b : array-like
        possibly unnormalized distribution. Must be of same size as ``a``.
    
    Returns
    -------
    j : float
    """
    a = np.asanyarray(a, dtype=np.float)
    b = np.asanyarray(b, dtype=np.float)
    a = a/a.sum()
    b = b/b.sum()
    m = (a + b)
    m /= 2.
    m = np.where(m, m, 1.)
    return 0.5*np.sum(special.xlogy(a, a/m)+special.xlogy(b, b/m))

def _morisita_horn_index(a, b):
    """Compute the Morisita-Horn overlap index between two count vectors

    https://en.wikipedia.org/wiki/Morisita%27s_overlap_index
    http://www.ncbi.nlm.nih.gov/pmc/articles/PMC3543521/

    Parameters
    ----------
    a : array-like
        possibly unnormalized distribution
    b : array-like
        possibly unnormalized distribution. Must be of same size as ``a``.
    
    Returns
    -------
    j : float
        
    """
    a = np.asanyarray(a, dtype=np.float)
    b = np.asanyarray(b, dtype=np.float)

    freqa = a/a.sum()
    freqb = b/b.sum()

    numer = 2 * (a*b).sum()
    """From wikipedia, confirmed in multiple texts and mothur"""
    denom = ( (a*a).sum()/(a.sum()**2) + (b*b).sum()/(b.sum()**2) ) * a.sum() * b.sum()
    mh1 = numer/denom

    """This is identical algebraically"""
    '''numer2 = 2 * (freqa * freqb).sum()
    denom2 = ((freqa*freqa).sum() + (freqb*freqb).sum())
    mh2 = numer2/denom2'''

    """Not sure where this is from but it gives a different answer..."""
    # mh3 = np.sum(np.sqrt(freqa * freqb))
   
    return mh1

def _simpsons_index(vec, variant='D'):
    """Simpson's Index (D)
    Measures the probability that two individuals randomly selected from
    a sample will belong to the same species. With this index, 0
    represents infinite diversity and 1, no diversity.

    Simpson's Index of Diversity (1-D)
    The value of this index also ranges between 0 and 1, but now, the greater
    the value, the greater the sample diversity. The index represents the
    probability that two individuals randomly selected from a sample will
    belong to different species.

    Simpson's Reciprocal Index (1/D)
    Ranges from 1 to the number of species. The higher the value,
    the greater the diversity.

    Parameters
    ----------
    vec : ndarray, shape [nCategories,]
        Number or frequencies of observations for each category
    variant : str
        Indicates variation to apply: "D", "1-D" or "1/D"

    Returns
    -------
    index : float"""

    tot = np.sum(vec).astype(float)
    p = np.array(vec, dtype=float) / tot

    D = (p * p).sum()
    if variant == 'D':
        pass    
    elif variant == '1-D':
        D = 1 - D
    elif variant == '1/D':
        D = 1/D
    return D