from __future__ import division
import numpy as np
from matplotlib.pyplot import plot, xticks,is_numlike,bar
from numpy.random import permutation,randint
from scipy import stats

__all__ = ['objhist',
           'countdict']

def objhist(x,keys=None):
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

    for k in x:
        out.add(k)
    return out

class countdict(dict):
    """Subclass of dict to represent a histogram of discrete frequency distribution.
    Used by objhist() to generate a histogram of (hashable) objects.
    Adds methods for a few common operations on distributions of counts, but could be expanded..."""

    def sum(self):
        """Return the total counts over all categories"""
        tot = 0
        for k in self.keys():
            tot+=self.get(k)
        return tot
    def freq(self):
        """Return the fraction of the total counts for each category"""
        tot = self.sum()
        return {k:self.get(k)/tot for k in self.keys()}
    def entropy(self,logFunc=np.log2):
        """Compute the entropy of the discrete distribution"""
        return -np.array([p*logFunc(p) for p in self.freq().values()]).sum()
    def relative_entropy(self,reference_freq,logFunc=np.log2):
        """Compute the relative entropy between the frequencies
        in this countdict object and those in referenceFreq.

        The Kullback–Leibler divergence is the negative sum of these values.

        Parameters
        ----------
            referenceFreq : dict
                Keys for each key in the calling object 

            logFunc : function
                Function for computing log(). Allows for specification of the base to use.
        Returns
        -------
             : ndarray"""

        keys = self.keys()
        freq = self.freq()
        p = np.array([freq[k] for k in keys])
        q = np.array([reference.freq()[k] for k in keys])
        divergence = -p*logFunc(p/q)
        return {k:v for k,v in zip(keys,divergence)}
    def sortedKeys(self,reverse=False):
        """Returns a list of the keys sorted ascending by frequency"""
        return sorted(self.keys(), key=self.get, reverse=reverse)

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
            return [(k,self.freq()[k]) for i,k in zip(np.arange(n),self.sortedKeys(reverse=reverse))]
        else:
            return [(k,self[k]) for i,k in zip(np.arange(n),self.sortedKeys(reverse=reverse))]

    def add(self,newIter):
        """Add items in newIter to the existing frequency object.
        Object is updated in-place."""
        for k in newIter:
            try:
                self[k] += 1
            except KeyError:
                self[k] = 1
    def subset(self,newkeys):
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
        if all([is_numlike(k) for k in self.keys()]):
            """If keys are numbers then use the x-axis scale"""
            if all([round(k)==k for k in self.keys()]):
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
            for x,k in zip(xvec,xlab):
                bar(x, yDict[k], align = 'center', color=color)
        else:
            plot(xvec, [yDict[k] for k in xlab], 's-', color=color)
        xticks(xvec,xlab)

    def generateRandomSequence(self, n=1, useFreqs=True):
        """Generate a random sequence of the objects in keys.
        Frequencies are optionally based on the observed frequencies.
        Returns a list of length n."""

        keys = self.keys()
        if useFreqs:
            freqDict = self.freq()
            """Ensure that it sums to 1 for stats.rv_discrete()"""
            freqArr = np.round(np.array([freqDict[k] for k keys]), decimals=7)
            freqArr = freqArr/freqArr.sum()

            gridint = np.arange(len(keys))
            arbdiscrete = stats.rv_discrete(values=(gridint,freqArr), name='arbdiscrete')
            indices = arbdiscrete.rvs(size=n)
        else:
            indices = randint(len(keys), size=n)
        out = [keys[i] for i in indices]
        return out
    def returnList(self):
        """Return a list of objs that correspond exactly to the observed counts."""
        out = []
        for k in self.keys():
            out.extend([k for i in arange(self[k])])
        return out

