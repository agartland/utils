from __future__ import division
import numpy as np
import pandas as pd
from numpy.random import permutation
import time
import logging

try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError:
    print 'Imported epitope_hotspot without matplotlib.'


import statsmodels.api as sm
from HLAPredCache import grabKmer
from objhist import objhist

__all__ = ['getBAMat',
           'get2DBAMat',
           'populationBindingMap',
           'plotBindingGrid',
           'BAMat2map',
           'computeHLAHotspots',
           'plotKmerHotpots',
           'plotSiteHotpots',
           'plotAlignmentBindingGrid']

_MAX_IC50 = 15

"""Set up logger for printing progress of long-running simulations and analysis"""
logger = logging.getLogger('epitope_hotspot')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s:%(asctime)s:%(message)s')

"""Create console handler and set level to INFO"""
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)

def get2DBAMat(peptides, hlas, ba):
    """Generate a pd.DataFrame of log(IC50) binding affinities
    with index of peptides and columns of HLA alleles.

    If HLA alleles are keys in dict then they are sorted() to generate the matrix.

    Some predictions may be nan if they are not found in ba.
    
    Parameters
    ----------
    peptides : list/iterator
        Peptides whose HLA binding affinities should be in ba.
    hlas : dictionary or list of HLA alleles
        HLA alleles as keys in dict or elements in list
    ba : hlaPredCache
        Should contain log(IC50) for all HLA allele and peptide combinations.

    Returns
    -------
    bamat : pd.DatFrame
        Containing all HLA:peptide pairs of log(IC50) predictions with shape [index: hlaList x columns: peptides]
    """
    try:
        hlaList = sorted(hlas.keys())
    except AttributeError:
        hlaList = hlas

    hlaList = convertHLAAsterisk(hlaList)

    bamat = _MAX_IC50 * np.ones((len(hlaList),len(peptides)))
    for hlai,h in enumerate(hlaList):
        for pepi, pep in enumerate(peptides):
            bamat[hlai,pepi] = ba[(h,pep)]
    baDf = pd.DataFrame(bamat, index = hlaList, columns = peptides)
    return baDf

def getBAMat(seqs, hlas, ba, k=9, gapless=False):
    """Generate a 3D np.ndarray of log(IC50) binding affinities
    with shape [nKmers x nSeqs x nHLAs]

    If HLA alleles are keys in dict then they are sorted() to generate the matrix.

    Some predictions may be nan if they are not found in ba.

    By default all kmers containing a gap have Nan affinity.
    
    Parameters
    ----------
    seqs : list/iterator
        Aligned AA sequences.
    hlas : dictionary or list of HLA alleles
        HLA alleles as keys in dict or elements in list
    ba : hlaPredCache
        Should contain log(IC50) for all HLA allele and kmer combinations.
    k : int
        Width of the peptides to use for HLA binding prediction.
    gapless : bool
        If True then skip gap characters when grabbing kmers from the alignment.
        Kmers starting with a gap are returned as nan.
        NOTE: This option will "break" the alignment by skipping gapped sites.
              Do not use if you intend to make an integrated site-based map.

    Returns
    -------
    bamat : np.ndarray
        Array of shape [nKmers x nSeqs x nHLAs]
    """

    """Do all seqs have the same length"""
    if len(np.unique(map(len,seqs))) > 1:
        raise ValueError('Sequences must all have the same length (%s)' % np.unique(map(len,seqs)))

    nHLAs = len(hlas)
    nSeqs = len(seqs)
    nSites = int(np.median(map(len,seqs)))
    nKmers = nSites - k + 1

    try:
        hlaList = sorted(hlas.keys())
    except AttributeError:
        hlaList = hlas

    baMat = np.nan * np.ones((nKmers,nSeqs,nHLAs), dtype = np.float64)

    """Initialize the 3 dimensional binding affinity matrix"""
    for seqi,s in enumerate(seqs):
        for meri in range(nKmers):
            gappedMer,gaplessMer = grabKmer(s,meri,k=k)
            if gapless:
                """Use the gapped 9mer to preserve the alignment precisely"""
                mer = gaplessMer
            else:
                mer = gappedMer
            
            if mer is None:
                baMat[meri,seqi,:] = np.nan
            else:
                for hlai,h in enumerate(hlaList):
                    baMat[meri,seqi,hlai] = ba[(h,mer)]
    return baMat

def BAMat2map(mat, hlas, siteMap=True, k=9, phenotypicFrequencies=False):
    """Take a matrix containing HLA binding data organized by [kmers, seqs, hlas]
    and return a vector of length nKmers or nSites that is:

    (1) Weighted by and aggregated across HLA allele frequencies (optionally phenotypicFrequencies)
    (2) Summed over seqs
    (3) Optionall integrated within kmers (i.e. translated to sites)

    To work properly this should be matrix of 1/0's indicating binding for each HLA:kmer pair
    over the alignment. But if phenotypicFrequencies is False then its simply
    a weighted sum over HLAs, summed over seqs and optionally integrated ove kmers.

    If hlas is a list of alleles then it assumes uniform weights.

    Parameters
    ----------
    mat : ndarray
        Matrix of binding affinities (or values that have been operated on)
        with shape [nKmers x nSeqs x nHLAs]
    hlas : dict or list
        Allele frequencies as key/value pairs.
        A uniform distribution is assumed over all alleles if a list is given.
    siteMap : bool
        Return a site-wise map if True, otherwise return a kmer-based map.
        A site-wise map is integrated over kmers.
    k : int
        Width of the peptide for HLA binding affinity prediction
    phenotypicFrequencies : bool
        Convert HLA allele frequencies to phenotypic frequencies
        (sums homozygous and heterozygous probabilities,
        assuming no linkage disequilibrium with other allelels)

    Returns
    -------
    vec : ndarray shape [1,]
        Vector/map of HLA binding by kmer or optionally by site.
    """

    try:
        hlaList = sorted(hlas.keys())
    except AttributeError:
        logger.info('No HLA frequencies given, assuming uniform distribution.')
        hlaList = sorted(hlas)
        hlas = {h:1./len(hlaList) for h in hlaList}
    
    freqVec = np.array([hlas[h] for h in hlaList])

    nKmers,nSeqs,nHLAs = mat.shape

    """Multiply element-wise by hla allele frequencies before summing"""
    weightedMat = mat * np.tile(freqVec.reshape((1,1,nHLAs)), (nKmers,nSeqs,1))

    if phenotypicFrequencies:
        """Convert total allele frequency into phenotypic frequency"""
        locusFreq = np.zeros((2,nKmers,nSeqs))
        for loci,locus in enumerate('AB'):
            locusInd = [i for i,h in enumerate(hlas) if h[0]==locus]
            tmpFreq = weightedMat[:,:,locusInd].sum(axis=2)
            locusFreq[loci,:,:] = tmpFreq**2 + 2*tmpFreq*(1-tmpFreq)
        hlaBindingFreq = locusFreq[0,:,:]*locusFreq[1,:,:] + locusFreq[0,:,:]*(1-locusFreq[1,:,:]) + locusFreq[1,:,:]*(1-locusFreq[0,:,:])
    else:
        """Compute the total frequency of the alleles that will bind."""
        hlaBindingFreq = weightedMat.sum(axis=2)

    """Sum across sequences"""
    vec = hlaBindingFreq.sum(axis=1)

    if siteMap:
        """Sum across sites for each kmer, towards a Pr(epitope) by site"""
        return np.convolve(vec, np.ones(k))
    else:
        return vec

def populationBindingMap(align,hlas,ba,bindingThreshold=5,k=9,atLeastFracSeqs=0):
    """Compute the fraction of the population that will bind the kmer
    at each position, averaged across seqs.

    Parameters
    ----------
    align : list or pd.Series
        Set of amino acid sequences.
    hlas : dict
        HLA allele frequencies as key/value pairs (e.g. (A*0201:0.54))
        Allele frequencies should sum to 1 across all HLA-A or HLA-B alleles (separately).
    ba : hlaPredCache
        Dictionary of log binding affinity for all kmers in align and all HLAs in hlas
    bindingThreshold : float
        Units of logIC50 (or whatever is in ba)
    k : int
        Width of the mer for computing HLA binding predictions
    atLeastFracSeqs : float fraction
        Fraction of sequences required for each HLA to be considered a "binder" there.

    Returns
    -------
    vec : ndarray of shape[nKmers, 1]
        Vector with element for each start position
        indicating fraction of population binding the kmer, on average
    """
    """baMat is shape [nKmers x nSeqs x nHLAs]"""
    baMat = (getBAMat(align,hlas,ba) < bindingThreshold).astype(np.float64)
    
    if atLeastFracSeqs>0:
        """Squashes baMat into [nKmers x 1 x nHLAs] indicating
        if each HLA binds at least X pct of sequences"""
        baMat = (baMat.mean(axis=1)[:,None,:] > atLeastFracSeqs).astype(np.float64)

    """If baMat is not squashed then vec is an average over sequences"""
    emap = BAMat2map(baMat, hlas, siteMap = False, k = 9, phenotypicFrequencies = True)/baMat.shape[1]
    return emap

def computeHLAHotspots(seqs, hlas, ba, hlaMask = None, bindingThreshold = 5, nPerms = 1e2, k = 9):
    """Creates a map of epitope hotspots in the aligned sequences in seqs,
        based on the alleles and their frequencies in hlas e.g. {'A*0201':0.48, 'B*2701':0.35}

    Returns a pdf (sums to 1 across kmers) with the relative Pr that a kmer is an epitope,
        given the hlas, their frequencies and the sequences.
    Frequencies can be relative, but PDFs will be normalized as though all alleles and whole pop is represented.

    Computes Pr for each site by integrating across all overlapping kmers. 

    Also returns raw p-values for each kmer/site, based on the null hypothesis that
    the HLA predictions are random. Achieved by permuting the kmer dimension of the binding matrix.
    This means that under the null two related alleles will still have related binding affinities.

    Does not test against the null that HLA binding is uniform (although that also might be interesting).

    Parameters
    ----------
    seqs : pd.Series or any collection
        Aligned amino acid sequences.
    hlas : dict or list
        Dict of HLA alleles and population frequencies.
        If list of alleles then it assumes uniform frequencies.
    ba : hlaPredCache (dict-like)
        Dictionar of predicted binding affinities (e.g. ba[(h,pep)] = prediction)
    hlaMask : ndarray [seqs x alleles]
        Mask of alleles associated with each sequence (1 = expressing, 0 = not expressing)
    bindingThreshold : float
        Threshold for what is considered a HLA binder.
    nPerms : int
        Number of permutations for creating the null distribution.
    k : int
        Length of the kmer peptide.

    Returns
    -------
    kmerPdf : ndarray [kmers]
        Probability that the kmer is a predicted epitope restricted by any HLA allele.
    kmerPvalues : ndarray [kmers]
        One-sided p-value that the observed Pr that a kmer is an epitope is greater than under the null.
    sitePdf : ndarray [kmers]
        Probability that the site is in a predicted epitope restricted by any HLA allele.
    sitePvalues : ndarray [kmers]
        One-sided p-value that the observed Pr that a site is in an epitope is greater than under the null.
    kmerNull : ndarray [kmers x perms]
        Null distribution of kmer PDFs that was used to compute the p-values.
    siteNull : ndarray [kmers x perms]
        Null distribution of site PDFs that was used to compute the p-values.

    Example
    -------
    >> kmerPdf,kmerPvalues,sitePdf,sitePvalues,kmerNull,siteNull = computeHLAHotspots(seqDf.seq, hlaFreq, ba)"""

    nPerms = int(nPerms)
    nHLAs = len(hlas)
    nSeqs = len(seqs)
    nSites = int(np.median(map(len,seqs)))
    nKmers = nSites-k+1

    if hlaMask is None:
        hlaMask = np.ones((nSeqs,nHLAs))

    hlaMaskMat = np.tile(hlaMask,(nKmers,1,1)).astype(np.float64)

    try:
        hlaList = sorted(hlas.keys())
    except AttributeError:
        hlaList = sorted(hlas)
        hlas = {h:1./len(hlaList) for h in hlaList}
    freqVec = np.array([hlas[h] for h in hlaList])
    
    baMat = getBAMat(seqs,hlas,ba)
    
    """Identify binders with a 1"""
    binders = (baMat < bindingThreshold).astype(np.float64)

    """Multiply element-wise by hla frequencies before summing (to help with permutations later)"""
    weightedBinders = binders * np.tile(freqVec.reshape((1,1,nHLAs)), (nKmers,nSeqs,1))

    """Multiply by the HLA mask"""
    weightedBinders = weightedBinders * hlaMaskMat
    
    startT = time.time()
    logger.info('Starting computation of observed PDF')
    obsPdf,obsSitePdf = _computePDF(weightedBinders,k)
    
    startT = time.time()
    logger.info('Starting permutation test')
    shuffledPr = np.zeros((nKmers, nPerms))
    
    """Only save the sites that were in k kmers (so dim=0 is nKmers not nSites)"""
    shuffledSitePr = np.zeros((nKmers,nPerms))
    for permi in range(nPerms):
        if (permi % np.floor(nPerms/10)) == 0:
            logger.info('Completed %d%% of permutations', permi + 1)
        """Shuffle the kmer dimension for each HLA, but shuffle the same for all sequences
            (hotspots induced by a dominant HLA allele will still look like hotspots in the null
            so we can est. the p-value of the hotspot correctly)"""
        shuffWB = weightedBinders.copy()
        for hlai in range(nHLAs):
            shuffWB[:,:,hlai] = shuffWB[permutation(nKmers),:,hlai]
        pdf,sitePdf = _computePDF(shuffWB,k)
        shuffledPr[:,permi] = pdf
        
        """Don't include the sites at the begining which are in < k kmers"""
        shuffledSitePr[:,permi] = sitePdf[(k-1):]
    logger.info('Completed computation in %1.1f seconds', time.time()-startT)

    """All kmers should be the same under the null so I can combine permutations across kmers"""
    shuffledPr = shuffledPr.flatten()
    """Most sites are the same under the null so I combine. The significance of a hotspot in the first few
       sites will be underestimated since they will be artificially low from being in <k kmers"""
    shuffledSitePr = shuffledSitePr.flatten()

    """One-sided pvalue that the observed Pr that a kmer is an epitope is greater than under the null"""
    pvalues = np.array([(pr<shuffledPr).sum() for pr in obsPdf],dtype = np.float64) / len(shuffledPr)
    pvalues[pvalues < (1/len(shuffledPr))] = 1/len(shuffledPr)

    sitePvalues = np.array([(pr<shuffledSitePr).sum() for pr in obsSitePdf],dtype = np.float64) / len(shuffledSitePr)
    sitePvalues[sitePvalues < (1/len(shuffledSitePr))] = 1/len(shuffledSitePr)

    kmerNull = np.reshape(shuffledPr,(nKmers,nPerms))
    siteNull = np.reshape(shuffledSitePr,(nKmers,nPerms))
    return obsPdf,pvalues,obsSitePdf,sitePvalues,kmerNull,siteNull

def _computePDF(wb,k):
    """Used by computeHLAHotspots in the permutation test.
    Parameter wb is weighted binders [kmers x seqs x hlas]
    Returns pdf, sitePdf"""

    """Sum across sequences and then across HLAs"""
    pdf = wb.sum(axis=1).sum(axis=1)

    """Sum across sites for each kmer, towards a Pr(epitope) by site"""
    sitePdf = np.convolve(pdf, np.ones(k))

    """Normalize pdf to sum to 1"""
    return pdf/pdf.sum(), sitePdf/sitePdf.sum()


def plotKmerHotpots(pdf, pvalues = None, cutoff = 0.025, pvalueCut = True, seqTicks = ''):
    xvec = np.arange(len(pdf))

    if pvalueCut and not pvalues is None:
        """Interpret cutoff as p-value alpha threshold"""
        sigInd = pvalues < cutoff
    elif not pvalues is None:
        """Interpret cutoff as a percentile"""
        tmp = sorted(pvalues)[int(np.ceil(cutoff * len(pvalues)))]
        sigInd = pvalues < tmp
    else:
        sigInd = np.zeros(pdf.shape, dtype = bool)
    
    plt.clf()
    axh = plt.gca()
    if sigInd.sum() > 0:
        mn = min(pdf[sigInd])
        """Plot cutoff line"""
        axh.plot([-1,xvec[-1]+1], [mn,mn], '--', color = 'red', lw = 2, alpha = 0.5)

    for x,y in zip(xvec,pdf):
        axh.plot([x,x], [0,y], '-', color = 'gray', alpha = 0.7)
    axh.plot(xvec[sigInd],pdf[sigInd],'o',color='red')
    axh.plot(xvec[~sigInd],pdf[~sigInd],'o',color='black')
    plt.xlabel('9mer start position')
    
    plt.xlim((-1, xvec[-1]+1))
    axh.spines['right'].set_visible(False)
    axh.spines['top'].set_visible(False)
    axh.xaxis.set_ticks_position('none')
    axh.yaxis.set_ticks_position('left')
    if len(seqTicks)>0:
        axh.xticks(np.arange(len(seqTicks)), [s for s in seqTicks])

def plotSiteHotpots(pdf, pvalues, cutoff = 0.025, pvalueCut = True):
    xvec = np.arange(len(pdf))
    if pvalueCut:
        sigInd = pvalues < cutoff
    else:
        tmp = sorted(pvalues)[int(ceil(cutoff * len(pvalues)))]
        sigInd = pvalues < tmp
    
    plt.clf()
    axh = plt.gca()
    if sigInd.sum() > 0:
        mn = min(pdf[sigInd])
        """Plot cutoff line"""
        axh.plot([-1,xvec[-1]+1], [mn,mn], '--', color = 'red', lw = 2, alpha = 0.5)
    
    axh.plot(xvec, pdf, '-', color = 'gray', alpha = 0.7)
    axh.plot(xvec[~sigInd], pdf[~sigInd], 'o',color = 'gray', alpha = 0.7)
    axh.plot(xvec[sigInd], pdf[sigInd], 'o', color = 'red')
    plt.xlabel('AA site number')
    plt.ylabel('PDF of T-cell pressure')
    plt.xlim((-1, xvec[-1]+1))
 
def plotBindingGrid(hlas, peptides, ba, annotateIC50 = True, limitToBinders = False):
    """Heatmap of IC50 for a set of HLA alleles and peptides"""
    hlas = sorted(hlas, reverse  = True)
    """Make a pd.DataFrame [hla x peptides] full of binding affinity IC50"""
    bamat = _MAX_IC50 * np.ones((len(hlas), len(peptides)))
    for hlai,h in enumerate(hlas):
        for pepi, pep in enumerate(peptides):
            bamat[hlai,pepi] = ba[(h,pep)]
    baDf = pd.DataFrame(bamat, index = hlas, columns = peptides)
    if limitToBinders:
        ind = (bamat < np.log(1000)).any(axis=1)
        baDf = baDf.iloc[ind]
    _plotBindingGridCommon(baDf.index, peptides, baDf, annotateIC50 = annotateIC50)

def plotAlignmentBindingGrid(hlas, align, ba, topN = 5, k = 9, annotateIC50 = True, limitToBinders = False, returnDf = False):
    """Heatmap of IC50 for a set of HLA alleles and all kmers in an alignment.
    The heatmap value is the minimum IC50 of the topN most common variants of that peptide."""
    hlas = sorted(hlas, reverse = True)

    nHLAs = len(hlas)
    nSeqs = len(align)
    nSites = int(np.median(map(len,align)))
    nKmers = nSites - k + 1

    """[hla,kmer]"""
    bamat = _MAX_IC50 * np.ones((nHLAs,nKmers))

    peptides=[]
    for hlai,h in enumerate(hlas):
        for meri in np.arange(bamat.shape[1]):
            seqs = _sliceAlign(align,(meri,meri+k))
            """Remove any sequences with a gap at any position"""
            seqs = [s for s in seqs if s.find('-')==-1]

            topSeqs = [s for s in objhist(seqs).sortedKeys(reverse=True)][:topN]
            tmp = np.array([ba[(h,pep)] for pep in topSeqs])
            bamat[hlai,meri] = np.nanmin(tmp)

            if hlai == 0:
                peptides.append(topSeqs[0])

    baDf = pd.DataFrame(bamat, index = hlas, columns = peptides)

    if limitToBinders:
        ind = (bamat < np.log(1000)).any(axis=1)
        baDf = baDf.iloc[ind]
    if returnDf:
        return baDf
    _plotBindingGridCommon(baDf.index, peptides, baDf, annotateIC50 = annotateIC50)

def _plotBindingGridCommon(hlas, peptides, baDf, annotateIC50):
    """Heatmap of IC50 for a set of HLA alleles and peptides"""
    qmat = baDf.values.copy()
    quantization = {4:0.0, 5:0.2, 6:0.4 , 7:0.6 , 16:0.9}

    for bound in sorted(quantization.keys(), reverse = True):
        qmat[baDf.values < bound] = quantization[bound]
    qmat[np.isnan(qmat)] = 0.9

    """Heatmap of covariation"""
    cdict = {'green'  :  ((0, 0, 0), (1, 0, 0)),
             'red':  ((0, 1, 1), (1, 0, 0)),
             'blue' :  ((0, 0, 0), (1, 0, 0))}

    """Generate the colormap with 1024 interpolated values"""
    heatCmap = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)
    #heatCmap = get_cmap('hot')

    plt.clf()
    axh = plt.subplot2grid((10,1), (1, 0), rowspan = 9)
    labelTxtProp = dict(family='monospace',size='medium',weight='bold',color='white',ha='center',va='center')

    axh.pcolor(qmat, cmap = heatCmap, vmin = 0, vmax = 1)
    if annotateIC50:
        for hlai,h in enumerate(hlas):
            for pepi, pep in enumerate(peptides):
                if baDf.values[hlai,pepi] <= 6.2:
                    plt.text(pepi+0.5,hlai+0.5,'%1.0f' % (round(np.exp(baDf.values[hlai,pepi]),-1)),**labelTxtProp)

    axh.colorbar(fraction = 0.05, values = [0,0.2,0.4,0.6,1], boundaries = [0,50,150,400,1000], label = 'IC50 (nM)')

    axh.grid(color = 'white')
    plt.xlim((0,len(peptides)))
    plt.ylim((0,len(hlas)))
    plt.yticks(np.arange(len(hlas)) + 0.5, hlas, fontname = 'Consolas')
    if len(peptides) < 50:
        axh.xaxis.tick_top()
        plt.xticks(np.arange(len(peptides))+0.5, peptides, rotation = 90, fontname = 'Consolas')
    else:
        plt.xticks(np.arange(0,len(peptides),10)+0.5, np.arange(0,len(peptides),10))

    plt.xlabel('Peptides')
    plt.ylabel('HLA alleles')

def _sliceAlign(align, region, sites=False):
    """Return a region of the alignment where region is (start, end)
    OR if sites is True then include all sites in region (not range)"""
    if region is None:
        return align
    elif sites:
        return align.map(lambda seq: ''.join([seq[r] for r in region]))
    else:
        return align.map(lambda seq: seq[region[0]:region[1]])