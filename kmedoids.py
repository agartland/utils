import numpy as np
from vectools import unique_rows

__all__ = ['kmedoids',
           'fuzzycmedoids',
           'precomputeWeightedSqDmat',
           'reassignClusters',
           'computeInertia',
           'computeMembership']

def kmedoids(dmat, k=3, weights=None, nPasses=1, maxIter=1000, initInds=None, potentialMedoidInds=None):
    """Identify the k points that minimize all intra-cluster distances.

    The algorithm completes nPasses of the algorithm with random restarts.
    Each pass consists of iteratively assigning/improving the medoids.
    
    Uses Partioning Around Medoids (PAM) as the EM.

    To apply to points in euclidean space pass dmat using:
    dmat = sklearn.neighbors.DistanceMetric.get_metric('euclidean').pairwise(points_array)
    
    Parameters
    ----------
    dmat : array-like of floats, shape (n_samples, n_samples)
        The pairwise distance matrix of observations to cluster.
    weights : array-like of floats, shape (n_samples)
        Relative weights for each observation in inertia computation.
    k : int
        The number of clusters to form as well as the number of
        medoids to generate.
    nPasses : int
        Number of times the algorithm is restarted with random medoid initializations. The best solution is returned.
    maxIter : int, optional, default None (inf)
        Maximum number of iterations of the k-medoids algorithm to run.
    initInds : ndarray
        Medoid indices used for random initialization and restarts for each pass.
    potentialMedoidInds : array of indices
        If specified then medoids are constrained to be chosen from this array.

    Returns
    -------
    medoids : float ndarray with shape (k)
        Indices into dmat that indicate medoids found at the last iteration of k-medoids.
    labels : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the medoid the
        i'th observation is closest to.
    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest medoid for all observations).
    nIter : int
        Number of iterations run.
    nFound : int
        Number of unique solutions found (out of nPasses)"""

    """Number of points"""
    N = dmat.shape[0]

    if initInds is None:
        initInds = np.arange(N)

    wdmat2 = precomputeWeightedSqDmat(dmat, weights)

    if not potentialMedoidInds is None:
        potentialMedoidSet = set(potentialMedoidInds)
        initInds = np.array([i for i in potentialMedoidSet.intersection(set(initInds))], dtype=int)
    else:
        potentialMedoidSet = set(np.arange(N))

    if len(initInds)==0:
        print 'No possible initInds provided.'
        return

    bestInertia = None
    allMedoids = np.zeros((nPasses,k))
    for passi in range(nPasses):
        """Pick k random medoids"""
        currMedoids = np.random.permutation(initInds)[:k]
        newMedoids = np.zeros(k, dtype=int)
        labels = currMedoids[np.random.randint(k, size=N)]
        for i in range(maxIter):
            """Assign each point to the closest cluster,
            but don't reassign a point if the distance isn't an improvement."""
            labels = reassignClusters(dmat, currMedoids, oldLabels=labels)
            
            """If clusters are lost during (re)assignment step, pick random points
            as new medoids and reassign until we have k clusters again"""
            uLabels = np.unique(labels[potentialMedoidInds])
            while uLabels.shape[0]<k:
                for medi,med in enumerate(currMedoids):
                    if not med in uLabels:
                        choices = list(set(initInds).difference(set(uLabels)))
                        currMedoids[medi] = choices[np.random.randint(len(choices))]
                        
                        labels = reassignClusters(dmat, currMedoids, oldLabels=labels)
                        uLabels = np.unique(labels[potentialMedoidInds])
                        break

            """ISSUE: If len(unique(labels)) < k there is an error"""

            """Choose new medoids for each cluster, minimizing intra-cluster distance"""
            totInertia = 0
            for medi,med in enumerate(currMedoids):
                clusterInd = np.where(labels == med)[0]
                """Limit medoids to those specified by indexing axis=0 with the intersection of potential medoids and all points in the cluster"""
                potentialInds = np.array([i for i in potentialMedoidSet.intersection(set(clusterInd))])
                """Inertia is the sum of the squared distances (vec is shape (len(clusterInd))"""
                inertiaVec = (wdmat2[potentialInds,:][:,clusterInd]).sum(axis=1)
                mnInd = np.argmin(inertiaVec)
                newMedoids[medi] = potentialInds[mnInd]
                """Add inertia of this new medoid to the running total"""
                totInertia += inertiaVec[mnInd]

            if (newMedoids == currMedoids).all():
                """If the medoids didn't need to be updated then we're done!"""
                allMedoids[passi,:] = sorted(currMedoids)
                break
            currMedoids = newMedoids.copy()
        if bestInertia is None or totInertia < bestInertia:
            """For multiple passes, see if this pass was better than the others"""
            bestInertia = totInertia
            bestMedoids = currMedoids.copy()
            bestLabels = labels.copy()
            bestNIter = i
    
    """nfound is the number of unique solutions (each row is a solution)"""
    nfound = len(unique_rows(allMedoids)[:])
    """Return the results from the best pass"""
    return bestMedoids, bestLabels, bestInertia, bestNIter, nfound

def precomputeWeightedSqDmat(dmat, weights, squared=True):
    """Compute the weighted and squared distance matrix for kmedoids.

    Optionally, do not square dmat before applying weights (for FCMdd)
    
    Adding weight to a point increases its impact on inertia linearly,
    such that the algorithm will tend to favor minimization of distances
    to that point.

    Note: weights are applied along the rows so to correctly compute
    inertia for a given medoid one would index the cluster along axis=1,
    index the medoid along axis=0 and then sum along axis=1.

    Parameters
    ----------
    dmat : ndarray shape[N x N]
        Pairwise distance matrix (unweighted).
    weights : ndarray shape[N]
        Should sum to one if one wants to preserve
        inertial units with different weights.

    Returns
    -------
    wdmat2 : ndarray shape[N x N]
        Weighted and squared distance matrix, ready for computing inertia."""
    
    N = dmat.shape[0]
    """Default weights are ones"""
    if weights is None:
        weights = np.ones(N)

    assert weights.shape[0]==N

    """Tile weights for multiplying by dmat"""
    tiledWeights = np.tile(weights[None,:], (N,1))

    """Precompute weighted squared distances"""
    if squared:
        wdmat2 = (dmat**2) * tiledWeights
    else:
        wdmat2 = dmat * tiledWeights
    return wdmat2

def reassignClusters(dmat, currMedoids, oldLabels=None):
    """Assigns/reassigns points to clusters based on the minimum (unweighted) distance.
    
    Note: if oldLabels are specified then only reassigns points that
    are not currently part of a cluster that minimizes their distance.
    
    This ensures that when there are ties for best cluster with the current cluster,
    the point is not reassigned to a new cluster.

    Parameters
    ----------
    dmat : ndarray shape[N x N]
        Pairwise distance matrix (unweighted).
    currMedoids : ndarray shape[k]
        Index into points/dmat that specifies the k current medoids.
    oldLabels : ndarray shape[N]
        Old labels that will be reassigned.

    Returns
    -------
    labels : ndarray shape[N]
        New labels such that unique(labels) equals currMedoids."""

    N = dmat.shape[0]
    k = len(currMedoids)

    """Assign each point to the closest cluster,
    but don't reassign a point if the distance isn't an improvement."""
    if not oldLabels is None:
        labels = oldLabels
        oldD = dmat[np.arange(N), labels]
        minD = (dmat[:,currMedoids]).min(axis=1)
        """Points where reassigning is neccessary"""
        reassignInds = (minD<oldD) | ~np.any(np.tile(labels[:,None],(1,k)) == np.tile(currMedoids[None,:],(N,1)),axis=1)
    else:
        reassignInds = np.arange(N)
        labels = np.zeros(N)
    labels[reassignInds] = currMedoids[np.argmin(dmat[reassignInds,:][:,currMedoids], axis=1)]
    return labels

def computeInertia(wdmat2, labels, currMedoids):
    """Computes inertia for a set of clustered points using
    a precomputed weighted and squared distance matrix.
    
    Note: wdmat2 needs to be summed along axis=1

    assert all(sorted(unique(labels)) == sorted(currMedoids))

    Parameters
    ----------
    wdmat2 : ndarray shape[N x N]
        Weighted and squared distance matrix, ready for computing inertia.
    labels : ndarray shape[N]
        The cluster assignment (medoid index) of each point
    currMedoids : ndarray shape[k]
        Index into points/dmat that specifies the k current medoids.

    Returns
    -------
    inertia : float
        Total inertia of all k clusters
    """
    assert np.all(np.unique(labels) == np.unique(currMedoids))
    
    totInertia = 0
    for medi,med in enumerate(currMedoids):
        clusterInd = np.where(labels == med)[0]
        """Inertia is the sum of the squared distances"""
        totInertia += wdmat2[med,clusterInd].sum()
    return totInertia

def fuzzycmedoids(dmat, c=3, weights=None, nPasses=1, maxIter=1000, initInds=None, potentialMedoidInds=None):
    """Implementation of fuzz c-medoids (FCMdd)

    Krishnapuram, Raghu, Anupam Joshi, Liyu Yi, Computer Sciences, and Baltimore County.
        "A Fuzzy Relative of the K-Medoids Algorithm with Application to Web Documen."
        Electrical Engineering. doi:10.1109/FUZZY.1999.790086.

    The algorithm completes nPasses of the algorithm with random restarts.
    Each pass consists of iteratively assigning/improving the medoids.
    
    To apply to points in euclidean space pass dmat using:
    dmat = sklearn.neighbors.DistanceMetric.get_metric('euclidean').pairwise(points_array)
    
    Parameters
    ----------
    dmat : array-like of floats, shape (n_samples, n_samples)
        Pairwise distance matrix of observations to cluster.
    weights : array-like of floats, shape (n_samples)
        Relative weights for each observation in inertia computation.
    c : int
        The number of clusters to form as well as the number of medoids to generate.
    nPasses : int
        Number of times the algorithm is restarted with random medoid initializations.
        The best solution is returned.
    maxIter : int, optional, default None (inf)
        Maximum number of iterations of the c-medoids algorithm to run.
    initInds : ndarray
        Medoid indices used for random initialization and restarts for each pass.
    potentialMedoidInds : array of indices
        If specified, then medoids are constrained to be chosen from this array.

    Returns
    -------
    medoids : float ndarray with shape (c)
        Indices into dmat that indicate medoids found at the last iteration of FCMdd
    membership : float ndarray with shape (n_samples, c)
        Each row contains the membership of a point to each of the clusters.
    nIter : int
        Number of iterations run.
    nFound : int
        Number of unique solutions found (out of nPasses)"""

    """Number of points"""
    N = dmat.shape[0]

    if initInds is None:
        initInds = np.arange(N)

    wdmat = precomputeWeightedSqDmat(dmat, weights, squared=False)

    if not potentialMedoidInds is None:
        initInds = np.array([i for i in initInds if i in potentialMedoidInds], dtype=int)
    else:
        potentialMedoidInds = np.arange(N)

    if len(initInds) == 0:
        print 'No possible initInds provided.'
        return

    allMedoids = np.zeros((nPasses, c))
    bestInertia = None
    for passi in range(nPasses):
        """Pick c random medoids"""
        currMedoids = np.random.permutation(initInds)[:c]
        newMedoids = np.zeros(c, dtype=int)
        for i in range(maxIter):
            """(Re)compute memberships [N x c]"""
            membership = computeMembership(dmat, currMedoids)
            
            """Choose new medoid for each cluster, minimizing fuzzy objective function"""
            totInertia = 0
            for medi,med in enumerate(currMedoids):
                """Within each cluster find the new medoid
                by minimizing the dissimilarities,
                weighted by membership to the cluster"""

                """Inertia is the sum of the membership times the distance matrix over all points.
                (membership for cluster medi [a column vector] is applied across the columns of wdmat
                [and broadcast to all row vectors] before summing)"""
                inertiaVec = (membership[:,medi][:,None].T * wdmat[potentialMedoidInds,:]).sum(axis=1)
                mnInd = np.argmin(inertiaVec)
                newMedoids[medi] = potentialMedoidInds[mnInd]
                """Add inertia of this new medoid to the running total"""
                totInertia += inertiaVec[mnInd]

            if (newMedoids == currMedoids).all():
                """If the medoids didn't need to be updated then we're done!"""
                allMedoids[passi,:] = sorted(currMedoids)
                break
            currMedoids = newMedoids.copy()
        if bestInertia is None or totInertia < bestInertia:
            """For multiple passes, see if this pass was better than the others"""
            bestInertia = totInertia
            bestMedoids = currMedoids.copy()
            bestMembership = membership.copy()
            bestNIter = i
    
    """nfound is the number of unique solutions (each row is a solution)"""
    nfound = len(unique_rows(allMedoids)[:])
    """Return the results from the best pass"""
    return bestMedoids, bestMembership, bestNIter, nfound

def computeMembership(dmat, medoids, param=2, method='FCM'):
    """Compute membership of each instance in each cluster,
    defined by the provided medoids.

    Possible methods come from the manuscript by Krishnapuram et al.
    and may include an additional parameter (typically a "fuzzifier")

    Parameters
    ----------
    dmat : ndarray shape[N x N]
        Pairwise distance matrix (unweighted).
    medoids : ndarray shape[c]
        Index into points/dmat that specifies the c current medoids.
    method : str
        Method for computing memberships:
            "FCM" (from fuzzy c-means)
            Equations from Krishnapuram et al. (2, 3, 4 or 5)
    param : float
        Additional parameter required by the method.
        Note: param must be shape[c,] for methods 4 or 5

    Returns
    -------
    membership : ndarray shape[N x c]
        New labels such that unique(labels) equals currMedoids."""

    N = dmat.shape[0]
    c = len(medoids)

    r = dmat[:,medoids]

    if method in ['FCM', 2, '2']:
        assert param >= 1
        tmp = (1 / r)**(1 / (param - 1))
    elif method in [3, '3']:
        assert param > 0
        tmp = np.exp(-param * r)
    elif method in [4, '4']:
        assert param.shape == (c,)
        tmp = 1/(1 + r/param[:,None])
    elif method in [5, '5']:
        assert param.shape == (c,)
        tmp = np.exp(-r/param[:,None])

    membership = tmp / tmp.sum(axis=1, keepdims=True)
    return membership

def _rangenorm(vec, mx=1, mn=0):
    """Normazlize values of vec in-place to [mn, mx] interval"""
    vec = vec - np.nanmin(vec)
    vec = vec / np.nanmax(vec)
    vec = vec * (mx-mn) + mn
    return vec

def _test_kmedoids(nPasses=20):
    from sklearn import neighbors, datasets
    import brewer2mpl
    from Bio.Cluster import kmedoids as biokmedoids
    import time
    import matplotlib.pyplot as plt
    import palettable

    iris = datasets.load_iris()
    obs = iris['data']

    dmat = neighbors.DistanceMetric.get_metric('euclidean').pairwise(obs)
    weights = np.random.rand(obs.shape[0])
    k = 3

    cmap = palettable.colorbrewer.qualitative.Set1_9.mpl_colors
    #cmap = brewer2mpl.get_map('set1','qualitative',min([max([3,k]),9])).mpl_colors

    plt.figure(2)
    plt.clf()
    plt.subplot(2,2,1)
    startTime = time.time()
    medoids,labels,inertia,niter,nfound = kmedoids(dmat, k=k, maxIter=1000, nPasses=nPasses)
    et = time.time() - startTime
    for medi,med in enumerate(medoids):
        plt.scatter(obs[labels==med,0],obs[labels==med,1],color=cmap[medi])
        plt.plot(obs[med,0],obs[med,1],'sk',markersize=10,color=cmap[medi], alpha=0.5)
    plt.title('K-medoids (%1.3f sec, %d iterations, %d solns)' % (et,niter,nfound))

    plt.subplot(2,2,3)
    startTime = time.time()
    medoids,labels,inertia,niter,nfound = kmedoids(dmat, k=k, maxIter=1000, nPasses=nPasses, weights=weights)
    et = time.time() - startTime
    for medi,med in enumerate(medoids):
        nWeights = _rangenorm(weights,mn=10,mx=200)
        plt.scatter(obs[labels==med,0], obs[labels==med,1], color=cmap[medi], s=nWeights, edgecolor='black', alpha=0.5)
        plt.plot(obs[med,0], obs[med,1], 'sk', markersize=10, color=cmap[medi])
    plt.title('Weighted K-medoids (%1.3f sec, %d iterations, %d solns)' % (et,niter,nfound))

    plt.subplot(2,2,2)
    startTime = time.time()
    biolabels, bioerror, bionfound = biokmedoids(dmat, nclusters=k, npass=nPasses)
    biomedoids = np.unique(biolabels)
    bioet = time.time() - startTime
    for medi,med in enumerate(biomedoids):
        plt.scatter(obs[biolabels==med,0], obs[biolabels==med,1], color=cmap[medi])
        plt.plot(obs[med,0], obs[med,1],'sk', color=cmap[medi], markersize=10, alpha = 0.5)
    plt.title('Bio.Cluster K-medoids (%1.3f sec, %d solns)' % (bioet,bionfound))

    plt.subplot(2,2,4)
    startTime = time.time()
    medoids,membership,niter,nfound = fuzzycmedoids(dmat, c=k, maxIter=1000, nPasses=nPasses)
    labels = medoids[np.argmax(membership, axis=1)]
    et = time.time() - startTime
    
    for medi,med in enumerate(medoids):
        plt.scatter(obs[labels==med,0], obs[labels==med,1], color=cmap[medi])
        plt.plot(obs[med,0], obs[med,1], 'sk', markersize=10, color=cmap[medi], alpha=0.5)
    plt.title('Fuzzy c-medoids (%1.3f sec, %d iterations, %d solns)' % (et,niter,nfound))

