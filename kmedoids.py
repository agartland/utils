from numpy import *
from numpy.random import permutation,randint
from utilHelpers import unique_rows

__all__ = ['kmedoids',
           'precomputeWeightedSqDmat',
           'reassignClusters',
           'computeInertia']

def kmedoids(dmat, k=3, weights = None, nPasses = 1, maxIter=1000,initInds=None):
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
        Number of unique solutions found (out of nPasses)
    """

    """Number of points"""
    N = dmat.shape[0]

    if initInds is None:
        initInds = arange(N)

    wdmat2 = precomputeWeightedSqDmat(dmat,weights)

    bestInertia = None
    allMedoids = zeros((nPasses,k))
    for passi in range(nPasses):
        """Pick k random medoids"""
        currMedoids = permutation(initInds)[:k]
        newMedoids = zeros(k,dtype=int32)
        labels = currMedoids[randint(k,size=N)]
        for i in range(maxIter):
            """Assign each point to the closest cluster,
            but don't reassign a point if the distance isn't an improvement."""
            labels = reassignClusters(dmat,currMedoids,oldLabels=labels)
            
            """If clusters are lost during (re)assignment step, pick random points
            as new medoids and reassign until we have k clusters again"""
            uLabels = unique(labels)
            while uLabels.shape[0]<k:
                for medi,med in enumerate(currMedoids):
                    if not med in uLabels:
                        choices = list(set(initInds).difference(set(uLabels)))
                        currMedoids[medi] = choices[randint(len(choices))]
                        
                        labels = reassignClusters(dmat,currMedoids,oldLabels=labels)
                        uLabels = unique(labels)
                        break

            """ISSUE: If len(unique(labels)) < k there is an error"""

            """Choose new medoids for each cluster, minimizing intra-cluster distance"""
            totInertia = 0
            for medi,med in enumerate(currMedoids):
                clusterInd = where(labels==med)[0]
                """Inertia is the sum of the squared distances (vec is shape (len(clusterInd))"""
                inertiaVec = (wdmat2[clusterInd,:][:,clusterInd]).sum(axis=1)
                mnInd = argmin(inertiaVec)
                newMedoids[medi] = clusterInd[mnInd]
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

def precomputeWeightedSqDmat(dmat,weights):
    """Compute the weighted and squared distance matrix for kmedoids.
    
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
        Weighted and squared distance matrix, ready for computing inertia.
    """
    N = dmat.shape[0]
    """Default weights are ones"""
    if weights is None:
        weights = ones(N)

    assert weights.shape[0]==N

    """Tile weights for multiplying by dmat"""
    tiledWeights = tile(weights[None,:],(N,1))

    """Precompute weighted squared distances"""
    wdmat2 = (dmat**2) * tiledWeights
    return wdmat2

def reassignClusters(dmat,currMedoids,oldLabels=None):
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
        New labels such that unique(labels) equals currMedoids.
    """
    N = dmat.shape[0]
    k = len(currMedoids)

    """Assign each point to the closest cluster,
    but don't reassign a point if the distance isn't an improvement."""
    if not oldLabels is None:
        labels = oldLabels
        oldD = dmat[arange(N),labels]
        minD = (dmat[:,currMedoids]).min(axis=1)
        """Points where reassigning is neccessary"""
        reassignInds = (minD<oldD) | ~any(tile(labels[:,None],(1,k))==tile(currMedoids[None,:],(N,1)),axis=1)
    else:
        reassignInds = arange(N)
        labels = zeros(N)
    #print unique(labels).shape[0],sorted(unique(labels)),sorted(currMedoids)
    #print reassignInds.sum(),currMedoids[argmin(dmat[reassignInds,:][:,currMedoids], axis=1)]
    labels[reassignInds] = currMedoids[argmin(dmat[reassignInds,:][:,currMedoids], axis=1)]
    #print unique(labels).shape[0],sorted(unique(labels)),sorted(currMedoids)
    return labels

def computeInertia(wdmat2,labels,currMedoids):
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
    assert all(sorted(unique(labels)) == sorted(currMedoids))
    
    totInertia = 0
    for medi,med in enumerate(currMedoids):
        clusterInd = where(labels==med)[0]
        """Inertia is the sum of the squared distances"""
        totInertia += wdmat2[med,clusterInd].sum()
    return totInertia

def _test_kmedoids(nPasses=20):
    from sklearn import neighbors, datasets
    import brewer2mpl
    from Bio.Cluster import kmedoids as biokmedoids
    import time
    from seqtools import mynorm

    iris = datasets.load_iris()
    obs = iris['data']

    dmat = neighbors.DistanceMetric.get_metric('euclidean').pairwise(obs)
    weights = rand(obs.shape[0])
    k = 3

    cmap = brewer2mpl.get_map('set1','qualitative',min([max([3,k]),9])).mpl_colors

    figure(2)
    clf()
    subplot(2,2,3)
    startTime = time.time()
    medoids,labels,inertia,niter,nfound = kmedoids(dmat,k=k,maxIter=1000,nPasses=nPasses,weights=weights)
    et = time.time()-startTime
    for medi,med in enumerate(medoids):
        scatter(obs[labels==med,0],obs[labels==med,1],color=cmap[medi],s=mynorm(weights,mn=10,mx=200),edgecolor='black',alpha=0.5)
        plot(obs[med,0],obs[med,1],'sk',markersize=10,color=cmap[medi])
    title('Weighted K-medoids (%1.3f sec, %d iterations, %d solns)' % (et,niter,nfound))

    subplot(2,2,1)
    startTime = time.time()
    medoids,labels,inertia,niter,nfound = kmedoids(dmat,k=k,maxIter=1000,nPasses=nPasses)
    et = time.time()-startTime
    for medi,med in enumerate(medoids):
        scatter(obs[labels==med,0],obs[labels==med,1],color=cmap[medi])
        plot(obs[med,0],obs[med,1],'sk',markersize=10,color=cmap[medi],alpha = 0.5)
    title('K-medoids (%1.3f sec, %d iterations, %d solns)' % (et,niter,nfound))

    subplot(2,2,2)
    startTime = time.time()
    biolabels, bioerror, bionfound = biokmedoids(dmat,nclusters = k,npass=nPasses)
    biomedoids = unique(biolabels)
    bioet = time.time()-startTime
    for medi,med in enumerate(biomedoids):
        scatter(obs[biolabels==med,0],obs[biolabels==med,1],color=cmap[medi])
        plot(obs[med,0],obs[med,1],'sk',color=cmap[medi],markersize=10,alpha = 0.5)
    title('Bio.Cluster K-medoids (%1.3f sec, %d solns)' % (bioet,bionfound))


 