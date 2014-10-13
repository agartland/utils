from numpy import *
import random
from utilHelpers import unique_rows

__all__ = ['kmedoids']

def kmedoids(dmat, k=3, weights = None, nPasses = 1, maxIter=1000,initInds=None,verbose=False):
    """Identify the k points that minimize all intra-cluster distances.
    Uses Lloyd's algorithm for EM.

    The algorithm completes nPasses of the algorithm with random restarts.
    Each pass consists of iteratively assigning/improving the medoids.
    
    Parameters
    ----------
    dmat : array-like of floats, shape (n_samples, n_samples)
        The pairwise distance matrix of observations to cluster.
    weights : array-like of floats, shape (n_samples)
        Relative weights for each observation in inertia computation.
    k: int
        The number of clusters to form as well as the number of
        medoids to generate.
    maxIter : int, optional, default None (inf)
        Maximum number of iterations of the k-medoids algorithm to run.
    initInds : ndarray of shape [>=k]
        Indices used to initiate medoids in each pass.
        If None, choose from all indices.
    verbose : boolean, optional
        Verbosity mode

    Returns
    -------
    medoids: float ndarray with shape (k)
        Indices into dmat that indicate medoids found at the last iteration of k-medoids.
    labels: integer ndarray with shape (n_samples,)
        label[i] is the code or index of the medoid the
        i'th observation is closest to.
    inertia: float
        The final value of the inertia criterion (sum of squared distances to
        the closest medoid for all observations).
    nIter: int
        Number of iterations in best pass.
    nFound: int
        Number of unique solutions found (out of nPasses)
    
    To apply to points in euclidean space pass dmat using:
    dmat = sklearn.neighbors.DistanceMetric.get_metric('euclidean').pairwise(points_array)"""

    """Number of points"""
    N = dmat.shape[0]

    """Choose starting medoids from any of the points"""
    if initInds is None:
        initInds = arange(N)

    """Default weights are ones"""
    if weights is None:
        weights = ones(N)

    assert weights.shape[0]==N

    """Tile weights for multiplying by dmat"""
    tiledWeights = tile(weights[None,:],(N,1))

    """Precompute weighted squared distances"""
    wdmat2 = (dmat**2) * tiledWeights
    
    bestInertia = None
    allMedoids = zeros((nPasses,k))
    for passi in range(nPasses):
        """Pick k random medoids"""
        currMedoids = permutation(initInds)[:k]
        newMedoids = zeros(k,dtype=int32)
        for i in range(maxIter):
            """Assign each point to the closest cluster"""
            labels = currMedoids[argmin(dmat[:,currMedoids], axis=1)]

            """ISSUE: If len(unique(labels)) < k there is an error"""

            """Choose new medoids for each cluster, minimizing intra-cluster distance"""
            totInertia = 0
            if verbose:
                print "Iter %d:" % i
            for medi,med in enumerate(currMedoids):
                clusterInd = find(labels==med)
                """Inertia is the sum of the squared distances (vec is shape (len(clusterInd))"""
                inertiaVec = (wdmat2[clusterInd,:][:,clusterInd]).sum(axis=1)
                mnInd = argmin(inertiaVec)
                newMedoids[medi] = clusterInd[mnInd]
                """Add inertia of this new medoid to the running total"""
                totInertia += inertiaVec[mnInd]
                
                if verbose:
                    print '\tCurrent medoid (%1.1f): %d' % (wdmat2[med,:].sum(),med)
                    print '\tCluster size: %d' % len(clusterInd)
                    print '\tNew medoid (%1.1f): %d' % (inertiaVec[mnInd],clusterInd[mnInd])

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
    medoids,labels,inertia,niter,nfound = kmedoids(dmat,k=k,maxIter=1000,nPasses=nPasses,verbose=False)
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


 