from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
import numpy as np

# Adapted from https://github.com/mwburke/cluster-stability/blob/master/cluster_stability.py
__all__ = ['align_clusters']

def align_clusters(labels_v1, labels_v2):
    """
    Use the Hungarian (Kuhn–Munkres) algorithm to align two sets of cluster
    labels for the same dataset as phrased as a combinational optimziation
    algorithm that runs in O(n^3) time. Recreates the cluster labels for
    both versions of the cluster labels and returns them.

    Parameters
    ----------
    labels_v1 : int array : shape=[num_cluster_items]
        Labels for the first version of the cluster data
    labels_v2 : int array : shape=[num_cluster_items]
        Labels for the second version of the cluster data

    Returns
    -------
    new_labels_v2 : 
        Re-mapped labels such that they use the same labels in v1 and have mapped
        for optimal alignment.
    """

    """Here the labels are turned into integers and this mapping is retained for mapping back
    into the original labels using vX_unmapper"""
    contingency, mapped_labels_v1, mapped_labels_v2 = contingency_matrix(labels_v1, labels_v2)
    
    """Turn the contingency table into a cost and minimize with scipy"""
    contingency = contingency * -1
    assignments = linear_sum_assignment(contingency)

    v1_unmapper = {i:s for i,s in zip(mapped_labels_v1, labels_v1)}
    v2_unmapper = {i:s for i,s in zip(mapped_labels_v2, labels_v2)}
    
    mapper_v2_to_v1 = {v2:v1 for v1, v2 in zip(assignments[0], assignments[1])}
    # mapper_v1_to_v2 = {v1:v2 for v1, v2 in zip(assignments[0], assignments[1])}

    """Two steps: use assignment for v2 to get the closest v1 cluster (integer labels),
    then unmap back into labels using the v1 unmapper"""
    """out = []
    for v in mapped_labels_v2:
        v1_lab = mapper_v2_to_v1.get(v, v2_unmapper[v])
        newv2 = v1_unmapper[v1_lab]
        out.append(newv2)"""
    new_labels_v2 = np.array([v1_unmapper.get(mapper_v2_to_v1.get(v, v2_unmapper[v]), v2_unmapper[v]) for v in mapped_labels_v2])

    return new_labels_v2

def map_labels(label_map, labels):
    return [label_map[label] for label in labels]


def contingency_matrix(clus1_labels, clus2_labels, eps=None):
    """
    Taken from this public implementation of the Munkres/Kuhns algorithm
    under the Apache 2 license.
    Can be found here: https://github.com/bmc/munkres

    Build a contengency matrix describing the relationship between labels.
    Parameters
    ----------
    clus1_labels : int array, shape = [n_samples]
        Ground truth class labels to be used as a reference
    clus2_labels : array, shape = [n_samples]
        Cluster labels to evaluate
    eps: None or float
        If a float, that value is added to all values in the contingency
        matrix. This helps to stop NaN propogation.
        If ``None``, nothing is adjusted.

    Returns
    -------
    contingency: array, shape=[n_classes_true, n_classes_pred]
        Matrix :math:`C` such that :math:`C_{i, j}` is the number of samples in
        true class :math:`i` and in predicted class :math:`j`. If
        ``eps is None``, the dtype of this array will be integer. If ``eps`` is
        given, the dtype will be float.
    class_idx: array, shape=[n_samples]
        Array of class labels with new mappings from from 0..n_classes_true
    clusters_idx: array, shape=[n_samples]
        Array of class labels with new mappings from from 0..n_classes_pred
    """
    classes, class_idx = np.unique(clus1_labels, return_inverse=True)
    clusters, cluster_idx = np.unique(clus2_labels, return_inverse=True)
    n_classes = classes.shape[0]
    n_clusters = clusters.shape[0]
    # Using coo_matrix to accelerate simple histogram calculation,
    # i.e. bins are consecutive integers
    # Currently, coo_matrix is faster than histogram2d for simple cases
    contingency = coo_matrix((np.ones(class_idx.shape[0]),
                            (class_idx, cluster_idx)),
                            shape=(n_classes, n_clusters),
                            dtype=int).toarray()
    if eps is not None:
        # don't use += as contingency is integer
        contingency = contingency + eps
    return contingency, class_idx, cluster_idx