import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from scipy import sparse
import itertools

def quasi_sharing(mdf, pwmat, index_col, groupby_cols, radius=0):
    """Count number of shared clones among all pairs of groups
    defined by the unique values in groupby_cols. Uses a distance-based definition
    of sharing with all distances between pairs of TCRs in mdf (D_ij), stored in pwmat
    (optionally sparse) and with a sharing definition of D_ij <= radius.
    
    Parameters
    ----------
    mdf : pd.DataFrame
        A TCRRep.clone_df with one row per clonotype. Must contain meta-data to group on.
    pwmat : np.ndarray
        Pairwise distance matrix, optionally a scipy.sparse.csr_matrix.
        Must have same shape as rows in mdf
    index_col : str
        Column in mdf of integers that can be used to index pwmat
    groupby_cols : list
        Columns in mdf to make unique groups from
    radius : float
        Minimum threshold to define a distance that defines a shared TCR

    Returns
    -------
    share_cts : pd.DataFrame
        Matrix of the number of shared TCRs between each pair of groups. Groups
        are represented along the column index and row index as a MultiIndex.
    share_prop : pd.DataFrame
        Identically shaped matrix to share_cts indicating the proportion of TCRs shared
        using the column group as the denominator.
    share_idx : pd.DataFrame
        Identically shaped matrix to share_cts indicating the sharing index
        which is the proportion of shared TCRs (n1 + n2) / (N1 + N2) where
        nx is the number shared from group x and Nx is the total number.
        This index is symetric.
    """

    """Initialize the sharing matrix"""
    if len(groupby_cols) == 1:
        indices = [(i, ) for i,gby in mdf.groupby(groupby_cols)]    
    else:
        indices = [i for i,gby in mdf.groupby(groupby_cols)]
    mi = pd.MultiIndex.from_tuples(indices, names=groupby_cols)
    share_cts = pd.DataFrame(np.nan * np.zeros((len(indices), len(indices))), columns=mi, index=mi)
    #row_cts = pd.DataFrame(np.nan * np.zeros((len(indices), len(indices))), columns=mi, index=mi)
    #col_cts = pd.DataFrame(np.nan * np.zeros((len(indices), len(indices))), columns=mi, index=mi)
    tot_cts = pd.DataFrame(np.nan * np.zeros((len(indices), len(indices))), columns=mi, index=mi)
    share_idx = pd.DataFrame(np.nan * np.zeros((len(indices), len(indices))), columns=mi, index=mi)

    # shared = []
    """for i1, gby1 in mdf.groupby(groupby_cols):
        row_cts.loc[i1, :] = gby1.shape[0]
        col_cts.loc[:, i1] = gby1.shape[0]"""

    for (i1, gby1), (i2, gby2) in itertools.combinations(mdf.groupby(groupby_cols), 2):
        submat = pwmat[gby1[index_col], :][:, gby2[index_col]]
        if sparse.issparse(submat):
            submat = np.asarray(submat.todense())
            submat[submat == 0] = radius + 1
        #if i1[0]==421400291 and i2[0]==421400291:
        #    raise

        share_cts.loc[i1, i1] = gby1.shape[0]
        share_cts.loc[i2, i2] = gby2.shape[0]

        if i1 > i2:
            share_cts.loc[i1, i2] = (submat <= radius).any(axis=1).sum()
            tot_cts.loc[i1, i2] = gby1.shape[0]

            share_cts.loc[i2, i1] = (submat <= radius).any(axis=0).sum()
            tot_cts.loc[i2, i1] = gby2.shape[0]
        else:
            share_cts.loc[i2, i1] = (submat <= radius).any(axis=1).sum()
            tot_cts.loc[i2, i1] = gby1.shape[0]

            share_cts.loc[i1, i2] = (submat <= radius).any(axis=0).sum()
            tot_cts.loc[i1, i2] = gby2.shape[0]

        sidx = ((submat <= radius).any(axis=1).sum() + (submat <= radius).any(axis=0).sum()) / (gby1.shape[0] + gby2.shape[0])
        share_idx.loc[i1, i2] = sidx
        share_idx.loc[i2, i1] = sidx

    share_prop = share_cts / tot_cts.values
    share_prop.values[np.diag_indices_from(share_prop)] = 1
    share_idx.values[np.diag_indices_from(share_idx)] = 1
    return share_cts, share_prop, share_idx


def plot_quasi_sharing(sharing_matrix, row_linkage=None, col_linkage=None, color_lut={}, figsize=(10,10), legend_cols=None, vmin=None, vmax=None):
    if legend_cols is None:
        legend_cols = sharing_matrix.index.names
    def lookup_add(key):
        if key in color_lut:
            return color_lut[key]
        else:
            color_lut.update({key:plt.cm.Set3.colors[np.random.randint(12)]})
            return color_lut[key]
    color_df = sharing_matrix.index.to_frame().applymap(lookup_add)
    cmobj = sns.clustermap(sharing_matrix,
                           yticklabels=True, xticklabels=True,
                           row_linkage=row_linkage,
                           col_linkage=col_linkage,
                           figsize=figsize,
                           row_colors=color_df,
                           dendrogram_ratio=0.1,
                           linewidth=0.1,
                           cmap='magma_r',
                           vmin=vmin,
                           vmax=vmax)
    if len(legend_cols) > 0:
        uvals = set(sharing_matrix.index.to_frame()[legend_cols].values.flatten())
        keys = [k for k in list(color_lut.keys()) if k in uvals]
        handles = [mpl.patches.Patch(facecolor=lut[name]) for name in keys]
        plt.legend(handles, keys,
                   bbox_to_anchor=(1., 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    return cmobj

"""These only apply to binary sharing, so don't apply directly to fuzzy matching which is not symetrical"""
def overlap_similarity(share_cts, row_cts, col_cts):
    return share_cts/((row_cts.values + col_cts.values)/2)

def jaccard_similarity(share_cts, row_cts, col_cts):
    return share_cts / (row_cts.values + col_cts.values - share_cts.values)