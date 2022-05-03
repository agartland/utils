import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import feather
from os.path import join as opj
import sys
import time
import sklearn
import itertools
from scipy.stats import gaussian_kde

def mnmx(v, margin=0.02):
    mn, mx = np.min(v), np.max(v)
    mx += margin * (mx - mn)
    mn -= margin * (mx - mn)
    return mn, mx

def scipy_kde(grp, weights=None, bw=0.25, positions=None, npoints=100, xrange=[], yrange=[]):
    if grp.shape[0] < 10:
        Z = np.zeros((npoints, npoints))
        return Z
    if positions is None:
        X, Y = np.mgrid[xrange[0]:xrange[1]:complex(0, npoints), yrange[0]:yrange[1]:complex(0, npoints)]
        positions = np.vstack([X.ravel(), Y.ravel()])
    
    data = np.vstack([grp[0].values, grp[1].values])
    kde = gaussian_kde(data, bw_method=bw, weights=weights)
    Z = np.reshape(kde.evaluate(positions), (npoints, npoints))
    return Z

def sk_kde(grp, bw=0.25, positions=None, npoints=100, xrange=[], yrange=[]):
    kdeP = dict(bandwidth=bw,
            metric='euclidean',
            kernel='gaussian',
            algorithm='ball_tree',
            rtol=1e-5)
    if positions is None:
        X, Y = np.mgrid[xrange[0]:xrange[1]:complex(0, npoints), yrange[0]:yrange[1]:complex(0, npoints)]
        positions = np.vstack([X.ravel(), Y.ravel()])
    
    kde = sklearn.neighbors.KernelDensity(**kdeP)
    kde.fit(grp[[0, 1]].values)
    Z = np.reshape(kde.score_samples(positions.T), (npoints, npoints))
    return np.exp(Z)

def gby_densities(xy_df, gby_cols, bw=0.25, npoints=100):
    xmin, xmax = mnmx(xy_df[0])
    ymin, ymax = mnmx(xy_df[1])

    """Parameters for KDE"""
    # d = 2
    # bw = (n * (d + 2) / 4.)**(-1. / (d + 4)) # silverman
    # bw = n**(-1./(d+4)) # scott
    X, Y = np.mgrid[xmin:xmax:complex(0, npoints), ymin:ymax:complex(0, npoints)]
    positions = np.vstack([X.ravel(), Y.ravel()])

    """Compute the histogram for each group"""
    tot = {}
    xy = {}
    sk_kdes = {}
    kdes = {}
    counter = 0
    for i, grp in xy_df.groupby(gby_cols):
        tot[i] = grp.shape[0]
        xy[i] = grp[[0, 1]].values
        # sk_kdes[i] = pctpos * sk_kde(grp)
        kdes[i] = scipy_kde(grp, positions=positions)
    return X, Y, kdes

def plot_grid(axh, xmin=None, xmax=None, ymin=None, ymax=None, n=3, color='w', lw=1):
    if None in [xmin, xmax, ymin, ymax]:
        xmin, xmax = axh.get_xlim()
        ymin, ymax = axh.get_ylim()
    xpos = np.linspace(xmin, xmax, n+2)
    ypos = np.linspace(ymin, ymax, n+2)

    for x in xpos[1:-1]:
        axh.plot([x, x], [ymin, ymax], '-', color=color, linewidth=lw, alpha=0.5)
    for y in ypos[1:-1]:
        axh.plot([xmin, xmax], [y, y], '-', color=color, linewidth=lw, alpha=0.5)
    axh.set_xticks(())
    axh.set_yticks(())
    axh.set_ylim((ymin, ymax))
    axh.set_xlim((xmin, xmax))