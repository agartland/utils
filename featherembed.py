#!/usr/bin/env python

"""
Usage:

python featherembed.py --listcolumns /fh/fast/gilbert_p/grp/compass_hvtn602_aw/tmpdata/flow-data-2of9-responses.feather
python featherembed.py --columns CCR6,CCR7,CD154,CD45RA,CXCR3,GzA,HLA-DR,IFNg,IL13/4,IL17a,IL2,IL22,KLRG1,Perforin,TNFa /fh/fast/gilbert_p/grp/compass_hvtn602_aw/tmpdata/flow-data-2of9-responses.feather
"""

n_neighbors_help = """This parameter controls how UMAP balances local versus global
structure in the data. It does this by constraining the size of the
local neighborhood UMAP will look at when attempting to learn the
manifold structure of the data. This means that low values of
n_neighbors will force UMAP to concentrate on very local structure
(potentially to the detriment of the big picture), while large values
will push UMAP to look at larger neighborhoods of each point when
estimating the manifold structure of the data, loosing fine detail
structure for the sake of getting the broader of the data."""

min_dist_help = """The parameter controls how tightly UMAP is allowed to pack points
together. It, quite literally, provides the minimum distance apart
that points are allowed to be in the low dimensional representation.
This means that low values of min_dist will result in clumpier
embeddings. This can be useful if you are interested in clustering, or
in finer topological structure. Larger values of min_dist will prevent
UMAP from packing point together and will focus instead on the
preservation of the broad topological structure instead."""

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UMAP for dimensionality reduction of a matrix stored in feather format.')
    parser.add_argument('filename', type=str,
                        help='Path to the feather file.',
                        default='/fh/fast/gilbert_p/grp/compass_hvtn602_aw/tmpdata')
    parser.add_argument('--metric', type=str,
                        help='A scipy distance metric, e.g. correlation, euclidean, manhattan',
                        default='correlation')
    parser.add_argument('--out', type=str,
                        help='Out put filename.',
                        default='xxxx_out.feather')
    parser.add_argument('--n_neighbors', type=int,
                        help=n_neighbors_help,
                        default=20)
    parser.add_argument('--min_dist', type=float,
                        help=min_dist_help,
                        default=0.5)
    parser.add_argument('--columns', type=str,
                        help='Comma-sperated list of columns to consider as input dimensions',
                        default='ALL')
    parser.add_argument('--listcolumns', action='store_true', help='List the columns in the input feather file.')
    
    args = parser.parse_args()

    import numpy as np
    import pandas as pd
    import umap
    import feather
    import sys

    if args.columns != 'ALL':
        cols = args.columns.split(',')
        # cols = [c for c in cols if c in fDf.columns]
        # fDf = fDf[cols]
    else:
        cols = None
    fDf = feather.read_dataframe(args.filename, columns=cols)

    if args.listcolumns:
        print(','.join(fDf.columns))
        print('Rows: %d' % fDf.shape[0])
    else:
        umapObj = umap.UMAP(n_components=2, metric=args.metric, n_neighbors=args.n_neighbors, min_dist=args.min_dist)
        xy = umapObj.fit_transform(fDf.values)
        assert xy.shape[0] == fDf.shape[0]
        xyDf = pd.DataFrame(xy, index=fDf.index, columns=['X', 'Y'])

        if args.out == 'xxxx_out.feather':
            args.out = args.filename[:-len('.feather')] + '_out' + '.feather'

        feather.write_dataframe(xyDf, args.out)
        print('Successfully applied UMAP: %s' % args.out)