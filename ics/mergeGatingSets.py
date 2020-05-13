#!/usr/bin/env python

"""
Usage examples:
python /home/agartlan/gitrepo/utils/ics/mergeGatingSets.py --function functions --ncpus 4 --out functions_extract.csv

sbatch -n 1 -t 3-0 -c 4 -o functions_slurm.txt --wrap="python /home/agartlan/gitrepo/utils/ics/mergeGatingSets.py --function functions --ncpus 4 --out functions_extract.csv"
sbatch -n 1 -t 3-0 -c 4 -o functions_markers_slurm.txt --wrap="python /home/agartlan/gitrepo/utils/ics/mergeGatingSets.py --function functions_markers --ncpus 4 --out functions_markers_extract.csv"

sbatch -n 1 -t 3-0 -c 4 -o functions_markers_sparse_slurm_gby.txt --wrap="python /home/agartlan/gitrepo/utils/ics/mergeGatingSets.py --function sparse_functions --ncpus 4 --subsets /home/agartlan/gitrepo/utils/ics/allcombs_subsets.csv --out functions_markers_sparse_24Jul2018_gby.csv"

sbatch -n 1 -t 3-0 -c 4 -o cell_functions_slurm.txt --wrap="python /home/agartlan/gitrepo/utils/ics/mergeGatingSets.py --function cell_functions --ncpus 4 --out cell_functions_22Aug2018.feather --feather --subsets /home/agartlan/gitrepo/utils/ics/subsets_CD4_gd_Tcells.csv"

python /home/agartlan/gitrepo/utils/ics/mergeGatingSets.py --function cell_functions --ncpus 3 --out cell_functions_extract.csv --testbatch --testsamples --feather --subsets /home/agartlan/gitrepo/utils/ics/subsets_CD4_gd_Tcells.csv

python /home/agartlan/gitrepo/utils/ics/mergeGatingSets.py --function sparse_functions --ncpus 3 --out sparse_functions_extract_23Aug2018.csv --testbatch --testsamples --feather --subsets /home/agartlan/gitrepo/utils/ics/subsets_CD4_gd_Tcells.csv

python /home/agartlan/gitrepo/utils/ics/mergeGatingSets.py --function bool_functions --ncpus 6 --out bool_functions_extract_05May2020.csv --testbatch --testsamples --feather --subsets /home/agartlan/gitrepo/utils/ics/subsets_CD4_gd_Tcells.csv



To delete all tmp files use:
find . -name \merged_tmp*.feather -type f -delete
"""


def mergeBatches(dataFolder, extractionFunc, extractionKwargs, ncpus, testsamples, testbatch, outFile, metaCols=None, filters=None, useFeather=False):
    out = []
    batchList = [opj(dataFolder, bf) for bf in os.listdir(dataFolder) if os.path.isdir(opj(dataFolder, bf))]
    if testbatch:
        batchList = batchList[:1]
    matchStr = 'gs_*.feather'
    if ncpus > 1 and _PARMAP:
        res = parmap.map(mergeSamples,
                             batchList,
                             extractionFunc,
                             extractionKwargs,
                             matchStr,
                             testsamples,
                             metaCols,
                             filters,
                             pool=Pool(processes=ncpus))
    else:
        if _PARMAP:
            res = parmap.map(mergeSamples,
                             batchList,
                             extractionFunc,
                             extractionKwargs,
                             matchStr,
                             testsamples,
                             metaCols,
                             filters,
                             parallel=False)
        else:
            func = partial(mergeSamples,
                           extractionFunc=extractionFunc,
                           extractionKwargs=extractionKwargs,
                           matchStr=matchStr,
                           test=testsamples,
                           metaCols=metaCols,
                           filters=filters)
            res = list(map(func, batchList))
    
    outFilename = mergeFeathers(res, outFile, writeCSV=1 - int(useFeather))
    return outFilename

def testMatching(dataFolder):
    out = []
    for bf in os.listdir(dataFolder):
        batchFolder = opj(dataFolder, bf)
        if os.path.isdir(opj(dataFolder, bf)):
            featherLU = matchSamples(batchFolder, test=False)
            tmp = pd.Series(featherLU).to_frame()
            tmp.loc[:, 'batch'] = bf
            tmp.loc[:, 'batch_folder'] = opj(dataFolder, bf)
            out.append(tmp)
    return pd.concat(out, axis=0)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract features and merge batches into one CSV.')
    parser.add_argument('--folder', type=str,
                        help='Data folder containing all batch folders.',
                        default='/fh/fast/gilbert_p/grp/hvtn602_compass/tmpdata')
    parser.add_argument('--function', type=str,
                        help='Name of extraction to apply ("functions")',
                        default='functions')
    parser.add_argument('--subsets', type=str,
                        help='Filename listing subsets for analysis.',
                        default='/home/agartlan/gitrepo/utils/ics/sample_subsets2.csv')
    parser.add_argument('--out', type=str,
                        help='Output filename for CSV.',
                        default='merged_out.csv')
    parser.add_argument('--ncpus', type=int,
                        help='Number of CPUs/cores to use for parallelization.',
                        default=1)
    parser.add_argument('--testsamples', action='store_true', help='Only process two samples from each batch.')
    parser.add_argument('--testbatch', action='store_true', help='Only process twp samples from one batch.')
    parser.add_argument('--matchingonly', action='store_true', help='Only perform sample matching, to validate metadata.')
    parser.add_argument('--feather', action='store_true', help='Store as feather as oposed to CSV')
    parser.add_argument('--utils', default='/home/agartlan/gitrepo/utils', help='Location of agartland/utils repo from public github.com')
    
    args = parser.parse_args()

    try:
        import parmap
        from multiprocessing import Pool
        _PARMAP = True
    except:
        _PARMAP = False
        print('Could not find package "parmap", parallelization not enabled.')

    import itertools
    import pandas as pd
    import numpy as np
    from os.path import join as opj
    import os
    from functools import partial
    import time
    import sys
    import feather
    
    """Make sure the utils are on path before importing"""
    sys.path.append(args.utils)

    # from ics import extractFunctionsGBY, extractFunctionsMarkersGBY, parseSubsets, mergeSamples, matchSamples
    from ics import *

    if args.matchingonly:
        metaDf = testMatching(args.folder)
        metaDf.to_csv(opj(args.folder, 'metamatch_' + args.out))
        print('Wrote matching metadata to %s.' % opj(args.folder, 'metamatch_' + args.out))
    else:
        subsets, markers, functions, exclude = parseSubsets(args.subsets)

        features = {'sparse_functions':(extractFunctionsGBY, dict(subsets=subsets,
                                                           functions=functions,
                                                           mincells=5)),
                    'bool_functions':(extractFunctionsGBY, dict(subsets=subsets,
                                                           functions=functions,
                                                           mincells=0)),

                    'functions_markers':(extractFunctionsMarkersGBY, dict(subsets=subsets,
                                                                       functions=functions,
                                                                       markers=markers,
                                                                       compressions=[('ALL', 2),
                                                                                    (['IFNg','IL2', 'TNFa'], 2)])),
                    'functions':(extractFunctionsGBY, dict(subsets=subsets,
                                                         functions=functions,
                                                         compressions=[('ALL', 1),
                                                                        ('ALL', 2),
                                                                        (['IFNg','IL2', 'TNFa'], 1),
                                                                        (['IFNg','IL2', 'TNFa'], 2),
                                                                        (['IFNg','IL2'], 1)])),
                   'cell_functions':(extractRawFunctions, dict(subsets=subsets, functions=functions, downsample=1))}
        
        extractionFunc, extractionKwargs = features[args.function]
        if args.testbatch:
            print('Test: processing samples from one batch')

        if args.testsamples:
            print('Test: processing two samples per batch')
        
        outFile = opj(args.folder, args.out)
        
        if args.feather:
            outFile = outFile.replace('.csv', '.feather')
        wrote = mergeBatches(args.folder,
                          extractionFunc=extractionFunc,
                          extractionKwargs=extractionKwargs,
                          testsamples=args.testsamples,
                          testbatch=args.testbatch,
                          outFile=outFile,
                          metaCols=['PTID', 'VISITNO', 'Global.Spec.Id', 'TESTDT', 'STIM'],
                          filters={'STIM':['negctrl', 'TB WCL', 'BCG-Pasteur', 'Ag85B', 'TB 10.4'], 'VISITNO':[2, 6, 7, 10, 11, 12]},
                          useFeather=int(args.feather),
                          ncpus=args.ncpus)

        if wrote == outFile:
            print('Wrote extracted data to %s.' % outFile)
        else:
            print('Error writing file to disk: %s' % wrote)